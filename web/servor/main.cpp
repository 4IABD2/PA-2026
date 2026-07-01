#include <cstring>
#include <iostream>
#include <netinet/in.h>
#include <sys/socket.h>
#include <vector>
#include <unistd.h>
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <memory>

using namespace std;

// Global ONNX Runtime objects (initialized once at startup)
Ort::Env g_env(ORT_LOGGING_LEVEL_WARNING, "inference_server");
unique_ptr<Ort::Session> g_session;
const char* g_input_name = nullptr;
const char* g_output_name = nullptr;
vector<int64_t> g_input_shape;
size_t g_input_tensor_size = 0;

// Initialize ONNX model
bool initializeModel(const char* model_path)
{
    try
    {
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        g_session = make_unique<Ort::Session>(g_env, model_path, session_options);

        Ort::AllocatorWithDefaultOptions allocator;

        auto input_name_allocated = g_session->GetInputNameAllocated(0, allocator);
        auto output_name_allocated = g_session->GetOutputNameAllocated(0, allocator);

        g_input_name = input_name_allocated.release();
        g_output_name = output_name_allocated.release();

        auto input_type_info = g_session->GetInputTypeInfo(0);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto input_shape = input_tensor_info.GetShape();

        g_input_shape = input_shape;

        // Resolve dynamic dimensions
        for (auto& dim : g_input_shape)
        {
            if (dim == -1)
            {
                dim = 1;
            }
        }

        g_input_tensor_size = 1;
        for (auto dim : g_input_shape)
        {
            g_input_tensor_size *= static_cast<size_t>(dim);
        }

        cout << "Model loaded successfully.\n";
        cout << "Input name: " << g_input_name << "\n";
        cout << "Output name: " << g_output_name << "\n";
        cout << "Input shape: [";
        for (size_t i = 0; i < g_input_shape.size(); ++i)
        {
            cout << g_input_shape[i];
            if (i + 1 < g_input_shape.size()) cout << ", ";
        }
        cout << "]\n";

        return true;
    }
    catch (const Ort::Exception& e)
    {
        cerr << "Failed to initialize model: " << e.what() << "\n";
        return false;
    }
}

// Run inference on input data
vector<float> runInference(const vector<float>& input_data)
{
    try
    {
        vector<float> input_tensor_values = input_data;

        // Pad with zeros if needed
        while (input_tensor_values.size() < g_input_tensor_size)
        {
            input_tensor_values.push_back(0.0f);
        }

        // Trim if too large
        input_tensor_values.resize(g_input_tensor_size);

        Ort::MemoryInfo memory_info =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor_values.data(), g_input_tensor_size,
            g_input_shape.data(), g_input_shape.size());

        const char* input_names[] = {g_input_name};
        const char* output_names[] = {g_output_name};

        auto output_tensors = g_session->Run(Ort::RunOptions{nullptr},
                                             input_names, &input_tensor, 1,
                                             output_names, 1);

        vector<float> result;

        if (!output_tensors.empty() && output_tensors[0].IsTensor())
        {
            float* output_data = output_tensors[0].GetTensorMutableData<float>();
            auto output_info = output_tensors[0].GetTensorTypeAndShapeInfo();
            auto output_shape = output_info.GetShape();

            size_t output_size = 1;
            for (auto dim : output_shape)
            {
                output_size *= static_cast<size_t>(dim);
            }

            result.assign(output_data, output_data + output_size);
        }

        return result;
    }
    catch (const Ort::Exception& e)
    {
        cerr << "Inference error: " << e.what() << "\n";
        return {};
    }
}

int main()
{
    // Initialize model at startup
    const char* model_path = "./best_model.onnx";
    if (!initializeModel(model_path))
    {
        cerr << "Failed to initialize model\n";
        return 1;
    }

    // Creating socket
    int serverSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (serverSocket < 0)
    {
        cerr << "Socket creation error\n";
        return 1;
    }

    // Specifying the address
    sockaddr_in serverAddress;
    serverAddress.sin_family = AF_INET;
    serverAddress.sin_port = htons(8080);
    serverAddress.sin_addr.s_addr = INADDR_ANY;

    // Binding socket
    if (bind(serverSocket, (struct sockaddr*)&serverAddress,
             sizeof(serverAddress)) < 0)
    {
        cerr << "Bind error\n";
        return 1;
    }

    // Listening to the assigned socket
    if (listen(serverSocket, 5) < 0)
    {
        cerr << "Listen error\n";
        return 1;
    }

    cout << "Server listening on port 8080...\n";

    // Accepting multiple connections
    while (true)
    {
        int clientSocket = accept(serverSocket, nullptr, nullptr);
        if (clientSocket < 0)
        {
            cerr << "Accept error\n";
            continue;
        }

        // Receiving data
        char buffer[1024] = {0};
        recv(clientSocket, buffer, sizeof(buffer), 0);
        cout << "Message from client: " << buffer << endl;

        // Parsing input data
        vector<float> input_data = {};
        char buffer_copy[1024];
        strcpy(buffer_copy, buffer);

        char* token = strtok(buffer_copy, ";");
        while (token != nullptr)
        {
            try
            {
                input_data.push_back(std::stof(token));
            }
            catch (...)
            {
                cerr << "Error parsing input: " << token << "\n";
            }
            token = strtok(nullptr, ";");
        }

        cout << "Parsed " << input_data.size() << " input values\n";

        // Run inference
        vector<float> output_data = runInference(input_data);

        // Sending data
        string output_str = "";
        for (size_t i = 0; i < output_data.size(); i++)
        {
            output_str += to_string(output_data[i]);
            if (i != output_data.size() - 1)
            {
                output_str += ";";
            }
        }

        if (!output_str.empty())
        {
            send(clientSocket, output_str.c_str(), output_str.size(), 0);
        }

        // Closing the client socket
        close(clientSocket);
    }

    close(serverSocket);
    return 0;
}
