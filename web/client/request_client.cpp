#include "request_client.h"


std::vector<float> launchRequest(std::vector<float> input)
{
    // creating socket
    int clientSocket = socket(AF_INET, SOCK_STREAM, 0);

    // specifying address
    sockaddr_in serverAddress;
    serverAddress.sin_family = AF_INET;
    serverAddress.sin_port = htons(8080);
    serverAddress.sin_addr.s_addr = INADDR_ANY;

    // sending connection request
    connect(clientSocket, (struct sockaddr*)&serverAddress,
            sizeof(serverAddress));


    // parsing input

    char message[1024] = {0};
    for (size_t i = 0; i < input.size(); i++)
    {
        char buffer[32];
        snprintf(buffer, sizeof(buffer), "%f", input[i]);
        strcat(message, buffer);
        if (i < input.size() - 1)
        {
            strcat(message, ";");
        }
    }


    // sending data
    send(clientSocket, message, strlen(message), 0);

    // wait for a response from the server and print it
    char buffer[1024];
    ssize_t received = recv(clientSocket, buffer, sizeof(buffer) - 1, 0);


    std::vector<float> output = {};
    char* token = strtok(buffer, ";");
    while (token != nullptr)
    {
        output.push_back(std::stof(token));
        token = strtok(nullptr, ";");
    }


    close(clientSocket);

    return output;
}