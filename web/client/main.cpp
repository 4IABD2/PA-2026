#include <cstring>
#include <iostream>
#include <netinet/in.h>
#include <sys/socket.h>
#include <cstdio>
#include <unistd.h>
#include <vector>

int main()
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

    // sending data
    const char* message = "0;0;0;0;0;0;0";
    send(clientSocket, message, strlen(message), 0);

    // wait for a response from the server and print it
    char buffer[1024];
    ssize_t received = recv(clientSocket, buffer, sizeof(buffer) - 1, 0);


    // output
    std::vector<float> output = {};
    char* token = strtok(buffer, ";");
    while (token != nullptr)
    {
        output.push_back(std::stof(token));
        token = strtok(nullptr, ";");
    }
    for (int i = 0; i < output.size(); i++)
    {
        std::cout << output[i] << std::endl;
    }


    // closing socket
    close(clientSocket);

    return 0;
}
