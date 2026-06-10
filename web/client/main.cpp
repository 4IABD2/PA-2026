#include <cstring>
#include <iostream>
#include <netinet/in.h>
#include <sys/socket.h>
#include <cstdio>
#include <unistd.h>
#include <vector>

#include "request_client.h"


int main()
{
    std::vector<float> input = {};
    for (int i = 0; i < 7; i++)
    {
        input.push_back(rand() % 10);
    }
    std::vector<float> output = launchRequest(input);

    for (int i = 0; i < output.size(); i++)
    {
        std::cout << output[i] << std::endl;
    }
    return 0;
}
