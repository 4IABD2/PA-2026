#pragma once
#include <vector>
#include <cstring>
#include <iostream>
#include <netinet/in.h>
#include <sys/socket.h>
#include <cstdio>
#include <unistd.h>
#include <arpa/inet.h>

std::vector<float> launchRequest(std::vector<float> input);
