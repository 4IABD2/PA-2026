#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "request_client.h"

namespace py = pybind11;

PYBIND11_MODULE(client_bindings, m) {
    m.doc() = "Python bindings for launchRequest";
    m.def("launch_request", &launchRequest, "Send input vector to socket server and return output vector");
}