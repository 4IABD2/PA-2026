import sys

sys.path.insert(0, "build_py")
import client_bindings
import client_bindings
import random


def launch_request(input_data):
    result = client_bindings.launch_request(input_data)
    return result


if __name__ == "__main__":
    print(launch_request([random.randint(1, 100) / 10 for x in range(7)]))
