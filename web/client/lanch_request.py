import sys

sys.path.insert(0, "/home/victor/Documents/esgi/pa/PA-2026/web/client/build_py")
import pprint
print('executable:', sys.executable)
pprint.pprint(sys.path)

import client_bindings
import random


def launch_request(input_data):
    result = client_bindings.launch_request(input_data)
    return result


if __name__ == "__main__":
    print(launch_request([random.randint(1, 100) / 10 for x in range(7)]))
