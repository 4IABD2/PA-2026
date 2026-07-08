# build application

```bash
mkdir build 
cd build
cmake ..
make
```

# run application

```bash
./application
```

# Build lib for python

- install pybind11 

```bash
sudo apt install pybind11-dev
```

- build lib for python
```bash
mkdir build_py
cd build_py
cmake .. -G "Unix Makefiles" -Dpybind11_DIR="$(uv run -m pybind11 --cmakedir)"
make
```

# test python lib

```bash
uv run lanch_request.py
```