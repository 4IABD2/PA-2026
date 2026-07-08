#  Install onnxruntime

- use : https://dev.to/wolfram27/setting-up-and-using-onnx-runtime-for-c-in-linux-1ho9


# build application

```bash
mkdir build 
cd build
cmake ..
make
```

- put `best_model.onnx` in `build` directory before running application

# run application

```bash
cp ../best_model.onnx ./best_model.onnx
cp ../best_model.onnx.data ./best_model.onnx.data
./application
```