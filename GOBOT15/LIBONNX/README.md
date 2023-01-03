This is using [libonnx](https://github.com/xboot/libonnx).

libonnx is less performant than Keras2c, libonnx performs around 150 FPS while Keras2c is around 200-250 FPS.

The Keras model was converted to ONNX using [tf2onnx](https://github.com/onnx/tensorflow-onnx) and then I used [Netron](https://github.com/lutzroeder/netron) to extract the input and output names from the ONNX model.

