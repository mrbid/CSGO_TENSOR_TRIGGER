This uses Keras2c: https://github.com/f0uriest/keras2c<br>
Described here: https://control.princeton.edu/wp-content/uploads/sites/418/2021/04/conlin2021keras2c.pdf

Keras2c seems to have a slight precision loss over using ONNX or Keras directly, and runs at roughly half the performance of what ONNX would in respect to model predictions per second but it's still very performant running at around 200-250 FPS where was ONNX could easily hit 400-500 FPS.

The benefit of using Keras2c is that it is a more portable solution, it requires less dependencies and setup time, you can just install `clang` and `libx11-dev` from your package manager, execute `compile.sh` and the resultant binary will just run.

I have also provided pre-compiled binaries that should work.
