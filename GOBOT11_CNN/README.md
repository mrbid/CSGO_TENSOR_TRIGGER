**Legacy**, check out [GOBOT12](https://github.com/mrbid/CSGO_TENSOR_TRIGGER/tree/main/GOBOT12_CNN) for a more up to date release.

**Video:** https://youtu.be/UMBqk8CAe04

**Instructions**<br>
First run `Trainer/train.py` to generate a model, then run `PredictBot/exec.sh` to test it. Check the output of the predict bot as it doesn't always launch correctly first time and may need to be terminated and re-executed.

Not all systems will load the Keras model without errors, I'm not sure why because I assumed that was the point of exporting Keras models. If this is the case for you, then you will have to run the `train.py` file first to generate a new one on your system. _(this could be due to different Keras versions, the model provided was trained using Keras 2.6.0)_
