**Video:** _coming soon_

**Walking among giants**<br>
On [Line 26](https://github.com/mrbid/CSGO_TENSOR_TRIGGER/blob/main/GOBOT12_CNN/Trainer/train.py#L26) of `Trainer/train.py` you can increase the number of kernels/filters per layer. By default this is set to 16 as it's an adequate tradeoff of compute resources to minimal missfire. As you increase this value the compute resources tend to grow at disproportionate rate to the reduction in missfire. Although if you were to increase this to 64, or even 256 or more, you basically end up with, practically no missfire.

This is the best solution, if you have have the techical know how to set this up to utilise the GPU you should be fine to run this at whatever cost on your GPU without any impact on framerate. The FNN running on the CPU in comparison will NEVER be as good as the CNN. But if you are after a light weight alternative to the CNN, the FNN is almost as good as the CNN at 16 kernels per layer.

**Instructions**<br>
First run Trainer/train.py to generate a model, then run PredictBot/exec.sh to test it. Check the output of the predict bot as it doesn't always launch correctly first time and may need to be terminated and re-executed.

Not all systems will load the Keras model without errors, I'm not sure why because I assumed that was the point of exporting Keras models. If this is the case for you, then you will have to run the train.py file first to generate a new one on your system. (this could be due to different Keras versions, the model provided was trained using Keras 2.6.0)
