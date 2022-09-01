## Latest FNN Release [FGOLD3]

The latest release is a Standalone FNN which will only activate trigger mode when the player is stationary, it detects a stationary state by checking if any of the W,A,S,D keys are currently pressed. This ensures there is reduced spread when the bot fires but also that there is much less misfire when traveling around the map. This solution is not only lighter on the CPU but it is also a much more responsive, in my tests I gauge it sampling at ~120 FPS and because an FNN is more generalised it will not hesitate to fire as often as the CNN. Highly configurable to suite your needs. [`Source`](https://github.com/mrbid/CSGO_TENSOR_TRIGGER/blob/main/StandaloneSrc/csgo_gold3_fnn.c) [`Linux Binary`](https://github.com/mrbid/CSGO_TENSOR_TRIGGER/raw/main/GOBOT_BINS/DATASET_V4/fgold3)

**Prerequisites**
- `sudo apt install libxdo-dev libxdo3 libespeak1 libespeak-dev espeak`

**Notices**
- Disable crosshair or make the crosshair a single green pixel, or use monitor overlay crosshair.
- This bot will only auto trigger when stationary _(W,A,S,D are not being pressed)_.

**General user keybinds**
```
L-CTRL + L-ALT = Toggle BOT ON/OFF
R-CTRL + R-ALT = Toggle HOTKEYS ON/OFF

TAB + 1-7 = Select Single Model Mode (worst to best)

L-SHIFT + 1 = Desperate Mode (all models)
L-SHIFT + 2 = Confidence Mode (all models averaged)
L-SHIFT + 3 = Toggle Misfire Reduction (medium, high, off)

P = Toggle crosshair
```

**Dev keybinds**
```
L = Toggle sample capture
Q = Capture non-target sample
G = Get activation for reticule area.
H = Get scans per second.
```

## FNN vs CNN

While a CNN is generally better, an FNN has a considerable reduction in complexity and performs nearly as well or even better in some cases. A CNN can run on a CPU but will take more resources than an FNN and while a CNN can run on a GPU more efficiently, we're entering an era now where users are more likely to want to utilise a single CPU core that is being unused than to dedicate some GPU resources. A CNN will be better at detecting an enemy reliably with fast response times, but it will also detect the enemy without locality which means the FNN is more likely to get a shot in the head as where a CNN is more likely to fire and slightly miss a target.

Overall the benefit of a CNN can be seen as a trade-off, where the CNN excels, the FNN can excel to a different vantage, and when you consider the reduced complexity, an FNN is a ~3kb set of weights on each scan running faster on a CPU by utilising FMA and a CNN is ~500kb+ set of weights per scan and runs faster on a GPU.

On [Line 26](https://github.com/mrbid/CSGO_TENSOR_TRIGGER/blob/main/GOBOT12_CNN/Trainer/train.py#L26) of `Trainer/train.py` in [GOBOT12_CNN](https://github.com/mrbid/CSGO_TENSOR_TRIGGER/tree/main/GOBOT12_CNN) you can increase the number of kernels/filters per layer. By default this is set to 16 as it's an adequate tradeoff of compute resources to minimal misfire. As you increase this value the compute resources tend to grow at a disproportionate rate to the reduction in misfire. Although if you were to increase this to 64, or even 256 or more _(stick to powers of 2)_, you basically end up with, practically no misfire.

The CNN is the best solution, if you have the technical know-how to set this up to utilise the GPU with Tensorflow you should be fine to run this at whatever cost on your GPU without any impact on framerate _(within reason)_. The FNN running on the CPU in comparison will never be as good as the CNN. But if you are after a lightweight alternative to the CNN, or just don't know how to setup Tensorflow, the FNN is almost as good as the CNN at 16 kernels per layer.
