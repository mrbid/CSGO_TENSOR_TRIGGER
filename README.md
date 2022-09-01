# CSGO_TENSOR_TRIGGER
A series of machine learning trigger bots for Counter-Strike: Global Offensive (CS:GO).

Seems this dataset also work to some extent on CSS and 1.6 / CZ, to be honest it will probably have similar results on any game with humanoid characters.

**GOBOT12 Video:** https://youtu.be/R-nCL5zqZBQ<br>
**GOBOT11 Video:** https://youtu.be/UMBqk8CAe04

[Jim C. Williams](https://github.com/jcwml) has made a great implementation of this based on [TBVGG3](https://github.com/TFNN/TBVGG3) check it out [here](https://github.com/jcwml/CSGO-Trigger-Bot) and more recently [here](https://github.com/jcwml/CSGO-Trigger-Bot-2).

## Crosshair

The game crosshair will get in the way of the bots ability to detect player models on the screen, you have a few options to mitigate this, each option is just as effective as the other:
- Change your crosshair settings in CS:GO to a single green pixel center dot with no outline.
- Set your CS:GO crosshair to invisible by enabling the alpha channel and sliding it to invisible.
  - Then use the crosshair provided by the bot which encapsulates the scan region with a square.
  - Or if your monitor supplies a built-in crosshair, use that. _(because it does not get written to the game renders, it's a hardware overlay)_

## Tips 'n Tricks

The bot generally regulates your rate of fire for you but if you need more controlled bursts enable sample capture, this will ensure shots are fired in single rhythmic bursts. _(it's a side effect from ensuring duplicate samples of the same frame are not captured)_

## FNN vs CNN

While a CNN is generally better, an FNN has a considerable reduction in complexity and performs nearly as well or even better in some cases. A CNN can run on a CPU but will take more resources than an FNN and while a CNN can run on a GPU more efficiently, we're entering an era now where users are more likely to want to utilise a single CPU core that is being unused than to dedicate some GPU resources. A CNN will be better at detecting an enemy reliably with fast response times, but it will also detect the enemy without locality which means the FNN is more likely to get a shot in the head as where a CNN is more likely to fire and slightly miss a target.

Overall the benefit of a CNN can be seen as a trade-off, where the CNN excels, the FNN can excel to a different vantage, and when you consider the reduced complexity, an FNN is a ~3kb set of weights on each scan running faster on a CPU by utilising FMA and a CNN is ~500kb+ set of weights per scan and runs faster on a GPU.

On [Line 26](https://github.com/mrbid/CSGO_TENSOR_TRIGGER/blob/main/GOBOT12_CNN/Trainer/train.py#L26) of `Trainer/train.py` in [GOBOT12_CNN](https://github.com/mrbid/CSGO_TENSOR_TRIGGER/tree/main/GOBOT12_CNN) you can increase the number of kernels/filters per layer. By default this is set to 16 as it's an adequate tradeoff of compute resources to minimal misfire. As you increase this value the compute resources tend to grow at a disproportionate rate to the reduction in misfire. Although if you were to increase this to 64, or even 256 or more _(stick to powers of 2)_, you basically end up with, practically no misfire.

The CNN is the best solution, if you have the technical know-how to set this up to utilise the GPU with Tensorflow you should be fine to run this at whatever cost on your GPU without any impact on framerate _(within reason)_. The FNN running on the CPU in comparison will never be as good as the CNN. But if you are after a lightweight alternative to the CNN, or just don't know how to setup Tensorflow, the FNN is almost as good as the CNN at 16 kernels per layer.

## Information

This repository holds the best releases from a series of articles I made that document my research into making a CS:GO auto-trigger bot using machine learning: https://james-william-fletcher.medium.com/list/fps-machine-learning-autoshoot-bot-for-csgo-100153576e93

This is very similar to the [QUAKE3_TENSOR_TRIGGER](https://github.com/mrbid/QUAKE3_TENSOR_TRIGGER) project only that the sample normalisation was not 0-1 scaled but mean normalised up to DATASET_V2 _(apart from the GOLD_FNN)_ which then switched to using 0-1 scaled normalisation for bots using DATASET_V3+.

---

[`GOBOT15_CNN`](https://github.com/mrbid/CSGO_TENSOR_TRIGGER/tree/main/GOBOT15_CNN) - The best release so far.<br>
[`GOLD3_FNN`](https://github.com/mrbid/CSGO_TENSOR_TRIGGER/blob/main/StandaloneSrc/csgo_gold3_fnn.c) - An all-in-one FNN, good for low power pc's.<br>

[`GOBOT_BINS`](https://github.com/mrbid/CSGO_TENSOR_TRIGGER/tree/main/GOBOT_BINS) - A selection of pre-compiled linux binaries for each incremental release.<br>

Latest datasets are now maintained over at [TFNN/DOCS/DATASETS](https://github.com/TFNN/DOCS/tree/main/DATASETS).
