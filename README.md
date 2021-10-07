# CSGO_TENSOR_TRIGGER
A series of machine learning trigger bots for Counter-Strike: Global Offensive (CS:GO).

## Latest Release

The latest release is a Standalone FNN which will only activate trigger mode when the player is stationary, it detects a stationary state by checking if any of the W,A,S,D keys are currently pressed. This ensures there is reduced spread when the bot fires but also that there is much less missfire when travelling around the map. This solution is not only lighter on the CPU but it is also a much more responsive, in my tests I gauge it sampling at ~120 FPS and because an FNN is more generalised it will not hesitate to fire as often as the CNN. [`Source`](https://github.com/mrbid/CSGO_TENSOR_TRIGGER/blob/main/StandaloneSrc/csgo_gold_fnn.c) [`Linux Binary`](https://github.com/mrbid/CSGO_TENSOR_TRIGGER/raw/main/GOBOT_BINS/DATASET_V2/GOLD_FNN/fgold)

The prerequisites are: `sudo apt install libxdo-dev libxdo3 libespeak1 libespeak-dev espeak`

**This one is actually disruptive to online gameplay.** If your CPU can handle it.

## Crosshair

The game crosshair will get in the way of the bots ability to detect player models on the screen, you have a few options to mitigate this, each option is just as effective as the other:
- Change your crosshair settings in CS:GO to a single green pixel center dot with no outline.
- Set your CS:GO crossair to invisible by enabling the alpha channel and sliding it to invisible.
  - Then use the crosshair provided by the bot which encapsulates the scan region with a square.
  - Or if your monitor supplies a built-in crosshair, use that. _(because it does not get written to the game renders, it's a hardware overlay)_

## Tips 'n Tricks

The bot generally regulates your rate of fire for you but if you need more controlled bursts enable sample capture, this will ensure shots are fired in single rythmatic bursts. _(its a side effect from ensuring duplicate samples of the same frame are not captured)_

## Information

This repository holds the best releases from a series of articles I made that document my research into making a CS:GO auto-trigger bot using machine learning: https://james-william-fletcher.medium.com/list/fps-machine-learning-autoshoot-bot-for-csgo-100153576e93

A more complete release of the DATASET_V2 models, incuding various other trained models and the 96x192 version not included in this repository can be obtained [here](https://mega.nz/file/GvxXHCCB#yph08_eQ2jrb_ptXiKKJwXdcggfXPTILKMljBe31FI4) and the specifics of this release are documented in [this article.](https://james-william-fletcher.medium.com/creating-a-machine-learning-auto-shoot-bot-for-cs-go-part-6-af9589941ef3) I wont be continuing the 96x192 line of bots.

This is very similar to the [QUAKE3_TENSOR_TRIGGER](https://github.com/mrbid/QUAKE3_TENSOR_TRIGGER) project only that the sample normalisation is not 0-1 scaled but mean normalised which for this particular purpose had a much better result for a clear detection, 0-1 scaled felt more like a probability range than a binary detection.

Gabe has been very curtious concerning this project, I have noticed a new player model was since added to CS:GO which features a player helmet overloaded with spy gadgets which I believe had been added to throw a spoke into the wheel of computer vision based bots such as this _(that or they are running slim on ideas)_. He has not officially commented on this project, but I imagine if he did he would say something like "Don't make cheats, make games, it's more rewarding." or maybe, he has commented this in the past, to me, personally, which I ignored _(when he said that, he didn't know what kind of trash games I would go on to create.. check out [snowball](https://snapcraft.io/snowball))_. Anyway, this bot certainly does not have response times faster than a human player so it's basically a non-threat to online matches. Created for research purposes, funded by Gabe's beard.

---

[`GOBOT11_CNN`](https://github.com/mrbid/CSGO_TENSOR_TRIGGER/tree/main/GOBOT11_CNN) - The best release of the bot so far. Deadly.<br>
[`GOLD_FNN`](https://github.com/mrbid/CSGO_TENSOR_TRIGGER/blob/main/StandaloneSrc/csgo_gold_fnn.c) - An all-in-one FNN, it's a ruthless killer.<br>

[`GOBOT_BINS`](https://github.com/mrbid/CSGO_TENSOR_TRIGGER/tree/main/GOBOT_BINS) - A selection of pre-compiled linux binaries for each incremental version of the bot spanning two datasets.<br>
- [`DATASET_V1`](https://github.com/mrbid/CSGO_TENSOR_TRIGGER/tree/main/GOBOT_BINS/DATASET_V1) - The V1 dataset is 768 samples in size, 302 target samples. _(Counter-Terrorist only)_ ([download](https://github.com/TFCNN/Projects/blob/main/counter_terrorist_dataset_and_weights.zip))<br>
- [`DATASET_V2`](https://github.com/mrbid/CSGO_TENSOR_TRIGGER/tree/main/GOBOT_BINS/DATASET_V2) - The V2 dataset is ~750 target samples and ~1900 non-target samples ([download](https://github.com/mrbid/DATASETS/raw/main/CSGO.zip))
- [`DATASET_V3`](https://github.com/mrbid/CSGO_TENSOR_TRIGGER/tree/main/GOBOT_BINS/DATASET_V3) - The V3 dataset is ~5,000 samples with an almost even split. ([download](https://github.com/mrbid/DATASETS/raw/main/CSGO3.zip))

---

Terrorising Valve Software since 2003 with no legal reprecussions. God bless Gabe.

![Image of Gabe](https://static.wikia.nocookie.net/mlg-parody/images/3/39/Gabe_newell_meme-580x334.jpg/revision/latest/scale-to-width-down/580?cb=20190811113643)
