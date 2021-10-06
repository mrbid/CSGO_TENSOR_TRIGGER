# CSGO_TENSOR_TRIGGER
A series of machine learning trigger bots for Counter-Strike: Global Offensive (CS:GO).

This repository holds the best releases from a series of articles I made that document my research into making a CS:GO auto-trigger bot using machine learning: https://james-william-fletcher.medium.com/list/fps-machine-learning-autoshoot-bot-for-csgo-100153576e93

**This bot is designed to target only player heads of counter-terrorists models.**

`GOBOT9_CNN` is the final CNN I produced, I have supplied what I believe to be the best model trained for the computational cost vs accuracy tradeoff. A more complete release, incuding various other trained models and the 192x96 version not included in this repository can be obtained [here](https://mega.nz/file/GvxXHCCB#yph08_eQ2jrb_ptXiKKJwXdcggfXPTILKMljBe31FI4) and the specifics of this release are documented in [this article.](https://james-william-fletcher.medium.com/creating-a-machine-learning-auto-shoot-bot-for-cs-go-part-6-af9589941ef3)

This is very similar to the [QUAKE3_TENSOR_TRIGGER](https://github.com/mrbid/QUAKE3_TENSOR_TRIGGER) project only that the sample normalisation is not 0-1 scaled but mean normalised which for this particular purpose had a much better result. It goes to show that how you normalise your input data can have a profound impact upon your model, so don't just assume - test the water.

I am working on one final release of this bot currently, in the 28x28 sample form, but it is a side project that I am making slow incremental progress with, parly because I am generating a completely new dataset for it which is a slow and laborious process.

Gabe has been very curtious concerning this project, I have noticed a new player model was since added to CS:GO which features a player helmet overloaded with spy gadgets which I believe had been added to throw a spoke into the wheel of computer vision based bots such as this _(or they are running slim on ideas)_. He has not officially commented on this project, but I imagine if he did he would say something like "Don't make cheats, make games, it's more rewarding." or maybe, he has commented this in the past, to me, personally, which I ignored _(when he said that, he didn't know what kind of trash games I would go on to create.. check out [snowball](https://github.com/mrbid/Snowball.mobi))_. Anyway, this bot certainly does not have response times faster than a human player so it's basically a non-threat to online matches. Created for research purposes, funded by Gabe's beard.

---

[`GOBOT9_CNN`](https://github.com/mrbid/CSGO_TENSOR_TRIGGER/tree/main/GOBOT9_CNN) - The best release of the bot so far.<br>
[`GOBOT7_FNN`](https://github.com/mrbid/CSGO_TENSOR_TRIGGER/blob/main/gobot7_fnn.c) - An all-in-one FNN, not as good as the CNN but it's cute.<br>

[`GOBOT_BINS`](https://github.com/mrbid/CSGO_TENSOR_TRIGGER/tree/main/GOBOT_BINS) - A selection of pre-compiled linux binaries for each incremental version of the bot spanning two datasets.<br>
- [`DATASET_V1`](https://github.com/mrbid/CSGO_TENSOR_TRIGGER/tree/main/GOBOT_BINS/DATASET_V1) - The V1 dataset was ~300 samples in size. ([download](https://github.com/TFCNN/Projects/blob/main/counter_terrorist_dataset_and_weights.zip))<br>
- [`DATASET_V2`](https://github.com/mrbid/CSGO_TENSOR_TRIGGER/tree/main/GOBOT_BINS/DATASET_V2) - The V2 dataset was ~750 target samples and ~1900 non-target samples ([download](https://github.com/mrbid/DATASETS/raw/main/CSGO.zip))

---

Terrorising Valve Software since 2003 with no legal reprecussions. God bless Gabe.

![Image of Gabe](https://static.wikia.nocookie.net/mlg-parody/images/3/39/Gabe_newell_meme-580x334.jpg/revision/latest/scale-to-width-down/580?cb=20190811113643)
