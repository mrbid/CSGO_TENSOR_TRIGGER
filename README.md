# CSGO_TENSOR_TRIGGER-
A series of machine learning trigger bots for Counter-Strike: Global Offensive (CS:GO).

This repository holds the best releases from a series of articles I made that document my research into making a CS:GO auto-trigger bot using machine learning:<br>
https://james-william-fletcher.medium.com/list/fps-machine-learning-autoshoot-bot-for-csgo-100153576e93

GOBOT9_CNN is the final CNN I produced, I have supplied what I believe to be the best model trained for the computational cost vs accuracy tradeoff. A more complete release, incuding various other trained models and the 192x96 version not included in this repository can be obtained [here.](https://mega.nz/file/GvxXHCCB#yph08_eQ2jrb_ptXiKKJwXdcggfXPTILKMljBe31FI4) and the specifics of this release are documented in [this article.](https://james-william-fletcher.medium.com/creating-a-machine-learning-auto-shoot-bot-for-cs-go-part-6-af9589941ef3)

This is very similar to the [QUAKE3_TENSOR_TRIGGER](https://github.com/mrbid/QUAKE3_TENSOR_TRIGGER) project only that the sample normalisation is not 0-1 scaled but mean normalised which for this particular purpose had a much better result. It goes to show that how you normalise your input data can have a profound impact upon your model, so don't just assume - test the water.

I am working on one final release of this bot currently, in the 28x28 sample form, but it is a side project that I am making slow incremental progress with.