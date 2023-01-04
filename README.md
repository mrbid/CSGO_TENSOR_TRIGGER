# CSGO_TENSOR_TRIGGER
A series of machine learning trigger bots for Counter-Strike: Global Offensive (CS:GO).

Seems this dataset also works to some extent on CSS and 1.6 / CZ, to be honest it will probably have similar results on any game with humanoid characters.

**GOBOT12 Video:** https://youtu.be/R-nCL5zqZBQ<br>
**GOBOT11 Video:** https://youtu.be/UMBqk8CAe04

- A [Keras2c](https://github.com/f0uriest/keras2c) builds have been added [here.](GOBOT15/KERAS2C)
- A [libonnx](https://github.com/xboot/libonnx) build of the Keras CNN is [here.](GOBOT15/LIBONNX)
- A [TBVGG3](https://github.com/TFNN/TBVGG3) version of the CNN is [here.](GOBOT15/TBVGG3)

## Crosshair

The game crosshair will get in the way of the bots ability to detect player models on the screen, you have a few options to mitigate this, each option is just as effective as the other:
- Change your crosshair settings in CS:GO to a single green pixel center dot with no outline.
- Set your CS:GO crosshair to invisible by enabling the alpha channel and sliding it to invisible.
  - Then use the crosshair provided by the bot which encapsulates the scan region with a square.
  - Or if your monitor supplies a built-in crosshair, use that. _(because it does not get written to the game renders, it's a hardware overlay)_

## Tips 'n Tricks

The bot generally regulates your rate of fire for you but if you need more controlled bursts enable sample capture, this will ensure shots are fired in single rhythmic bursts. _(it's a side effect from ensuring duplicate samples of the same frame are not captured)_

## Information

This repository holds the best releases from a series of articles I made that document my research into making a CS:GO auto-trigger bot using machine learning: https://james-william-fletcher.medium.com/list/fps-machine-learning-autoshoot-bot-for-csgo-100153576e93

Latest datasets are now maintained over at [TFNN/DOCS/DATASETS](https://github.com/TFNN/DOCS/tree/main/DATASETS).
