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
