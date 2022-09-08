### This version converts input images to greyscale before training and now only saves dataset sample images in greyscale, this reduces the dataset size by a factor of 3 while retaining the same rates of accuracy. This also streamlines the process of passing data to and from the C client to some extent.

This uses DATASET_V5. _(7,683 targets & 6,581 nontargets - 14,264 samples total, ~4k more than DATASET_V4)_

The best way to improve or expand upon this dataset is to enable sample capture and once you finish a few rounds go through the `/targets` directory created on your desktop and remove every well centered enemy head/body from the folder and place it into a directory of targets, all the files left behind will be of nontargets or non centered heads/bodies of enemies, copy these into your nontargets dir.

It's easier to seperate targets from nontargest than vice-versa because I find my human brain can identify targets faster than it can identify random nontarget images.

Also this process helps to reduce any missfire that exists in the bot, honing in on it's accuracy, because we are adding only samples of nontargets that it thought are targets.

Collecting dataset samples can be as simple as just letting a match of bots play out while you spectate with this bot enabled and sample capture enabled.

### prerequisites 
```
sudo apt install clang xterm espeak python3 python3-pip libx11-dev
sudo pip3 install --upgrade pip
sudo pip3 install tensorflow-cpu
sudo pip3 install --upgrade tensorflow-cpu
```

### exec
```
./RUN_CNN.sh
```