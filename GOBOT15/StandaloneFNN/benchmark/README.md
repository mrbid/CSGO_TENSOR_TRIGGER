### bench1.c
- Column Major is faster with -Ofast enabled.
- Row Major is slightly faster without -Ofast enabled.

This is due to Row Major using branching.

### bench2.c
Branching removed from Row Major, and biases, just to stream line it a little. Row major code is now slightly faster.
