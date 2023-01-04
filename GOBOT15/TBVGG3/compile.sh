clang aim.c -Ofast -mfma -lX11 -lm -o aim
strip --strip-unneeded aim
upx --lzma --best aim
./aim

