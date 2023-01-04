clang aim.c -Ofast -mfma -L . -I . -lonnx -lX11 -lm -o aim
strip --strip-unneeded aim
upx --lzma --best aim
./aim