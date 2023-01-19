clang aim1.c -Ofast -mfma -lX11 -lm -o aim1
strip --strip-unneeded aim1
upx --lzma --best aim1
clang aim2.c -Ofast -mfma -lX11 -lm -o aim2
strip --strip-unneeded aim2
upx --lzma --best aim2
clang aim3.c -Ofast -mfma -lX11 -lm -o aim3
strip --strip-unneeded aim3
upx --lzma --best aim3
./aim1
