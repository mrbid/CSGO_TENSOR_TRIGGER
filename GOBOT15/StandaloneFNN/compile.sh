clang aim1.c -Ofast -mfma -lX11 -dn -lm -o aim1
strip --strip-unneeded aim1
upx --lzma --best aim1
clang aim2.c -Ofast -mfma -lX11 -dn -lm -o aim2
strip --strip-unneeded aim2
upx --lzma --best aim2
./aim1