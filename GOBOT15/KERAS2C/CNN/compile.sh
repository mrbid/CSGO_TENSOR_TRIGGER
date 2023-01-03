clang aim.c processModel.c ../include/k2c_pooling_layers.c ../include/k2c_convolution_layers.c ../include/k2c_helper_functions.c ../include/k2c_core_layers.c ../include/k2c_activations.c -I ../include -Ofast -mfma -lX11 -dn -lm -o aim
strip --strip-unneeded aim
upx --lzma --best aim
./aim