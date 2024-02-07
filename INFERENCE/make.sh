g++ -std=c++17 -Ofast -shared -fPIC src/batch_gemm.cpp -o libbatch_gemm.so -lopenblas
g++ -std=c++17 -Ofast -shared -fPIC src/conv_utils.cpp -o libconv.so -lopenblas
g++ -std=c++17 -Ofast -shared -fPIC src/pool_utils.cpp -o libpool.so -lopenblas
# g++ -std=c++17 -Ofast -fopenmp -shared -fPIC src/fastconv.cpp -o libfastconv.so -lopenblas

g++ -std=c++17 -I/home/a/LIB/openblas_0326/include -Ofast -fopenmp -shared -fPIC src/fastconv.cpp -o libfastconv.so \
-L/home/a/LIB/openblas_0326/lib -lopenblas
export LD_LIBRARY_PATH=/home/a/LIB/openblas_0326/lib:$LD_LIBRARY_PATH

MKL=mkl-static-lp64-iomp
g++ -std=c++17 -DMKL $(pkg-config --cflags ${MKL}) -Ofast -fopenmp -shared -fPIC src/fastconv.cpp \
-o libfastconv.so $(pkg-config --libs ${MKL})
