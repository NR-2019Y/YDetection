g++ -std=c++17 -Ofast -shared -fPIC src/batch_gemm.cpp -o libbatch_gemm.so -lopenblas
g++ -std=c++17 -Ofast -shared -fPIC src/conv_utils.cpp -o libconv.so -lopenblas
g++ -std=c++17 -Ofast -shared -fPIC src/pool_utils.cpp -o libpool.so -lopenblas
g++ -std=c++17 -Ofast -fopenmp -shared -fPIC src/fastconv.cpp -o libfastconv.so -lopenblas