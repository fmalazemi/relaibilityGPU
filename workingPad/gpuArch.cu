#include <iostream>
#include <cuda_runtime.h>

int main() {
    int count = 0;
    cudaGetDeviceCount(&count);

    for (int i = 0; i < count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        std::cout << "Device " << i << ": " << prop.name << "\n";
        std::cout << "Compute capability: " 
                  << prop.major << "." << prop.minor << "\n";
        std::cout << "CUDA arch flag: sm_" 
                  << prop.major << prop.minor << "\n\n";
    }
    return 0;
}
