#include "cudatools.h"
#include "sgemm.h"
#include "matrix.h"
#include <chrono>
#include <iostream>

decltype(std::chrono::high_resolution_clock::now()) start;
decltype(std::chrono::high_resolution_clock::now()) stop;

inline void timing_start() {
	start = std::chrono::high_resolution_clock::now();
}

inline long timing_stop() {
	stop = std::chrono::high_resolution_clock::now();
	auto delta_t = std::chrono::duration_cast<std::chrono::microseconds>(stop - start); // fix type issue
	return delta_t.count();
}

constexpr uint N = 2048;
constexpr uint blocksize = 32; // length of one side!!!! -> blockDim = blocksize ** 2
static_assert(N%blocksize == 0);
constexpr uint BN = 64;
static_assert(N%BN == 0);
constexpr uint BK = 8;
static_assert(N%BK == 0);
int main () {

	Matrix A(N), B(N), C(N);
	//A.fill_random(0);
	A.fill_ones();
	A.upload();
	//B.fill_random(1);
	B.fill_ones();
	B.upload();
	//C.fill_random(2);
	C.fill_zeros();
	C.upload();


	dim3 gridDim(N/blocksize, N/blocksize, 1);
	dim3 blockDim(blocksize*blocksize, 1, 1);

	float alpha = 1;
	float beta = 0;
	unsigned reps = 1;
	std::cout << "STARTED TIMING" << std::endl;
	timing_start();
	for (unsigned rep = 0; rep < reps; rep++) {
		//sgemm_coalesced<float, N, blocksize> <<< gridDim , blockDim >>> (C.d_data, A.d_data, B.d_data, alpha, beta);
		//sgemm_sharedmem<float, N, blocksize> <<< gridDim , blockDim >>> (C.d_data, A.d_data, B.d_data, alpha, beta);
		sgemm_1D_blocktiling<float, N, BN, BK> <<<dim3(N/BN, N/BN, 1), dim3(BN*BK, 1, 1)>>> (C.d_data, A.d_data, B.d_data, alpha, beta);
		CLCE();
		CCE(cudaDeviceSynchronize());
	}
	unsigned microsecs = timing_stop();
	std::cout << "BANDWIDTH (MByte/s): " << sizeof(float)*reps*(4*N*N)/(double)microsecs << std::endl;
	CLCE();


	// check for correctness
	uint i = 1;
	uint j = 0;
	std::cout << Matrix::matmul(A, B, i, j) << std::endl;
	C.download();
	std::cout << C.get(i, j) << std::endl;
}
