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

void checkMatmul(Matrix & C, const Matrix & A, const Matrix & B) {
	C.download();
	uint N = C.N;
	std::cout << "Start errorcheck:" << std::endl;
	bool doBreak = false;
	for (uint i = 0; i < N; i++) {
		for (uint j = 0; j < N; j++) {
			if (Matrix::matmul(A, B, i, j) != C.get(i,j)) {
				std::cout << "Error detected: " << std::endl;
				std::cout << "i: " << i << " j: " << j << std::endl;
				doBreak = true;
				break;
			}
		}
		if (doBreak) break;
	}
	if (not doBreak) std::cout << "No errors" << std::endl;
}

constexpr unsigned long N = 2048;
constexpr uint BN = 128; // 16*16 = 256 threads per block
static_assert(N%BN == 0);
constexpr uint BK = 8;
static_assert(N%BK == 0);
constexpr uint TN = 8;
int main () {
	Matrix A(N), B(N), C(N);
	A.fill_random(0);
	A.upload();
	B.fill_random(1);
	B.upload();
	C.fill_random(2);
	C.upload();


	dim3 gridDim(N/BN, N/BN, 1);
	dim3 blockDim((BN*BN)/(TN*TN), 1, 1);

	float alpha = 1;
	float beta = 0;
	unsigned reps = 50;
	std::cout << "STARTED TIMING" << std::endl;
	timing_start();
	for (unsigned rep = 0; rep < reps; rep++) {
		sgemm_2D_blocktiling<float, N, BN, BK, TN> <<< gridDim , blockDim >>> (C.d_data, A.d_data, B.d_data, alpha, beta);
		CLCE();
		CCE(cudaDeviceSynchronize());
	}
	unsigned microsecs = timing_stop();
	std::cout << "BANDWIDTH (MByte/s): " << sizeof(float)*reps*(4*N*N)/(double)microsecs << std::endl;
	std::cout << "ARITHETICS (GFLOPS/s) (TODO): " << reps*(2*N*N*N + N*N)/((double)microsecs*1000) << std::endl;
	CLCE();

	// check for correctness
	C.download();
	uint i = 123;
	uint j = 456;
	std::cout << "i: " << i << " j: " << j << std::endl;
	std::cout << Matrix::matmul(A, B, i, j) << " <---> " << C.get(i,j) << std::endl;
	// checkMatmul(C, A, B);
}
