#pragma once

template<class T, uint N, uint blocksize>
__global__ void sgemm_coalesced(T * C, const T * A, const T * B, T alpha, T beta) {
	const uint i = blocksize*blockIdx.x + threadIdx.x/blocksize;
	const uint j = blocksize*blockIdx.y + threadIdx.x%blocksize;
	// j is fast running here in respect to inceasing threadIdx.x
	T tmp = 0.0;
	for (uint k = 0; k < N; k++) {
		tmp += A[i*N + k] * B[k*N + j];
	}
	C[i*N + j] = beta*C[i*N + j] + alpha*tmp;
}

template<class T, uint N, uint blocksize>
__global__ void sgemm_sharedmem(T * C, const T * A, const T * B, T alpha, T beta) {
	const uint i_t = threadIdx.x/blocksize;
	const uint j_t = threadIdx.x%blocksize;
	const uint i = blocksize*blockIdx.x + i_t;
	const uint j = blocksize*blockIdx.y + j_t;

	// statically allocate SMEM
	__shared__ T sA[blocksize*blocksize];
	__shared__ T sB[blocksize*blocksize];

	// okay, so outer loop and inner loop for k
	T tmp = 0.0;
	for (uint kk = 0; kk < N; kk+=blocksize) {
		// load block-matrix into shared memory
		sA[threadIdx.x] = A[i*N + kk+j_t];
		sB[threadIdx.x] = B[(kk+i_t)*N + j];
		__syncthreads();
		for (uint delta_k = 0; delta_k < blocksize; delta_k++) {
			// perform matmul on the block-matrix
			tmp += sA[i_t*blocksize + delta_k]*sB[delta_k*blocksize + j_t];
		}
		__syncthreads();
	}
	C[i*N + j] = alpha*tmp + beta*C[i*N + j];
}

template<class T, uint N, uint blocksize>
__global__ void sgemm_sharedmem_pointerarithm(T * C, const T * A, const T * B, T alpha, T beta) {
	const uint i_t = threadIdx.x/blocksize;
	const uint j_t = threadIdx.x%blocksize;

	// statically allocate SMEM
	__shared__ T sA[blocksize*blocksize];
	__shared__ T sB[blocksize*blocksize];

	// progress A and B to starting points
	A += blocksize*blockIdx.x*N;
	B += blocksize*blockIdx.y;
	C += blocksize*(blockIdx.x*N + blockIdx.y);

	// okay, so outer loop and inner loop for k
	T tmp = 0.0;
	for (uint kk = 0; kk < N; kk+=blocksize) {
		// load block-matrix into shared memory
		sA[threadIdx.x] = A[i_t*N + j_t];
		sB[threadIdx.x] = B[i_t*N + j_t];
		__syncthreads();

		A += blocksize;
		B += blocksize*N;

		for (uint delta_k = 0; delta_k < blocksize; delta_k++) {
			// perform matmul on the block-matrix
			tmp += sA[i_t*blocksize + delta_k]*sB[delta_k*blocksize + j_t];
		}
		__syncthreads();
	}
	C[i_t*N + j_t] = alpha*tmp + beta*C[i_t*N + j_t];
}

// WARNING: 
// assert(blockDim == BN * BK)
// assert(BN % BK == 0)
template<class T, uint N, uint BN, uint BK>
__global__ void sgemm_1D_blocktiling(T * C, const T * A, const T * B, T alpha, T beta) {
	static_assert(BN % BK == 0);
	constexpr uint numThreads = BN * BK;
	constexpr uint TM = BN / BK;

	const uint cRow = blockIdx.y;
	const uint cCol = blockIdx.x;

	const uint aRow = threadIdx.x/BK;
	const uint aCol = threadIdx.x%BK;
	
	const uint bRow = threadIdx.x/BN;
	const uint bCol = threadIdx.x%BN;
	
	// okay lets set the starting points
	C += cRow*BN*N + cCol*BN;
	A += cRow*BN*N;
	B += cCol*BN;

	// statically allocate SMEM
	__shared__ T sA[numThreads];
	__shared__ T sB[numThreads];

	// buffer for results -> lies in registries
	T threadResults[TM] = {0.0};

	// lets define the slow-running k-loop 
	for (uint k = 0; k < N; k+=BK) {
		// fill shared memory and advance pointers
		sA[threadIdx.x] = A[aRow*N + aCol];
		sB[threadIdx.x] = B[bRow*N + bCol];
		__syncthreads();
		A += BK;
		B += BK*N;

		// okay what now
		// inner k-loop and loop over threadResults and somehow buffer B
		for (uint delta_k = 0; delta_k < BK; delta_k++) {
			T bTmp = sB[delta_k*BN + threadIdx.x%BN];
			for (uint resIdx = 0; resIdx < TM; resIdx++) {
				threadResults[resIdx] += sA[((threadIdx.x/BN)*TM + resIdx)*BK + delta_k] * bTmp;
			}
		}
		__syncthreads();
	}

	for (uint resIdx = 0; resIdx < TM; resIdx++) {
		uint cIdx = ((threadIdx.x/BN)*TM + resIdx)*BN + threadIdx.x%BN;
		C[cIdx] += alpha*threadResults[resIdx] + beta*C[cIdx];
	}
}
