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
	constexpr uint TN = BN / BK; // I dont think this is true lol ... wait it is

	const uint cRow = blockIdx.y;
	const uint cCol = blockIdx.x;

	const uint aRow = threadIdx.x/BK;
	const uint aCol = threadIdx.x%BK;
	
	const uint bRow = threadIdx.x/BN;
	const uint bCol = threadIdx.x%BN;

	const uint threadRow = bRow;
	const uint threadCol = bCol;
	
	// okay lets set the starting points
	C += cRow*BN*N + cCol*BN;
	A += cRow*BN*N;
	B += cCol*BN;

	// statically allocate SMEM
	__shared__ T sA[numThreads];			// this is a BN X BK matrix
	__shared__ T sB[numThreads];			// this is a BK X BN matrix

	// buffer for results -> lies in registries
	T threadResults[TN] = {0.0};

	// lets define the slow-running k-loop 
	for (uint k = 0; k < N; k+=BK) { // K/BK iterations
		// fill shared memory and advance pointers
		sA[threadIdx.x] = A[aRow*N + aCol]; // 1 access to GMEM
		sB[threadIdx.x] = B[bRow*N + bCol]; // 1 access to GMEM
		__syncthreads();
		A += BK;
		B += BK*N;

		// okay what now
		// inner k-loop and loop over threadResults and somehow buffer B
		for (uint delta_k = 0; delta_k < BK; delta_k++) { // BK iterations (8)
			T bTmp = sB[delta_k*BN + threadCol]; // 1 access to SMEM
			for (uint resIdx = 0; resIdx < TN; resIdx++) { // TN iterations (8)
				threadResults[resIdx] += sA[(threadRow*TN + resIdx)*BK + delta_k] * bTmp; // 1 access to SMEM
			}
		}
		__syncthreads();
	}

	for (uint resIdx = 0; resIdx < TN; resIdx++) {
		uint cIdx = (threadRow*TN + resIdx)*N + threadCol;
		C[cIdx] = alpha*threadResults[resIdx] + beta*C[cIdx];
	}
}

template<class T, uint N, uint BN, uint BK, uint TN>
__global__ void sgemm_2D_blocktiling(T * C, const T * A, const T * B, T alpha, T beta) {
	static_assert(BN%TN == 0);
	constexpr uint numThreads = (BN*BN)/(TN*TN); // in our case maybe 
	static_assert(numThreads%BK == 0); // for SMEM loading of A: guaranteeing rectangular load tile
	static_assert(numThreads%BN == 0); // for SMEM loading of B
	static_assert((BN*BK)%numThreads == 0); // for SMEM loading: guaranteeing no overloading

	const uint cRow = blockIdx.y;
	const uint cCol = blockIdx.x;

	const uint threadRow = threadIdx.x / (BN/TN);
	const uint threadCol = threadIdx.x % (BN/TN);

	// advance pointers
	B += cCol*BN;
	A += cRow*BN*N;
	C += cRow*BN*N + cCol*BN;

	// allocate shared memory
	__shared__ T sA[BN*BK];
	__shared__ T sB[BK*BN];

	// allocate result array (in registers)
	T results[TN*TN] = {0.0}; 

	// allocate register caches
	T regACol[TN];
	T regBRow[TN];

	// outer k loop
	for (uint k = 0; k < N; k+=BK) {
		// need to fill shared memory
		for (uint offset = 0; offset < BN; offset += numThreads/BN) {
			sA[threadIdx.x + offset*BN] = A[(threadIdx.x/BN + offset)*N + threadIdx.x%BN];
		}
		for (uint offset = 0; offset < BK; offset += numThreads/BK) {
			sB[threadIdx.x + offset*BK] = B[(threadIdx.x/BK + offset)*N + threadIdx.x%BK];
		}
		__syncthreads();

		// advance pointers
		A += BN;
		B += BN*N;

		// inner k loop
		// TODO: CONTINUE HERE
		// TODO: explicit registry loading ;)
		for (uint delta_k = 0; delta_k < BK; delta_k++) {
			// populate registries to save smem accesses
			for (uint resIdxRow = 0; resIdxRow < TN; resIdxRow++) {
				regACol[resIdxRow] = sA[(threadRow*TN + resIdxRow)*BK + delta_k];
			}
			for (uint resIdxCol = 0; resIdxCol < TN; resIdxCol++) {
				regBRow[resIdxCol] = sB[delta_k*BN + threadCol*TN + resIdxCol];
			}
			for (uint resIdxRow = 0; resIdxRow < N; resIdxRow+=TN) {
				for (uint resIdxCol = 0; resIdxCol < N; resIdxCol++) {
					results[resIdxRow*TN+resIdxCol] += regACol[resIdxRow] * regBRow[resIdxCol];
				}
			}
		}
	}
	
	// write results into C
	for (uint resIdxRow = 0; resIdxRow < N; resIdxRow+=TN) {
		for (uint resIdxCol = 0; resIdxCol < N; resIdxCol++) {
			const uint cIdx = (threadRow*TN + resIdxRow)*N + threadCol*TN + resIdxCol;
			C[cIdx] = alpha * results[resIdxRow*TN + resIdxCol] + beta*C[cIdx];
		}
	}
}
