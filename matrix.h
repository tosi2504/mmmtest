#pragma once

#include "cudatools.h"
#include <random>
#include <cassert>

struct Matrix {
	const int N;
	float * h_data; 
	float * d_data;

	Matrix(int N):
		N(N)
	{
		h_data = new float[N*N];
		CCE(cudaMalloc(&d_data, sizeof(float)*N*N));
	}
	~Matrix() {
		delete[] h_data;
		CCE(cudaFree(d_data));
	}

	float get(uint i, uint j) const {
		return h_data[i*N + j];
	}
	float & get(uint i, uint j) {
		return h_data[i*N + j];
	}

	void upload() {
		CCE(cudaMemcpy(d_data, h_data, sizeof(float)*N*N, cudaMemcpyHostToDevice));
	}
	void download() {
		CCE(cudaMemcpy(h_data, d_data, sizeof(float)*N*N, cudaMemcpyDeviceToHost));
	}

	void fill_random(unsigned seed) {
		std::default_random_engine rng(seed);
		std::uniform_real_distribution<float> dist(0.0, 1.0);
		for (int i = 0; i < N*N; i++) h_data[i] = dist(rng);
	}
	void fill_ones() {
		for (int i = 0; i < N*N; i++) h_data[i] = 1;
	}
	void fill_zeros() {
		for (int i = 0; i < N*N; i++) h_data[i] = 0;
	}

	static float matmul(const Matrix & A, const Matrix & B, uint i, uint j) {
		assert(A.N == B.N);
		uint N = A.N;
		float res = 0;
		for (uint k = 0; k < N; ++k) {
			res += A.h_data[i*N + k] * B.h_data[k*N + j];
		}
		return res;
	} 
}; 
