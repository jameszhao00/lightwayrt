#pragma once

#include "cuda_runtime_api.h"

#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
	fprintf(stderr, "Error %s at line %d in file %s\n",					\
	cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
	exit(1);															\
	} }

void unit_test();
struct Kernel
{
	void setup(cudaGraphicsResource* output, int width, int height);
	void execute(int iteration_idx, int iterations, int bounces, int width, int height);

	void* camera_ptr;
	void* scene_ptr;
	void* pass_ptr;
	cudaGraphicsResource* framebuffer_resource;
	size_t output_byte_size;

	char* buffer;
};
