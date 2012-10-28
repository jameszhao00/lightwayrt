#pragma once

#include "cuda_runtime_api.h"

#include "Stats.h"
struct Vec3Buffer;
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
	void execute(int iteration_idx, int width, int height, bool bdpt_debug, Stats* stats);

	void* scene_ptr;
	void* pass_ptr;
	cudaGraphicsResource* framebuffer_resource;
	size_t output_byte_size;

	Vec3Buffer* new_buffer;
	Vec3Buffer* existing_buffer;

	Vec3Buffer* light_vtx_pos;
	Vec3Buffer* light_vtx_normal;
	Vec3Buffer* light_throughput;
	int* light_vtx_material_id_ptr;
	Vec3Buffer* light_wi;

	Vec3Buffer* eye_vtx_pos;
	Vec3Buffer* eye_vtx_normal;
	Vec3Buffer* eye_throughput;
	int* eye_vtx_material_id_ptr;
	Vec3Buffer* eye_wi;
};
