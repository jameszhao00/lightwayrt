#include <stdio.h>
#include <stdlib.h>
#include "util.h"
#include "bitmap_image.hpp"
#include "assert.h"

#define CATCH_CONFIG_RUNNER
#include "catch.hpp"

#include "validate_importance_sampling.h"

const int NUM_ITERATION = 3;
const int NUM_BOUNCES = 3;

__global__ void gfx_kernel(ref::glm::vec4 *data, const Camera* camera, int width, int height) {
	ref::glm::uvec2 screen_size(width, height);
	ref::glm::uvec2 xy = screen_xy();
	if(ref::glm::any(ref::glm::greaterThan(xy, screen_size - ref::glm::uvec2(1)))) return;
	Scene scene;
	color value(0,0,0);
	for(int iteration_idx = 0; iteration_idx < NUM_ITERATION; iteration_idx++)
	{
		Ray<World> ray = camera_ray(*camera, xy, screen_size);
		color throughput(1,1,1);
		for(int bounce_idx = 0; bounce_idx < NUM_BOUNCES; bounce_idx++)
		{
			Hit<World> hit;
			if(ray.intersect_scene(scene, &hit))
			{
				position<World> light_pos(0, 10, 0);
				direction<World> light_dir(hit.position, light_pos);
				//shade
				value = value + throughput * saturate(dot(light_dir, hit.normal)) * hit.material.albedo;
				if(iteration_idx != NUM_ITERATION - 1)
				{
					//make new ray
					ref::glm::vec2 u = rand2(xy, ref::glm::uvec2(iteration_idx, bounce_idx));
					float inv_pdf;
					direction<World> wi = sampleCosWeightedHemi(hit.normal, u, &inv_pdf);
					throughput = throughput * inv_pdf * hit.material.albedo * dot(wi, hit.normal);
					ray = Ray<World>(hit.position, wi).offset_by(RAY_EPSILON);
				}
			}
			else
			{
				break;
			}
		}
	}
	data[xy.y * width + xy.x] = ref::glm::vec4(to_glm(value / (float)NUM_ITERATION), 1);
}


const int WIDTH = 1000;
const int HEIGHT = 1000;

int main(int argc, char* const argv[]) {
#ifdef UNIT_TEST
	Catch::Main( argc, argv );
#else

	ref::glm::vec4 *d = NULL;
	Camera *camera_ptr = NULL;
	ref::glm::vec4* odata = new ref::glm::vec4[WIDTH * HEIGHT];

	Camera camera(position<World>(0,1,-4), position<World>(0,0,1));
	CUDA_CHECK_RETURN(cudaMalloc((void**) &d, sizeof(ref::glm::vec4) * WIDTH * HEIGHT));
	CUDA_CHECK_RETURN(cudaMalloc((void**) &camera_ptr, sizeof(Camera)));
	CUDA_CHECK_RETURN(cudaMemcpy(camera_ptr, &camera, sizeof(Camera), cudaMemcpyHostToDevice));
	dim3 threadPerBlock(8, 8, 1);
	dim3 blocks((unsigned int)ceil(WIDTH / (float)threadPerBlock.x), (unsigned int)ceil(HEIGHT / (float)threadPerBlock.y), 1);
	gfx_kernel<<<blocks, threadPerBlock>>>(d, camera_ptr, WIDTH, HEIGHT);

	CUDA_CHECK_RETURN(cudaThreadSynchronize());	// Wait for the GPU launched work to complete
	CUDA_CHECK_RETURN(cudaGetLastError());
	CUDA_CHECK_RETURN(cudaMemcpy((void*)odata, (void*)d, sizeof(float4) * WIDTH * HEIGHT, cudaMemcpyDeviceToHost));

	CUDA_CHECK_RETURN(cudaFree((void*) d));
	CUDA_CHECK_RETURN(cudaDeviceReset());

	bitmap_image image(WIDTH, HEIGHT);
	for(int j = 0; j < HEIGHT; j++)
	{
		for(int i = 0; i < WIDTH; i++)
		{
			ref::glm::vec4 col = odata[j * WIDTH + i];
			col /= 1.f + col;
			col *= 255;
			ref::glm::uvec4 ucolor(col);
			image.set_pixel(i, j, ucolor.x, ucolor.y, ucolor.z);
		}
	}
	image.save_image("out.bmp");
	
#endif
	return 0;
}
//tests
TEST_CASE("camera/camera_ray", "standard camera_ray") 
{
	{
		Camera camera(position<World>(0.f,0.f,0.f), position<World>(0, 0, -100));
		ref::glm::uvec2 screen_size(5, 5);
		{
			Ray<World> ray = camera_ray(camera, ref::glm::uvec2(2, 2), screen_size);
			REQUIRE(close_to(ray.origin.z, -1));
		}
		{
			Ray<World> ray = camera_ray(camera, ref::glm::uvec2(0, 0), screen_size);
			REQUIRE(ray.origin.x < 0);
			REQUIRE(ray.origin.y > 0);	
			REQUIRE(close_to(ray.origin.z, -1));
		}
		{
			Ray<World> ray = camera_ray(camera, ref::glm::uvec2(4, 4), screen_size);
			REQUIRE(ray.origin.x > 0);
			REQUIRE(ray.origin.y < 0);	
			REQUIRE(close_to(ray.origin.z, -1));
		}
	}
}

#include <random>
TEST_CASE("diffuse/sample_uniform", "sample hemi") 
{	
	std::mt19937 rng;
	std::uniform_real_distribution<float> normalized_dist(0, 1);
	
	auto b =[&]() -> RandomPair { 
		return RandomPair(normalized_dist(rng), normalized_dist(rng));	
	};
	{
		auto a = [&](RandomPair u, InversePdf* inv_pdf) -> direction<World> {
			return direction<World>(sampleUniformHemi(direction<World>(0, 0, 1), ref::glm::vec2(u), (float*)inv_pdf));			
		};
		auto invPdfFunc = [](NormalizedSphericalCS cs)-> float {
			return 2 * PI;
		};
		validate_importance_sampling<10, 10>(a, b, invPdfFunc, 1000000);
	}
	{
		auto a = [&](RandomPair u, InversePdf* inv_pdf) -> direction<World> {
			return direction<World>(sampleCosWeightedHemi(direction<World>(0, 0, 1), ref::glm::vec2(u), (float*)inv_pdf));			
		};
		auto invPdfFunc = [](NormalizedSphericalCS cs)-> float {
			return PI / cos(cs.x);
		};
		validate_importance_sampling<10, 10>(a, b, invPdfFunc, 1000000);
	}
}


TEST_CASE("intersect/intersect_plane", "hits plane everywhere") 
{
	InfiniteHorizontalPlane plane(0, Material(color(1,1,1)));
	{
		Ray<World> ray(position<World>(0,1,0), direction<World>(0,-1,0));
		Hit<World> hit;
		REQUIRE(ray.intersect_plane(plane, &hit));
		REQUIRE(hit.t == 1);
		REQUIRE(hit.normal.y == 1.f);
	}
}
