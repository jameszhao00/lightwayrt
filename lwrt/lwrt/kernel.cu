#include <stdio.h>
#include <stdlib.h>
#include "util.h"
#include "bitmap_image.hpp"
#include "assert.h"

#define CATCH_CONFIG_RUNNER
#include "catch.hpp"

#include "validate_importance_sampling.h"

const int NUM_ITERATION = 300;
const int NUM_BOUNCES = 3;

__global__ void gfx_kernel(vec4 *data, const Camera* camera, int width, int height) {
	uvec2 screen_size(width, height);
	uvec2 xy = screen_xy();
	if(glm::any(glm::greaterThan(xy, screen_size - uvec2(1)))) return;
	Scene scene;
	color value(0,0,0);
	for(int iteration_idx = 0; iteration_idx < NUM_ITERATION; iteration_idx++)
	{
		Ray<WorldCS> ray = camera_ray(*camera, xy, screen_size);
		color throughput(1,1,1);
		for(int bounce_idx = 0; bounce_idx < NUM_BOUNCES; bounce_idx++)
		{
			Hit<WorldCS> hit;
			if(ray.intersect_scene(scene, &hit))
			{
				position<WorldCS> light_pos(0, 10, 0);
				direction<WorldCS> light_dir(hit.position, light_pos);
				//shade
				value += throughput * saturate(light_dir.dot(hit.normal)) * hit.material.albedo;
				if(iteration_idx != NUM_ITERATION - 1)
				{
					//make new ray
					vec2 u = rand2(xy, uvec2(iteration_idx, bounce_idx));
					float inv_pdf;
					direction<WorldCS> wi = sampleCosWeightedHemi(hit.normal, u, &inv_pdf);
					throughput *= inv_pdf * hit.material.albedo * wi.dot(hit.normal);
					ray = Ray<WorldCS>(hit.position, wi).offset_by(RAY_EPSILON);
				}
			}
			else
			{
				break;
			}
		}
	}
	data[xy.y * width + xy.x] = vec4(value / (float)NUM_ITERATION, 1);
}


const int WIDTH = 1000;
const int HEIGHT = 1000;

int main(int argc, char* const argv[]) {
#ifdef UNIT_TEST
	Catch::Main( argc, argv );
#else

	vec4 *d = NULL;
	Camera *camera_ptr = NULL;
	vec4* odata = new vec4[WIDTH * HEIGHT];

	Camera camera(position<WorldCS>(0,1,-4), position<WorldCS>(0,0,1));
	CUDA_CHECK_RETURN(cudaMalloc((void**) &d, sizeof(vec4) * WIDTH * HEIGHT));
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
			vec4 color = odata[j * WIDTH + i];
			color /= 1.f + color;
			color *= 255;
			uvec4 ucolor(color);
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
		Camera camera(position<WorldCS>(0.f,0.f,0.f), position<WorldCS>(0, 0, -100));
		uvec2 screen_size(5, 5);
		{
			auto ray = camera_ray(camera, uvec2(2, 2), screen_size);
			REQUIRE(close_to(ray.origin.z, -1));
		}
		{
			auto ray = camera_ray(camera, uvec2(0, 0), screen_size);
			REQUIRE(ray.origin.x < 0);
			REQUIRE(ray.origin.y > 0);	
			REQUIRE(close_to(ray.origin.z, -1));
		}
		{
			auto ray = camera_ray(camera, uvec2(4, 4), screen_size);
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
		auto a = [&](RandomPair u, InversePdf* inv_pdf) -> direction<WorldCS> {
			return direction<WorldCS>(sampleUniformHemi(direction<WorldCS>(0, 0, 1), vec2(u), (float*)inv_pdf));			
		};
		auto invPdfFunc = [](NormalizedSphericalCS cs)-> float {
			return 2 * PI;
		};
		validate_importance_sampling<10, 10>(a, b, invPdfFunc, 1000000);
	}
	{
		auto a = [&](RandomPair u, InversePdf* inv_pdf) -> direction<WorldCS> {
			return direction<WorldCS>(sampleCosWeightedHemi(direction<WorldCS>(0, 0, 1), vec2(u), (float*)inv_pdf));			
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
		Ray<WorldCS> ray(position<WorldCS>(0,1,0), direction<WorldCS>(0,-1,0));
		Hit<WorldCS> hit;
		REQUIRE(ray.intersect_plane(plane, &hit));
		REQUIRE(hit.t == 1);
		REQUIRE(glm::all(glm::equal(hit.normal, vec3(0.f,1.f,0.f))));
	}
}