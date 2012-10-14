#include <stdio.h>
#include <stdlib.h>
#include "util.h"
#include "bitmap_image.hpp"
#include "assert.h"

#define CATCH_CONFIG_RUNNER
#include "catch.hpp"

#include "validate_importance_sampling.h"

const int NUM_ITERATION = 900;
const int NUM_BOUNCES = 3;

GPU_ENTRY void gfx_kernel(ref::glm::vec4 *data, const Camera* camera, const Scene* scene, int width, int height
#ifdef LW_CPU
	, ref::glm::uvec2 xy
#endif
	) {
	ref::glm::uvec2 screen_size(width, height);
		
#ifndef LW_CPU
	ref::glm::uvec2 xy = screen_xy();
#endif
	if(ref::glm::any(ref::glm::greaterThan(xy, screen_size - ref::glm::uvec2(1)))) return;
	int linid = xy.y * width + xy.x;
	
	color value(0,0,0);
	ray<World> ray0 = camera_ray(*camera, xy, screen_size);
	for(int iteration_idx = 0; iteration_idx < NUM_ITERATION; iteration_idx++)
	{
		bool use_implicit_light = true;
		ray<World> eye_ray = ray0;
		color throughput(1,1,1);
		for(int bounce_idx = 0; bounce_idx < NUM_BOUNCES; bounce_idx++)
		{
			Hit<World> hit;
			if(eye_ray.intersect(*scene, &hit, use_implicit_light))
			{
				if(hit.material.emission.is_black())
				{
					if(!hit.material.is_specular)
					{
						RandomPair u = rand2(RandomKey(xy), RandomCounter(iteration_idx, NUM_BOUNCES + bounce_idx));
						color inv_light_pdf;
						position<World> light_pos = scene->sample_light(hit.position, u, &inv_light_pdf);
						direction<World> light_dir(hit.position, light_pos);

						if(!ray<World>(hit.position, light_dir)
							.offset_by(RAY_EPSILON)
							.intersect_shadow(*scene, light_pos))
						{
							value = value + throughput 
								* clamp01(dot(light_dir, hit.normal)) 
								* inv_light_pdf 
								* hit.material.brdf();
						}

						use_implicit_light = false;
					}
					else
					{
						use_implicit_light = true;
					}
				}
				else
				{
					value = value + throughput * hit.material.emission;
					break;
				}			

				if(bounce_idx != NUM_BOUNCES - 1)
				{
					if(hit.material.is_specular)
					{
						throughput = throughput * hit.material.albedo;
						eye_ray = ray<World>(hit.position, eye_ray.dir.reflect(hit.normal))
							.offset_by(RAY_EPSILON);
					}
					else
					{
						RandomPair u = rand2(RandomKey(xy), RandomCounter(iteration_idx, bounce_idx));
						InverseProjectedPdf ip_pdf;
						direction<World> wi = sampleUniformHemi(hit.normal, u, &ip_pdf);
						throughput = throughput * ip_pdf * hit.material.brdf();
						eye_ray = ray<World>(hit.position, wi).offset_by(RAY_EPSILON);
					}
				}
			}
			else
			{
				break;
			}

		}
	}
	
	data[linid] = ref::glm::vec4(to_glm(value / (float)NUM_ITERATION), 1);
}


const int WIDTH = 600;
const int HEIGHT = 600;

int main(int argc, char* const argv[]) {
#ifdef LW_UNIT_TEST
	Catch::Main( argc, argv );
#else

	ref::glm::vec4 *d = NULL;
	Camera *camera_ptr = NULL;
	Scene *scene_ptr= NULL;
	ref::glm::vec4* odata = new ref::glm::vec4[WIDTH * HEIGHT];

	Camera camera(position<World>(0,7,-5), position<World>(0,0,1));
	Scene scene;
	CUDA_CHECK_RETURN(cudaMalloc((void**) &d, sizeof(ref::glm::vec4) * WIDTH * HEIGHT));
	CUDA_CHECK_RETURN(cudaMalloc((void**) &camera_ptr, sizeof(Camera)));
	CUDA_CHECK_RETURN(cudaMalloc((void**) &scene_ptr, sizeof(Scene)));
	CUDA_CHECK_RETURN(cudaMemcpy(camera_ptr, &camera, sizeof(Camera), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(scene_ptr, &scene, sizeof(Scene), cudaMemcpyHostToDevice));
	dim3 threadPerBlock(8, 8, 1);
	dim3 blocks((unsigned int)ceil(WIDTH / (float)threadPerBlock.x), (unsigned int)ceil(HEIGHT / (float)threadPerBlock.y), 1);
	gfx_kernel<<<blocks, threadPerBlock>>>(d, camera_ptr, scene_ptr, WIDTH, HEIGHT);

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
#ifdef LW_UNIT_TEST
//tests
TEST_CASE("camera/camera_ray", "standard camera_ray") 
{
	{
		Camera camera(position<World>(0.f,0.f,0.f), position<World>(0, 0, -100));
		ref::glm::uvec2 screen_size(5, 5);
		{
			ray<World> ray = camera_ray(camera, ref::glm::uvec2(2, 2), screen_size);
			REQUIRE(close_to(ray.origin.z, -1));
		}
		{
			ray<World> ray = camera_ray(camera, ref::glm::uvec2(0, 0), screen_size);
			REQUIRE(ray.origin.x < 0);
			REQUIRE(ray.origin.y > 0);	
			REQUIRE(close_to(ray.origin.z, -1));
		}
		{
			ray<World> ray = camera_ray(camera, ref::glm::uvec2(4, 4), screen_size);
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
		auto a = [&](RandomPair u, InverseProjectedPdf* inv_pdf) -> direction<World> {
			return direction<World>(sampleUniformHemi(direction<World>(0, 0, 1), ref::glm::vec2(u), (float*)inv_pdf));			
		};
		auto invPdfFunc = [](NormalizedSphericalCS cs)-> float {
			return 2 * PI * cos(cs.x);
		};
		validate_importance_sampling<10, 10>(a, b, invPdfFunc, 1000000);
	}
	{
		auto a = [&](RandomPair u, InverseProjectedPdf* inv_pdf) -> direction<World> {
			return direction<World>(sampleCosWeightedHemi(direction<World>(0, 0, 1), ref::glm::vec2(u), (float*)inv_pdf));			
		};
		auto invPdfFunc = [](NormalizedSphericalCS cs)-> float {
			return PI;// / cos(cs.x);
		};
		validate_importance_sampling<10, 10>(a, b, invPdfFunc, 1000000);
	}
}


TEST_CASE("intersect/intersect_plane", "hits plane everywhere") 
{
	InfiniteHorizontalPlane plane(0, Material(color(1,1,1), color(), false));
	{
		ray<World> ray(position<World>(0,1,0), direction<World>(0,-1,0));
		Hit<World> hit;
		REQUIRE(ray.intersect_plane(plane, &hit));
		REQUIRE(hit.t == 1);
		REQUIRE(hit.normal.y == 1.f);
	}
}

#endif