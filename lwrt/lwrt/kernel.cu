#include <stdio.h>
#include <stdlib.h>
#include "kernel.h"
#include "util.h"
#include "bitmap_image.hpp"
#include "assert.h"

#define CATCH_CONFIG_RUNNER
#include "catch.hpp"

#include "validate_importance_sampling.h"

struct Vec3Buffer
{
	float* x;
	float* y;
	float* z;
	void init(int size) 
	{
		CUDA_CHECK_RETURN(cudaMalloc(&x, sizeof(float) * size));
		CUDA_CHECK_RETURN(cudaMalloc(&y, sizeof(float) * size));
		CUDA_CHECK_RETURN(cudaMalloc(&z, sizeof(float) * size));
	}
	GPU_CPU v3 get(int idx) const
	{
		return v3(x[idx], y[idx], z[idx]);
	}
	GPU_CPU void set(int idx, const v3& v)
	{
		x[idx] = v.x; y[idx] = v.y; z[idx] = v.z;
	}
};
struct Pass
{
	Pass(int iteration_idx, int num_iterations, int num_bounces) 
		: iteration_idx(iteration_idx), num_iterations(num_iterations), num_bounces(num_bounces) { }
	int iteration_idx;
	int num_iterations;
	int num_bounces;
};

#ifndef LW_UNIT_TEST
surface<void, cudaSurfaceType2D> output_surf;
GPU RandomPair mutate_rand(Random& rng, RandomPair& u)
{
	//TODO: change this
	RandomPair pair = u + (rng.next2() - 0.5f) / 5.f;
	u.x = fmodf(u.x, 1.f);
	u.y = fmodf(u.y, 1.f);
	u = pair;
	return pair;
}
#define MAX_PATH_VERTS 4
GPU_ENTRY void gfx_kernel(Vec3Buffer buffer, const Camera* camera, const Scene* scene, const Pass* pass, int width, int height
#ifdef LW_CPU
//	, ref::glm::uvec2 xy
#endif
	) {
	ref::glm::uvec2 screen_size(width, height);
		
#ifndef LW_CPU
	ref::glm::uvec2 xy = screen_xy();
#endif
	if(ref::glm::any(ref::glm::greaterThan(xy, screen_size - ref::glm::uvec2(1)))) return;
	int linid = xy.y * width + xy.x;
	//TODO: add specular support
	RandomPair u[MAX_PATH_VERTS];
	Random rng(xy, RandomCounter(pass->iteration_idx, 0));
	//do an initial walk
	{
		bool use_implicit_light = true;
		ray<World> eye_ray = ray0;
		color throughput(1,1,1);
		for(int bounce_idx = 0; bounce_idx < (pass->num_bounces + 1); bounce_idx++)
		{
			Hit<World> hit;
			if(eye_ray.intersect(*scene, &hit, use_implicit_light))
			{
				if(hit.material.emission.is_black() && (bounce_idx < pass->num_bounces))
				{
					if(!hit.material.is_specular)
					{
						RandomPair u = rand2(RandomKey(xy), RandomCounter(iteration_idx, pass->num_bounces + bounce_idx));
						color inv_light_pdf;
						position<World> light_pos = scene->sample_light(hit.position, u, &inv_light_pdf);
						
						direction<World> light_dir(hit.position, light_pos);

						if(!ray<World>(hit.position, light_dir)
							.offset_by(RAY_EPSILON)
							.intersect_shadow(*scene, light_pos))
						{
							float d = (hit.position - light_pos).length();
							value = value + throughput 
								* clamp01(dot(light_dir, hit.normal)) 
								* inv_light_pdf 
								/ (d * d)
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
					value = value + throughput * hit.material.emission * PI;
					break;
				}		
				if(bounce_idx == pass->num_bounces)
				{
					break;
				}
				
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
					direction<World> wi = sampleCosWeightedHemi(hit.normal, u, &ip_pdf);
					throughput = throughput * ip_pdf * hit.material.brdf();
					eye_ray = ray<World>(hit.position, wi).offset_by(RAY_EPSILON);
				}
				
			}
			else
			{
				break;
			}

		}
	}
	color existing = buffer.get(linid);
	float existing_weight = (float)pass->iteration_idx / (pass->iteration_idx + pass->num_iterations);
	color combined = (value / (float)pass->num_iterations) * (1 - existing_weight);
	if(pass->iteration_idx > 0)
	{
		combined = combined + existing * (existing_weight);
	}
	buffer.set(linid, combined);
	color combined_tonemapped = combined / (combined + color(1,1,1));
	surf2Dwrite(make_float4(combined_tonemapped.x, combined_tonemapped.y, combined_tonemapped.z, 1), output_surf, xy.x*sizeof(float4), xy.y);
}

void Kernel::setup(cudaGraphicsResource* output, int width, int height)
{
	framebuffer_resource = output;
	CUDA_CHECK_RETURN(cudaMalloc((void**) &camera_ptr, sizeof(Camera)));
	CUDA_CHECK_RETURN(cudaMalloc((void**) &scene_ptr, sizeof(Scene)));
	CUDA_CHECK_RETURN(cudaMalloc((void**) &pass_ptr, sizeof(Pass)));
	//CUDA_CHECK_RETURN(cudaMalloc((void**) &buffer, sizeof(ref::glm::vec4) * width * height));
	buffer = new Vec3Buffer();
	buffer->init(width * height);
}
void Kernel::execute(int iteration_idx, int iterations, int bounces, int width, int height)
{	
	Pass pass(iteration_idx, iterations, bounces);
	Camera camera(position<World>(0,9,-7), position<World>(0,0,1));
	Scene scene;

	CUDA_CHECK_RETURN(cudaMemcpy(camera_ptr, &camera, sizeof(Camera), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(scene_ptr, &scene, sizeof(Scene), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(pass_ptr, &pass, sizeof(Pass), cudaMemcpyHostToDevice));

	cudaArray_t framebuffer_ptr;
	
	CUDA_CHECK_RETURN(cudaGraphicsMapResources(1, &framebuffer_resource));
	CUDA_CHECK_RETURN(cudaGraphicsSubResourceGetMappedArray(&framebuffer_ptr, framebuffer_resource, 0, 0));
	CUDA_CHECK_RETURN(cudaBindSurfaceToArray(output_surf, framebuffer_ptr));
	dim3 threadPerBlock(16, 16, 1);
	dim3 blocks((unsigned int)ceil(width / (float)threadPerBlock.x), 
		(unsigned int)ceil(height / (float)threadPerBlock.y), 1);	
	gfx_kernel<<<blocks, threadPerBlock>>>(*buffer, (Camera*)camera_ptr, (Scene*)scene_ptr, (Pass*)pass_ptr, width, height);
	CUDA_CHECK_RETURN(cudaGraphicsUnmapResources(1, &framebuffer_resource));
}
#endif
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
TEST_CASE("math/color", "color math works") 
{
	REQUIRE(all_equal(color(1,1,1) / 2, color(0.5f, 0.5f, 0.5f)));
	//REQUIRE(all_equal(color(1,1,1) + 2, color(3,3,3)));
	REQUIRE(all_equal(color(1,1,1) / color(2,2,2), color(0.5f, 0.5f, 0.5f)));
}
#endif