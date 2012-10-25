#include <stdio.h>
#include <stdlib.h>
#include "kernel.h"
#include "util.h"
#include "bitmap_image.hpp"
#include "assert.h"

#define CATCH_CONFIG_RUNNER
#include "catch.hpp"

#include "validate_importance_sampling.h"


#ifndef LW_UNIT_TEST
surface<void, cudaSurfaceType2D> output_surf;

template<int SAMPLES_PER_ITERATION>
GPU_ENTRY void transfer_image(Vec3Buffer new_buffer, Vec3Buffer existing_buffer, int iteration_idx, int width, int height)
{
	ref::glm::uvec2 screen_size(width, height);		
#ifndef LW_CPU
	ref::glm::uvec2 xy = screen_xy();
#endif
	if(ref::glm::any(ref::glm::greaterThan(xy, screen_size - ref::glm::uvec2(1)))) return;
	int linid = xy.y * width + xy.x;

	color existing = existing_buffer.get(linid);
	float existing_weight = (float)iteration_idx / (iteration_idx + 1);
	color combined = (color(new_buffer.get(linid)) / (float)SAMPLES_PER_ITERATION) * (1 - existing_weight);
	if(iteration_idx > 0)
	{
		combined = combined + existing * (existing_weight);
	}
	new_buffer.set(linid, v3(0,0,0));
	
	existing_buffer.set(linid, combined);
	color combined_tonemapped = combined  / (combined + color(1,1,1));
	surf2Dwrite(make_float4(combined_tonemapped.x, combined_tonemapped.y, combined_tonemapped.z, 1), output_surf, xy.x*sizeof(float4), xy.y);
	
}
template<int SamplesPerIteration, int NumBounces, bool BdptDebug>
GPU_ENTRY void gfx_kernel(Vec3Buffer buffer, const Camera* camera, const Scene* scene, int iteration_idx, int width, int height
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
	color summed(0,0,0);
	{
		float a = camera->a();
		for(int sample_idx = 0; sample_idx < SamplesPerIteration; sample_idx++)
		{			
			Random rng(xy, RandomCounter(iteration_idx, sample_idx));
			color light_throughput;
			Hit<World> light_vertex = sample_sphere_light(scene->sphere_lights[0], rng.next2(), &light_throughput);		
			ray<World> light_ray;
			for(int light_vertex_idx = 0; light_vertex_idx < NumBounces + 1; light_vertex_idx++)
			{			
				//direct connect with eye
				if(light_vertex.material.type != eSpecular)
				{
					bool in_bounds;
					ref::glm::vec2 ndc;
					ref::glm::vec2 uv = world_to_screen(light_vertex.position, *camera, width, height, &in_bounds, ndc);
					if(in_bounds)
					{
						ray<World> light_to_eye_shadow_ray = ray<World>(light_vertex.position, camera->eye)
							.offset_by(RAY_EPSILON);
						if(!light_to_eye_shadow_ray.intersect_shadow(*scene, camera->eye))
						{						
							float costheta_shadow_ev = clamp01(-dot(light_to_eye_shadow_ray.dir, camera->forward));
							float costheta_shadow_lv = clamp01(dot(light_to_eye_shadow_ray.dir, light_vertex.normal));
							float d = (light_vertex.position - camera->eye).length();
							float g = costheta_shadow_ev * costheta_shadow_lv / (d * d);
							float we = 1 / (a * pow3(costheta_shadow_ev));
							int variations = 2;
							color addition = light_throughput * light_vertex.material.brdf() 
								* g * we / (float)variations;
							if(BdptDebug)
							{
								store_bdpt_debug(buffer, addition, width, height, 1, light_vertex_idx + 1, uv);
							}
							else
							{								
								buffer.elementwise_atomic_add(uv, width, addition);
							}
						}
					}
				}
				if(light_vertex_idx < NumBounces)
				{
					/* HACK... should divide by PI for vertex 0 */
					light_throughput = light_throughput * light_vertex.material.brdf();
					extend(rng.next2(), light_vertex, &light_ray, &light_throughput);
					if(!light_ray.intersect(*scene, &light_vertex)) break;
				}
			}
		}
	}
	//eye
	{		
		color summed(0,0,0);
		
		for(int sample_idx = 0; sample_idx < SamplesPerIteration; sample_idx++)
		{			
			Random rng(xy, RandomCounter(iteration_idx, sample_idx + 1000));
			color value(0,0,0);
			color eye_throughput(1,1,1);
			ray<World> ray0 = camera_ray(*camera, xy, screen_size);
			Hit<World> eye_vertex_1;
			if(!ray0.intersect(*scene, &eye_vertex_1)) return;
			Hit<World> eye_vertex = eye_vertex_1;
			ray<World> eye_ray = ray0;
			for(int eye_vertex_idx = 1; eye_vertex_idx < NumBounces + 1; eye_vertex_idx++)
			{				
				int variations = 2;				
				color addition(0,0,0);
				if(eye_vertex.material.type == eEmissive) //last vertex is implied to be specular
				{
					addition = eye_vertex.material.emissive.emission * eye_throughput / (float)(variations);
				}
				else if(eye_vertex.material.type == eDiffuse)
				{
					color light_spatial_throughput;
					Hit<World> light = 
						scene->sample_light(eye_vertex.position, rng.next2(), &light_spatial_throughput);
												
					color light_throughput = light_spatial_throughput
						* connection_throughput(light, eye_vertex, *scene);
					
					addition = eye_vertex.material.brdf() * light_throughput * eye_throughput / (float)(variations);
					
				}
				if(!addition.is_black())
				{
					if(BdptDebug)
					{
						store_bdpt_debug(buffer, addition, width, height, eye_vertex_idx + 1, 
							eye_vertex.material.type == eEmissive ? 0 : 1, ref::glm::vec2(xy));
					}
					else
					{	
						value = value + addition;
					}
				}
				if(eye_vertex.material.type == eEmissive) break;
				if(eye_vertex_idx < NumBounces)
				{
					eye_throughput = eye_throughput * eye_vertex.material.brdf();
					extend(rng.next2(), eye_vertex, &eye_ray, &eye_throughput);

					if(!eye_ray.intersect(*scene, &eye_vertex, eye_vertex.material.type == eSpecular)) break;
				}
			}
			summed = summed + value;
		}
		
		if(!BdptDebug)
		{
			buffer.set(linid, summed);
		}
	}
}
void Kernel::setup(cudaGraphicsResource* output, int width, int height)
{
	framebuffer_resource = output;
	CUDA_CHECK_RETURN(cudaMalloc((void**) &camera_ptr, sizeof(Camera)));
	CUDA_CHECK_RETURN(cudaMalloc((void**) &scene_ptr, sizeof(Scene)));
	//CUDA_CHECK_RETURN(cudaMalloc((void**) &buffer, sizeof(ref::glm::vec4) * width * height));
	new_buffer = new Vec3Buffer();
	new_buffer->init(width * height);
	existing_buffer = new Vec3Buffer();
	existing_buffer->init(width * height);
}

void Kernel::execute(int iteration_idx, int width, int height, bool bdpt_debug, Stats* stats)
{	
	const int SAMPLES_PER_ITERATION = 6;
	const int NUM_BOUNCES = 4;
	*stats = Stats::two_way(SAMPLES_PER_ITERATION, NUM_BOUNCES, width, height);
	stats->start();
	Camera camera(position<World>(0,9,-7), position<World>(0,0,1), bdpt_debug ? 1 : (float)width/height);
	Scene scene;

	CUDA_CHECK_RETURN(cudaMemcpy(camera_ptr, &camera, sizeof(Camera), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(scene_ptr, &scene, sizeof(Scene), cudaMemcpyHostToDevice));

	dim3 threadPerBlock(16, 16, 1);
	dim3 blocks((unsigned int)ceil(width / (float)threadPerBlock.x), 
		(unsigned int)ceil(height / (float)threadPerBlock.y), 1);	
	gfx_kernel<SAMPLES_PER_ITERATION, NUM_BOUNCES, false><<<blocks, threadPerBlock>>>(*new_buffer, (Camera*)camera_ptr, 
		(Scene*)scene_ptr, iteration_idx, width, height);


	
	cudaArray_t framebuffer_ptr;
	
	CUDA_CHECK_RETURN(cudaGraphicsMapResources(1, &framebuffer_resource));
	CUDA_CHECK_RETURN(cudaGraphicsSubResourceGetMappedArray(&framebuffer_ptr, framebuffer_resource, 0, 0));
	CUDA_CHECK_RETURN(cudaBindSurfaceToArray(output_surf, framebuffer_ptr));

	transfer_image<SAMPLES_PER_ITERATION><<<blocks, threadPerBlock>>>(*new_buffer, 
		*existing_buffer, iteration_idx, width, height);
	CUDA_CHECK_RETURN(cudaGraphicsUnmapResources(1, &framebuffer_resource));
	stats->stop();
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

