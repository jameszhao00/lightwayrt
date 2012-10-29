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
struct PassData
{
	int iteration_idx;
	int sample_idx;
};
GPU int linid() { return blockIdx.x * blockDim.x + threadIdx.x; }
GPU ref::glm::uvec2 pixel_xy(int w, int h) { return ref::glm::uvec2(linid() % w, linid() / h); }

#define COLOR_ELEMENT_TYPE float

typedef int MaterialId;
GPU_CPU float gaussian(float num, float sigma)
{
	return expf(-1 * num / (2 * sigma * sigma));
}
GPU_ENTRY void bilateral(
	const Scene* scene,
	float DomainSigma, 
	float RangeSigma, 
	float PositionWeight, 
	float NormalWeight,
	LW_INOUT Vec3Buffer val_buf,
	LW_IN Vec3Buffer pos_buf,
	LW_IN Vec3Buffer norm_buf
	)
{
	int filter_size = 3;//ceil(DomainSigma * 2);
	int linid = blockIdx.x * blockDim.x + threadIdx.x;
	int x = linid % scene->camera.screen_width;
	int y = linid / scene->camera.screen_height;
	direction<World> self_normal = norm_buf.get(linid);
	position<World> self_pos = pos_buf.get(linid);
	//TODO: need to weight the discrete gaussian
	color sum(0,0,0);
	for(int i = -1 * filter_size; i <= filter_size; i++)
	{
		for(int j = -1 * filter_size; j <= filter_size; j++)
		{
			if(i < 0 || j < 0 || i > scene->camera.screen_width - 1 || i > scene->camera.screen_height - 1)
			{
				continue;
			}
			int sample_lin_id = (x + i) + (j + y) * scene->camera.screen_width;
			float d = gaussian(i*i+j*j, DomainSigma);
			float pos_dist2 = dot_self(v3_sub(pos_buf.get(sample_lin_id), self_pos));
			float norm_dist2 = dot_self(v3_sub(norm_buf.get(sample_lin_id), self_normal));
			float r = gaussian(pos_dist2, RangeSigma) * gaussian(norm_dist2, RangeSigma);
			sum = sum + v3_mul(val_buf.get(sample_lin_id), d * r);
		}
	}
	val_buf.set(linid, sum);
}

int aligned_size(int x)
{
	const int ALIGNMENT_SIZE = 32;
	return (int)ceil((float)x / ALIGNMENT_SIZE);
}
cudaChannelFormatDesc make_diffuse_texdesc()
{
	return cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
}
void Kernel::setup(cudaGraphicsResource* output, int width, int height)
{	
	//TODO: setup pass data
	framebuffer_resource = output;
	int num_pixels = (width) * (height);
	CUDA_CHECK_RETURN(cudaMalloc((void**) &scene_ptr, sizeof(Scene)));
	CUDA_CHECK_RETURN(cudaMalloc((void**) &pass_ptr, sizeof(PassData)));
	{
		cudaChannelFormatDesc format = make_diffuse_texdesc();
		cudaMallocArray(&cached_diffuse, &format, width, height, cudaArraySurfaceLoadStore);
	}
	//CUDA_CHECK_RETURN(cudaMalloc((void**) &buffer, sizeof(ref::glm::vec4) * width * height));
	new_buffer = new Vec3Buffer();
	new_buffer->init(num_pixels);
	existing_buffer = new Vec3Buffer();
	existing_buffer->init(num_pixels);

	light_vtx_pos = new Vec3Buffer();
	light_vtx_normal = new Vec3Buffer();
	light_throughput = new Vec3Buffer();
	light_wi = new Vec3Buffer();

	eye_vtx_pos = new Vec3Buffer();
	eye_vtx_normal = new Vec3Buffer();
	eye_throughput = new Vec3Buffer();
	eye_wi = new Vec3Buffer();
	
	CUDA_CHECK_RETURN(cudaMalloc((void**) &light_vtx_material_id_ptr, sizeof(int) * num_pixels));
	CUDA_CHECK_RETURN(cudaMalloc((void**) &eye_vtx_material_id_ptr, sizeof(int) * num_pixels));

	light_vtx_pos->init(num_pixels);
	light_vtx_normal->init(num_pixels);
	light_throughput->init(num_pixels);
	light_wi->init(num_pixels);

	eye_vtx_pos->init(num_pixels);
	eye_vtx_normal->init(num_pixels);
	eye_throughput->init(num_pixels);
	eye_wi->init(num_pixels);

	previous_frame_camera = new Camera();
}

surface<void, cudaSurfaceType2D> output_surf;
//surface<void, cudaSurfaceType2D> cached_diffuse_surf;
texture<float4, 2> cached_diffuse_tex;
template<int NumCycles, int NumDirectLight, int MaxBounces>
GPU_ENTRY void pt(
	const PassData* pass_data,
	const Scene* scene)
{
	color value_direct(0,0,0);
	color value_indirect(0,0,0); //3 reg
	color caustic(0,0,0);
 	ref::glm::uvec2 xy(linid() % scene->camera.screen_width, linid() / scene->camera.screen_height);
 	
 	auto ray01 = camera_ray(scene->camera, xy, ref::glm::uvec2(scene->camera.screen_width, scene->camera.screen_height)); //6 reg
	Hit<World> eye_vtx1; 
	//direct lighting
	{
		bool hit = ray01.offset_by(RAY_EPSILON).intersect(*scene, &eye_vtx1);
		for(int i = 0; i < NumDirectLight; i++)
		{
			color light_T;
			Hit<World> light_vtx = scene->sample_light(eye_vtx1.position, rand2(RandomKey(linid(), pass_data->iteration_idx), RandomCounter(clock(), 1)), &light_T);
			color contribution = light_T * /* eye_T * = 1*/
				scene->materials[light_vtx.material_id].brdf() * scene->materials[eye_vtx1.material_id].brdf() *
				connection_throughput(*scene, light_vtx, eye_vtx1);
			value_direct = value_direct + contribution;
		}
		value_direct = value_direct / (float)NumDirectLight;
	}
	//indirect lighting
	{
 		int bounce = 0;
 		int samples = 0;
 		Hit<World> eye_vtx; //8 reg
 		color eye_T(-1,-1,-1); //init to 'gen eye ray' mode //3 reg 
 		direction<World> wi; //3 reg
 		for(int i = 0; i < NumCycles; i++)
 		{
			//bool prev_diffuse = true;
			if(eye_T.x == -1)
			{
				samples++;
				eye_T = color(1,1,1);
				wi = ray01.dir;
				eye_vtx = eye_vtx1; //assume 1 full path already constructed				
			}
 			next_wi(*scene, rand2(RandomKey(linid(), pass_data->iteration_idx), RandomCounter(clock(), 0)), 
 				eye_vtx.normal, eye_vtx.material_id, &wi, &eye_T);			
		
			ray<World> eye_ray(eye_vtx.position, wi);
			bool hit = eye_ray.offset_by(RAY_EPSILON).intersect(*scene, &eye_vtx);
			//bool is_SD_path = false;//prev_diffuse && hit && scene->materials[eye_vtx.material_id].type == eSpecular;
			if(hit && scene->materials[eye_vtx.material_id].type != eSpecular)// && !is_SD_path)
			{
				color light_T;
				//connect
				Hit<World> light_vtx = scene->sample_light(eye_vtx.position, rand2(RandomKey(linid(), pass_data->iteration_idx), RandomCounter(clock(), 1)), &light_T);
				color contribution = light_T * eye_T * 
					scene->materials[light_vtx.material_id].brdf() * scene->materials[eye_vtx.material_id].brdf() *
					connection_throughput(*scene, light_vtx, eye_vtx);
				value_indirect = value_indirect + contribution;
			
			}
			//disable L(D|S)*SDE paths... sample those separately
			if(!hit || bounce == MaxBounces - 1)// || is_SD_path)
			{
				//reset
				eye_T.x = -1;
				bounce = 1; //start at 2 bounce
			}
			else
			{
				bounce++;
			}
 		}
		value_indirect = value_indirect / samples;
	}
	color total_contrib = value_indirect + value_direct;
	color tonemapped = total_contrib / (total_contrib + color(1,1,1));
	surf2Dwrite(tonemapped.to_f4(), output_surf, xy.x*sizeof(float4), xy.y);
}
void Kernel::execute(int iteration_idx, int width, int height, bool bdpt_debug, Stats* stats)
{			
	const int NUM_BOUNCES = 4;
	const int CYCLES_PER_ITERATION = NUM_BOUNCES * 20;
	const int NUM_DIRECT_SAMPLES = 5;
	*stats = Stats::pt(CYCLES_PER_ITERATION, width, height);
	stats->start();
	PassData pass_data;
	pass_data.iteration_idx = iteration_idx;
	pass_data.iteration_idx = 1;

	{		
		cached_diffuse_tex.addressMode[0] = cudaAddressModeClamp;
		cached_diffuse_tex.addressMode[1] = cudaAddressModeClamp;
		cached_diffuse_tex.filterMode = cudaFilterModeLinear;
		cached_diffuse_tex.normalized = false;
	}

	Camera camera(position<World>(0,9,-7), position<World>(0,0,1), bdpt_debug ? 1 : (float)width/height, 
		width, height, (width), (height));
	if(iteration_idx == 0) *previous_frame_camera = camera;
	Scene scene(camera, *previous_frame_camera);
	scene.sphere_lights[0].origin.z += iteration_idx / 10.f;
	*previous_frame_camera = camera;
	CUDA_CHECK_RETURN(cudaMemcpy(scene_ptr, &scene, sizeof(Scene), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(pass_ptr, &pass_data, sizeof(PassData), cudaMemcpyHostToDevice));

	int num_pixels = width * height;

	dim3 threadPerBlock(64, 1, 1);
	dim3 blocks(num_pixels / 64, 1, 1);

	cudaArray_t framebuffer_ptr;
	
	CUDA_CHECK_RETURN(cudaGraphicsMapResources(1, &framebuffer_resource));
	CUDA_CHECK_RETURN(cudaGraphicsSubResourceGetMappedArray(&framebuffer_ptr, framebuffer_resource, 0, 0));
	CUDA_CHECK_RETURN(cudaBindSurfaceToArray(output_surf, framebuffer_ptr));
	//CUDA_CHECK_RETURN(cudaBindSurfaceToArray(cached_diffuse_surf, cached_diffuse));
	
	cudaChannelFormatDesc format = make_diffuse_texdesc();
	CUDA_CHECK_RETURN(cudaBindTextureToArray(cached_diffuse_tex, cached_diffuse, format));
	pt<CYCLES_PER_ITERATION, NUM_DIRECT_SAMPLES, NUM_BOUNCES><<<blocks, threadPerBlock>>>((PassData*)pass_ptr, (Scene*)scene_ptr);
	cudaMemcpyArrayToArray(cached_diffuse, 0, 0, framebuffer_ptr, 0, 0, sizeof(float4) * width * height);
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

// 
// 
// template<int SamplesPerIteration, int NumBounces, bool BdptDebug>
// GPU_ENTRY void gfx_kernel(Vec3Buffer buffer, const Camera* camera, const Scene* scene, int iteration_idx, int width, int height
// #ifdef LW_CPU
// //	, ref::glm::uvec2 xy
// #endif
// 	) {
// 	ref::glm::uvec2 screen_size(width, height);
// 		
// #ifndef LW_CPU
// 	ref::glm::uvec2 xy = screen_xy();
// #endif
// 	if(ref::glm::any(ref::glm::greaterThan(xy, screen_size - ref::glm::uvec2(1)))) return;
// 	int linid = xy.y * width + xy.x;
// 	color summed(0,0,0);
// 	{
// 		float a = camera->a();
// 		for(int sample_idx = 0; sample_idx < SamplesPerIteration; sample_idx++)
// 		{			
// 			Random rng(xy, RandomCounter(iteration_idx, sample_idx));
// 			color light_throughput;
// 			Hit<World> light_vertex = sample_sphere_light(scene->sphere_lights[0], rng.next2(), &light_throughput);		
// 			ray<World> light_ray;
// 			for(int light_vertex_idx = 0; light_vertex_idx < NumBounces + 1; light_vertex_idx++)
// 			{			
// 				//direct connect with eye
// 				if(light_vertex.material.type != eSpecular)
// 				{
// 					bool in_bounds;
// 					ref::glm::vec2 ndc;
// 					ref::glm::vec2 uv = world_to_screen(light_vertex.position, *camera, width, height, &in_bounds, ndc);
// 					if(in_bounds)
// 					{
// 						ray<World> light_to_eye_shadow_ray = ray<World>(light_vertex.position, camera->eye)
// 							.offset_by(RAY_EPSILON);
// 						if(!light_to_eye_shadow_ray.intersect_shadow(*scene, camera->eye))
// 						{						
// 							float costheta_shadow_ev = clamp01(-dot(light_to_eye_shadow_ray.dir, camera->forward));
// 							float costheta_shadow_lv = clamp01(dot(light_to_eye_shadow_ray.dir, light_vertex.normal));
// 							float d = (light_vertex.position - camera->eye).length();
// 							float g = costheta_shadow_ev * costheta_shadow_lv / (d * d);
// 							float we = 1 / (a * pow3(costheta_shadow_ev));
// 							int variations = 2;
// 							color addition = light_throughput * light_vertex.material.brdf() 
// 								* g * we / (float)variations;
// 							if(BdptDebug)
// 							{
// 								store_bdpt_debug(buffer, addition, width, height, 1, light_vertex_idx + 1, uv);
// 							}
// 							else
// 							{								
// 								buffer.elementwise_atomic_add(uv, width, addition);
// 							}
// 						}
// 					}
// 				}
// 				if(light_vertex_idx < NumBounces)
// 				{
// 					/* HACK... should divide by PI for vertex 0 */
// 					light_throughput = light_throughput * light_vertex.material.brdf();
// 					extend(rng.next2(), light_vertex, &light_ray, &light_throughput);
// 					if(!light_ray.intersect(*scene, &light_vertex)) break;
// 				}
// 			}
// 		}
// 	}
// 	//eye
// 	{		
// 		color summed(0,0,0);
// 		
// 		for(int sample_idx = 0; sample_idx < SamplesPerIteration; sample_idx++)
// 		{			
// 			Random rng(xy, RandomCounter(iteration_idx, sample_idx + 1000));
// 			color value(0,0,0);
// 			color eye_throughput(1,1,1);
// 			ray<World> ray0 = camera_ray(*camera, xy, screen_size);
// 			Hit<World> eye_vertex_1;
// 			if(!ray0.intersect(*scene, &eye_vertex_1)) return;
// 			Hit<World> eye_vertex = eye_vertex_1;
// 			ray<World> eye_ray = ray0;
// 			for(int eye_vertex_idx = 1; eye_vertex_idx < NumBounces + 1; eye_vertex_idx++)
// 			{				
// 				int variations = 2;				
// 				color addition(0,0,0);
// 				if(eye_vertex.material.type == eEmissive) //last vertex is implied to be specular
// 				{
// 					addition = eye_vertex.material.emissive.emission * eye_throughput / (float)(variations);
// 				}
// 				else if(eye_vertex.material.type == eDiffuse)
// 				{
// 					color light_spatial_throughput;
// 					Hit<World> light = 
// 						scene->sample_light(eye_vertex.position, rng.next2(), &light_spatial_throughput);
// 												
// 					color light_throughput = light_spatial_throughput
// 						* connection_throughput(light, eye_vertex, *scene);
// 					
// 					addition = eye_vertex.material.brdf() * light_throughput * eye_throughput / (float)(variations);
// 					
// 				}
// 				if(!addition.is_black())
// 				{
// 					if(BdptDebug)
// 					{
// 						store_bdpt_debug(buffer, addition, width, height, eye_vertex_idx + 1, 
// 							eye_vertex.material.type == eEmissive ? 0 : 1, ref::glm::vec2(xy));
// 					}
// 					else
// 					{	
// 						value = value + addition;
// 					}
// 				}
// 				if(eye_vertex.material.type == eEmissive) break;
// 				if(eye_vertex_idx < NumBounces)
// 				{
// 					eye_throughput = eye_throughput * eye_vertex.material.brdf();
// 					extend(rng.next2(), eye_vertex, &eye_ray, &eye_throughput);
// 
// 					if(!eye_ray.intersect(*scene, &eye_vertex, eye_vertex.material.type == eSpecular)) break;
// 				}
// 			}
// 			summed = summed + value;
// 		}
// 		
// 		if(!BdptDebug)
// 		{
// 			buffer.set(linid, summed);
// 		}
// 	}
// }