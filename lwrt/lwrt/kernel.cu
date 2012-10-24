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
	GPU void elementwise_atomic_add(int idx, const v3& v3)
	{
		atomicAdd(x + idx, v3.x);
		atomicAdd(y + idx, v3.y);
		atomicAdd(z + idx, v3.z);
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
struct ColorSample
{
	int linid;
	color value;
};
struct Pass
{
	Pass(int iteration_idx, int num_iterations, int num_bounces, bool bdpt_debug) 
		: iteration_idx(iteration_idx), num_iterations(num_iterations), num_bounces(num_bounces), bdpt_debug(bdpt_debug) { }
	int iteration_idx;
	int num_iterations;
	int num_bounces;
	bool bdpt_debug;
};
GPU_CPU ref::glm::uvec2 world_to_screen(position<World> world, 
	const Camera& camera, 
	int width, int height,
	bool* in_bounds,
	ref::glm::vec2& ndc)
{
	auto pos_view = camera.view * ref::glm::vec4(to_glm(world), 1);
	auto pos_clip = camera.proj * pos_view;
	auto ndc_4 = pos_clip / pos_clip.w;
	*in_bounds = ndc_4.x > -1 && ndc_4.y > -1 && ndc_4.x < 1 && ndc_4.y < 1 && ndc_4.z > -1 && ndc_4.z < 1;
	ndc = ref::glm::vec2(ndc_4);
	return ref::glm::uvec2(ref::glm::floor(
		(ref::glm::vec2(ndc) * ref::glm::vec2(.5, -.5) + ref::glm::vec2(.5, .5)) * ref::glm::vec2(width, height)));
}
struct Random
{
	GPU_CPU __forceinline__ Random(RandomKey key, RandomCounter base_counter) : key(key), counter(base_counter) { }
	__device__ __forceinline__ RandomPair next2() 
	{
		counter.y++;
		return rand2(key, counter);
	}
	RandomKey key;
	RandomCounter counter;
};

#ifndef LW_UNIT_TEST
surface<void, cudaSurfaceType2D> output_surf;
GPU_ENTRY void transfer_image(Vec3Buffer new_buffer, Vec3Buffer existing_buffer, const Pass* pass, int width, int height)
{
	ref::glm::uvec2 screen_size(width, height);		
#ifndef LW_CPU
	ref::glm::uvec2 xy = screen_xy();
#endif
	if(ref::glm::any(ref::glm::greaterThan(xy, screen_size - ref::glm::uvec2(1)))) return;
	int linid = xy.y * width + xy.x;

	color existing = existing_buffer.get(linid);
	float existing_weight = (float)pass->iteration_idx / (pass->iteration_idx + pass->num_iterations);
	color combined = (color(new_buffer.get(linid)) / (float)pass->num_iterations) * (1 - existing_weight);
	if(pass->iteration_idx > 0)
	{
		combined = combined + existing * (existing_weight);
	}
	new_buffer.set(linid, v3(0,0,0));
	
	existing_buffer.set(linid, combined);
	color combined_tonemapped = combined / (combined + color(1,1,1));
	surf2Dwrite(make_float4(combined_tonemapped.x, combined_tonemapped.y, combined_tonemapped.z, 1), output_surf, xy.x*sizeof(float4), xy.y);
	
}
GPU_CPU ref::glm::uvec2 component_image_position(int width, int height, int eye_verts_count, int light_verts_count, int component_size,
	int original_x, int original_y)
{
	int center_x = width / 2;
	int total_verts = eye_verts_count + light_verts_count; //starts at 3
	int total_components = total_verts + 1; //4 images at 3 verts
	int y_idx = total_verts - 3; //implicit path with length=1 = 2 verts...
	int x_idx = light_verts_count;
	int y = component_size * y_idx;
	int x = center_x - (float)total_components / 2.f * component_size + eye_verts_count * component_size;
	return ref::glm::uvec2(x, y) 
		+ ref::glm::uvec2(ref::glm::vec2((float)original_x / width, (float)original_y / height) * ref::glm::vec2(component_size, component_size));
}
GPU_CPU color connection_throughput(const Hit<World>& light, const Hit<World>& eye, const Scene& scene)
{
	if(light.material.is_specular || eye.material.is_specular) return color(0,0,0);

	ray<World> shadow(light.position, eye.position);
	if(shadow.offset_by(RAY_EPSILON).intersect_shadow(scene, eye.position)) return color(0,0,0);
		
	offset<World> disp = eye.position - light.position;
	float d = disp.length();
	direction<World> dir(light.position, eye.position);
	float cos_light = clamp01(dot(dir, light.normal));
	float cos_eye = clamp01(-dot(dir, eye.normal));
	float g = cos_light * cos_eye / (d * d);
	return g;
}
GPU void store_bdpt_debug(Vec3Buffer& buffer, color value, int width, int height, int ev_count, int lv_count, ref::glm::uvec2 xy)
{
	int component_size = 200;
	ref::glm::uvec2 component_xy = component_image_position(width, height, ev_count, lv_count, component_size, xy.x, xy.y);
	color add = value * (float)(component_size * component_size) / (width * height);
	buffer.elementwise_atomic_add(component_xy.y * width + component_xy.x, add);
}
GPU void extend_bdpt(Random& rng, const Hit<World>& hit, ray<World>* path_ray, color* throughput)
{
	direction<World> wi;
	if(!hit.material.is_specular)
	{
		InverseProjectedPdf ippdf;
		wi = sampleCosWeightedHemi(hit.normal, rng.next2(), &ippdf);					
		*throughput = *throughput * ippdf;
	}
	else
	{
		wi = path_ray->dir.reflect(hit.normal);
		*throughput = *throughput * hit.material.albedo;
	}
	*path_ray = ray<World>(hit.position, wi)
					.offset_by(RAY_EPSILON);
}
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
	color summed(0,0,0);
	ray<World> ray0 = camera_ray(*camera, xy, screen_size);
	Hit<World> eye_vertex_1;
	if(!ray0.intersect(*scene, &eye_vertex_1)) return;

	for(int iteration_idx = pass->iteration_idx; iteration_idx < (pass->iteration_idx + pass->num_iterations); iteration_idx++)
	{
		Random rng(xy, RandomCounter(iteration_idx, 0));
		color value(0,0,0);
		
		Hit<World> light_vertex;
		color light_spatial_le;
		float light_spatial_ipdf;
		light_vertex.position = sample_sphere_light(scene->sphere_lights[0], rng.next2(), &light_spatial_le,
			&light_spatial_ipdf);
		light_vertex.normal = direction<World>(scene->sphere_lights[0].origin, light_vertex.position);
		light_vertex.material = scene->sphere_lights[0].material;
		
		color light_throughput = light_spatial_le * light_spatial_ipdf;

		ray<World> light_ray;
		for(int light_vertex_idx = 0; light_vertex_idx < pass->num_bounces + 1; light_vertex_idx++)
		{			
			color eye_throughput(1,1,1);
			Hit<World> eye_vertex = eye_vertex_1;
			ray<World> eye_ray = ray0;
			for(int eye_vertex_idx = 1; eye_vertex_idx < pass->num_bounces + 1; eye_vertex_idx++)
			{
				int num_vertices = light_vertex_idx + 1 + eye_vertex_idx + 1;
				if(num_vertices > pass->num_bounces + 2) break;
				int variations = num_vertices + 1 - 2;//s + t + 1	
				color light_brdf;
				if(light_vertex_idx == 0) light_brdf = 1;
				else light_brdf = light_vertex.material.brdf();
				color addition = eye_vertex.material.brdf() * light_brdf
					* light_throughput * eye_throughput * connection_throughput(eye_vertex, light_vertex, *scene)
					 / (float)(variations);
				if(pass->bdpt_debug)
				{
					store_bdpt_debug(buffer, addition, width, height, eye_vertex_idx + 1, light_vertex_idx + 1, xy);
				}
				else
				{	
					value = value + addition;
				}
				if(eye_vertex_idx < pass->num_bounces)
				{
					eye_throughput = eye_throughput * eye_vertex.material.brdf();
					extend_bdpt(rng, eye_vertex, &eye_ray, &eye_throughput);
					if(!eye_ray.intersect(*scene, &eye_vertex)) break;
				}
			}
			
			if(light_vertex_idx < pass->num_bounces)
			{
				if(light_vertex_idx > 0)
				{
					light_throughput = light_throughput * light_vertex.material.brdf();
				}
				extend_bdpt(rng, light_vertex, &light_ray, &light_throughput);
				if(!light_ray.intersect(*scene, &light_vertex)) break;
			}
			if(light_vertex_idx == 0)
			{
				light_throughput = light_throughput / PI; //le(0->1)
			}
		}

		summed = summed + value;
	}
	if(!pass->bdpt_debug)
	{
		buffer.set(linid, summed);
	}
}
/*
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
	color summed(0,0,0);
	ray<World> ray0 = camera_ray(*camera, xy, screen_size);
	Hit<World> eye_vertex_1;
	if(!ray0.intersect(*scene, &eye_vertex_1)) return;

	for(int iteration_idx = pass->iteration_idx; iteration_idx < (pass->iteration_idx + pass->num_iterations); iteration_idx++)
	{
		Random rng(xy, RandomCounter(iteration_idx, 0));
		color value(0,0,0);
		//gen vertex on light
		//gen light path vertex
		//shadow
		int eye_vertex_idx = 1;
		int light_vertex_idx = 0;
		Hit<World> light_vertex;
		Hit<World> eye_vertex = eye_vertex_1;
		color light_spatial_le;
		float light_spatial_ipdf;
		light_vertex.position = sample_sphere_light(scene->sphere_lights[0], rng.next2(), &light_spatial_le,
			&light_spatial_ipdf);
		light_vertex.normal = direction<World>(scene->sphere_lights[0].origin, light_vertex.position);
		light_vertex.material = scene->sphere_lights[0].material;
		//TODO: add eye0-light0, eye1-light0
		{
			color addition = eye_vertex.material.brdf()
				* light_spatial_le * light_spatial_ipdf
				* connection_throughput(eye_vertex, light_vertex, *scene); //spatial emission scale... gets cancelled out later
			if(pass->bdpt_debug)
			{
				store_bdpt_debug(buffer, addition, width, height, eye_vertex_idx + 1, light_vertex_idx + 1, xy);
			}
			else
			{	
				value = value + addition;
			}
		}
		direction<World> prev_light_ray_dir; //unset initially... light ray will initially be area source
		direction<World> prev_eye_ray_dir = ray0.dir;
		
		color light_angular_le = 1.f/PI;
		color light_throughput = light_spatial_le * light_spatial_ipdf * light_angular_le;
		color eye_throughput(1,1,1);


		while(eye_vertex_idx + light_vertex_idx < pass->num_bounces + 1)
		{
			//generate next light vertex		
			if(light_vertex_idx > 0)
			{
				light_throughput = light_throughput * light_vertex.material.brdf(); //do this with the last vertex
			}
			{	
				direction<World> wi;
				if(!light_vertex.material.is_specular)
				{
					InverseProjectedPdf light_ippdf;
					wi = sampleCosWeightedHemi(light_vertex.normal, rng.next2(), &light_ippdf);
					light_throughput = light_throughput * light_ippdf;
				}
				else
				{
					wi = prev_light_ray_dir.reflect(light_vertex.normal);
					//this doesn't make sense
					light_throughput = light_throughput * light_vertex.material.albedo;
				}
				ray<World> light_ray = ray<World>(light_vertex.position, wi)
					.offset_by(RAY_EPSILON);
				prev_light_ray_dir = light_ray.dir;
				if(!light_ray.intersect(*scene, &light_vertex)) break;
				light_vertex_idx++;
			}
			
			color eye_addition = eye_vertex.material.brdf() * light_vertex.material.brdf() *
					connection_throughput(light_vertex, eye_vertex, *scene) * eye_throughput * light_throughput;
			
			if(pass->bdpt_debug)
			{				
				store_bdpt_debug(buffer, eye_addition, width, height, eye_vertex_idx + 1, light_vertex_idx + 1, xy);
			}
			else
			{	
				value = value + eye_addition;
			}
			//TODO: connect new light vertex to eye v0
			
			
			if(eye_vertex_idx + light_vertex_idx >= pass->num_bounces + 1) break;
			//generate next eye vertex			
			eye_throughput = eye_throughput * eye_vertex.material.brdf(); //do this with the last vertex
			{
				direction<World> wi;
				if(!eye_vertex.material.is_specular)
				{
					InverseProjectedPdf eye_dir_ippdf;
					wi = sampleCosWeightedHemi(eye_vertex.normal, rng.next2(), &eye_dir_ippdf);					
					eye_throughput = eye_throughput * eye_dir_ippdf;
				}
				else
				{
					wi = prev_eye_ray_dir.reflect(eye_vertex.normal);
					eye_throughput = eye_throughput * eye_vertex.normal;
				}
				ray<World> eye_ray = ray<World>(eye_vertex.position, wi)
					.offset_by(RAY_EPSILON);
				prev_eye_ray_dir = eye_ray.dir;
				if(!eye_ray.intersect(*scene, &eye_vertex)) break;
				eye_vertex_idx++;
			}
			color light_addition = eye_vertex.material.brdf() * light_vertex.material.brdf() *
					connection_throughput(light_vertex, eye_vertex, *scene) * eye_throughput * light_throughput;
			if(pass->bdpt_debug)
			{
				store_bdpt_debug(buffer, light_addition, width, height, eye_vertex_idx + 1, light_vertex_idx + 1, xy);
			}
			else
			{	
				value = value + light_addition;
			}
		}
		summed = summed + value;
	}
	if(!pass->bdpt_debug)
	{
		buffer.set(linid, summed);
	}
}
*/
/*
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
	
	for(int iteration_idx = pass->iteration_idx; 
		iteration_idx < (pass->iteration_idx + pass->num_iterations); 
		iteration_idx++)
	{
		
		Random rng(xy, RandomCounter(iteration_idx, 0));
		ray<World> light_ray;
		InversePdf light_v0_ipdf;
		light_ray.origin = sample_sphere(scene->sphere_lights[0], rng.next2(), &light_v0_ipdf);
		InverseProjectedPdf light_v0v1_ippdf;
		light_ray.dir = sampleCosWeightedHemi(direction<World>(scene->sphere_lights[0].origin, light_ray.origin), rng.next2(), &light_v0v1_ippdf);
		color light_throughput = scene->sphere_lights[0].material.emission * light_v0_ipdf * light_v0v1_ippdf;
		light_ray = light_ray.offset_by(RAY_EPSILON);
		for(int bounce_idx = 0; bounce_idx < pass->num_bounces; bounce_idx++)
		{
			Hit<World> light_vn;
			if(light_ray.intersect(*scene, &light_vn, false))
			{		
				if(!light_vn.material.is_specular)
				{
					//unproject to camera u,v
					bool in_bounds;
					ref::glm::vec2 ndc;
					ref::glm::uvec2 uv = world_to_screen(light_vn.position, *camera, width, height, &in_bounds, ndc);
					if(in_bounds)
					{

						ray<World> light_to_eye_shadow_ray = ray<World>(light_vn.position, camera->eye)
							.offset_by(RAY_EPSILON);
						if(!light_to_eye_shadow_ray.intersect_shadow(*scene, camera->eye))
						{							
							float costheta_shadow_ev = -dot(light_to_eye_shadow_ray.dir, camera->forward);
							float costheta_shadow_lv = dot(light_to_eye_shadow_ray.dir, light_vn.normal);
							float d = (light_vn.position - camera->eye).length();
							float g = costheta_shadow_ev * costheta_shadow_lv / (d * d);
							float a = powf(2 * tanf(0.5 * (camera->fovy / 180) * PI), 2);
							float we = 1
								/ (a * costheta_shadow_ev * costheta_shadow_ev * costheta_shadow_ev);
							color value = light_throughput * light_vn.material.brdf() * g * we;	
							

							if(pass->bdpt_debug)
							{
								int component_size = 200;
								ref::glm::uvec2 component_xy = component_image_position(width, height, 1, bounce_idx + 2, component_size, uv.x, uv.y);
								value = value * (float)(component_size * component_size) / (width * height);
								buffer.elementwise_atomic_add(component_xy.y * width + component_xy.x, value);
							}
							else
							{								
								buffer.elementwise_atomic_add(uv.y * width + uv.x, value);
							}
						}
					}
				}
				//next ray
				if(bounce_idx < pass->num_bounces - 1)
				{
					if(light_vn.material.is_specular)
					{
						light_ray = ray<World>(light_vn.position, light_ray.dir.reflect(light_vn.normal))
							.offset_by(RAY_EPSILON);
						light_throughput = light_throughput * light_vn.material.albedo; //unchanged
					}
					else
					{
						
						InverseProjectedPdf ippdf;
						direction<World> wi = sampleCosWeightedHemi(light_vn.normal, rng.next2(), &ippdf);
						light_ray = ray<World>(light_vn.position, wi)
							.offset_by(RAY_EPSILON);

						light_throughput = light_throughput * ippdf * light_vn.material.brdf();
					}
				}
			}
			else
			{
				break;
			}
		}
	}
}
*/
void Kernel::setup(cudaGraphicsResource* output, int width, int height)
{
	framebuffer_resource = output;
	CUDA_CHECK_RETURN(cudaMalloc((void**) &camera_ptr, sizeof(Camera)));
	CUDA_CHECK_RETURN(cudaMalloc((void**) &scene_ptr, sizeof(Scene)));
	CUDA_CHECK_RETURN(cudaMalloc((void**) &pass_ptr, sizeof(Pass)));
	//CUDA_CHECK_RETURN(cudaMalloc((void**) &buffer, sizeof(ref::glm::vec4) * width * height));
	new_buffer = new Vec3Buffer();
	new_buffer->init(width * height);
	existing_buffer = new Vec3Buffer();
	existing_buffer->init(width * height);
}
void Kernel::execute(int iteration_idx, int iterations, int bounces, int width, int height, bool bdpt_debug)
{	
	Pass pass(iteration_idx, iterations, bounces, bdpt_debug);
	Camera camera(position<World>(0,9,-7), position<World>(0,0,1));
	Scene scene;

	CUDA_CHECK_RETURN(cudaMemcpy(camera_ptr, &camera, sizeof(Camera), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(scene_ptr, &scene, sizeof(Scene), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(pass_ptr, &pass, sizeof(Pass), cudaMemcpyHostToDevice));

	dim3 threadPerBlock(16, 16, 1);
	dim3 blocks((unsigned int)ceil(width / (float)threadPerBlock.x), 
		(unsigned int)ceil(height / (float)threadPerBlock.y), 1);	
	gfx_kernel<<<blocks, threadPerBlock>>>(*new_buffer, (Camera*)camera_ptr, 
		(Scene*)scene_ptr, (Pass*)pass_ptr, width, height);


	
	cudaArray_t framebuffer_ptr;
	
	CUDA_CHECK_RETURN(cudaGraphicsMapResources(1, &framebuffer_resource));
	CUDA_CHECK_RETURN(cudaGraphicsSubResourceGetMappedArray(&framebuffer_ptr, framebuffer_resource, 0, 0));
	CUDA_CHECK_RETURN(cudaBindSurfaceToArray(output_surf, framebuffer_ptr));

	transfer_image<<<blocks, threadPerBlock>>>(*new_buffer, 
		*existing_buffer, (Pass*)pass_ptr, width, height);
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