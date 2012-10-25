#pragma once
#define RAY_EPSILON 0.001f

#ifdef LW_UNIT_TEST
#define LW_CPU
#endif

#ifdef LW_CPU
	#undef __CUDACC__
	#define GPU_ENTRY
	#define GPU_CPU
#else
	#define GPU_ENTRY __global__
	#define GPU_CPU __device__ __host__
	#include "cuda_runtime.h"
	#include "device_launch_parameters.h"
	#include "math_functions.h"
	#include "math_constants.h"
#endif
	#define GPU __device__

#ifdef LW_CPU
	#define LW_ASSERT(X) assert((X));
#else
	#define LW_ASSERT(X)
#endif

#include "Random123/philox.h"
#include "Random123/u01.h"
namespace ref
{
#include "glm/glm.hpp"
#include "glm/gtx/quaternion.hpp"
#include "glm/gtx/matrix_operation.hpp"
#include "glm/gtc/matrix_transform.hpp"
}

__device__ __forceinline__ unsigned int laneId()
{
	unsigned int ret;
	asm("mov.u32 %0, %laneid;" : "=r"(ret) );
	return ret;
}
template<typename T>
GPU T shuffle_up_wrap(T var, int delta)
{
	int target_laneId = (laneId() + delta);
	target_laneId = (target_laneId > (warpSize - 1)) ? target_laneId - 32 : target_laneId;
	return __shfl(var,  target_laneId);
}
#define PI 3.1415926535897931e+0
enum CoordinateSystem
{
	World,
	Local,
	ZUp
};


GPU_CPU bool close_to(float test, float expected, float epsilon = 0.001f)
{
	return abs(test - expected) < epsilon;
}

struct v3
{
	float x,y,z;	
	GPU_CPU v3() : x(0), y(0), z(0) { }
	GPU_CPU v3(float3 f3) : x(f3.x), y(f3.y), z(f3.z) { }
	GPU_CPU v3(float xyz) : x(xyz), y(xyz), z(xyz) { }
	GPU_CPU v3(float x, float y, float z) : x(x), y(y), z(z) { }
	GPU_CPU void set(float v) { x=v;y=v;z=v; }
	GPU v3 shuffle_up(unsigned int delta) const 
	{
		if(delta == 0) return *this;
		return v3(shuffle_up_wrap(x, delta), shuffle_up_wrap(y, delta), shuffle_up_wrap(z, delta));
	}
	GPU v3 shuffle(unsigned int idx) const
	{
		return v3(__shfl(x, idx), __shfl(y, idx), __shfl(z, idx));
	}
};
GPU_CPU bool all_equal(const v3& a, const v3& b, float epsilon = 0.00000001f)
{
	return close_to(a.x, b.x, epsilon) && close_to(a.y, b.y, epsilon) && close_to(a.z, b.z, epsilon);
}
GPU_CPU v3 clamp(const v3& v, float min_val, float max_val)
{
	return v3(
		max(min(v.x, max_val), min_val),
		max(min(v.y, max_val), min_val),
		max(min(v.z, max_val), min_val));
}
GPU_CPU float clamp(float v, float min_val, float max_val)
{
	return max(min(v, max_val), min_val);
}
GPU_CPU v3 clamp01(const v3& v)
{
	return clamp(v, 0, 1);
}
GPU_CPU float clamp01(float v)
{
	return clamp(v, 0, 1);
}
GPU_CPU float dot(const v3& lhs, const v3& rhs)
{
	return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}
GPU_CPU v3 v3_add(const v3& lhs, const v3& rhs) { return v3(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z); }
GPU_CPU v3 v3_neg(const v3& v) { return v3(-v.x, -v.y, -v.z); }
GPU_CPU v3 v3_sub(const v3& lhs, const v3& rhs) { return v3_add(lhs, v3_neg(rhs)); }

GPU_CPU v3 v3_mul(const v3& v, float m) { return v3(v.x*m,v.y*m,v.z*m); }
GPU_CPU v3 v3_mul(const v3& a, const v3& b) { return v3(a.x*b.x,a.y*b.y,a.z*b.z); }

GPU_CPU v3 v3_div(const v3& v, float m) { float inv = 1/m; return v3_mul(v, inv); }
GPU_CPU v3 v3_div(const v3& a, const v3& b) { v3 inv(1/b.x, 1/b.y, 1/b.z); return v3_mul(a, inv); }

GPU_CPU float v3_len(const v3& v) { return sqrt(dot(v, v)); }
GPU_CPU v3 v3_normalized(const v3& v) { return v3_mul(v, (1.f / v3_len(v))); }

struct color : v3
{
	GPU_CPU color() : v3(0) { }
	GPU_CPU color(const float4& v) : v3(v.x, v.y, v.z) { }
	GPU_CPU color(const v3& v) : v3(v) { }
	GPU_CPU color(float x, float y, float z) : v3(x,y,z) { }
	GPU_CPU color operator*(v3 s) const { return v3_mul(*this, s); }
	GPU_CPU color& operator=(float v) { set(v); return *this;}
	GPU_CPU float4 to_f4() const { return make_float4(x, y, z, 1); }

	GPU_CPU color operator+(const color& rhs) const { return v3_add(*this, rhs); }
	GPU_CPU color operator-(const color& rhs) const { return v3_sub(*this, rhs); }
	GPU_CPU color operator*(float rhs) const { return v3_mul(*this, rhs); }
	GPU_CPU color operator/(float rhs) const { return *this * (1.f/rhs); }
	GPU_CPU color operator/(const color& rhs) const { return v3_div(*this, rhs); }
	GPU_CPU bool is_black() const { return x == 0 && y == 0 && z == 0; }
	GPU_CPU float lum() const { return 0.2126 * x + 0.7152 * y + 0.0722 * z;}
};
template<CoordinateSystem CS>
struct position : v3
{
	GPU_CPU position() : v3(0) { }
	GPU_CPU position(const v3& v) : v3(v) { }
	GPU_CPU position(float x, float y, float z) : v3(x,y,z) { }
	GPU_CPU position& operator=(float v) { set(v); return *this;}
};
GPU_CPU ref::glm::vec3 to_glm(const v3& v)
{
	return ref::glm::vec3(v.x, v.y, v.z);
}
template<CoordinateSystem CS>
struct offset : v3
{
	GPU_CPU offset(const v3& v) : v3(v) { }
	GPU_CPU offset(float x, float y, float z) : v3(x,y,z) { }
	GPU_CPU offset<CS> operator-() const { return offset<CS>(v3_neg(*this)); }
	GPU_CPU offset<CS> operator*(float s) const { return offset<CS>(v3_mul(*this, s)); }
	GPU_CPU offset<CS> operator*(const v3& rhs) const { return offset<CS>(v3_mul(*this, rhs)); }
	GPU_CPU offset<CS> operator+(const offset<CS>& rhs) const { return v3_add(*this, rhs); }
	GPU_CPU offset<CS> operator-(const offset<CS>& rhs) const { return v3_sub(*this, rhs); }
	GPU_CPU offset& operator=(float v) { set(v); return *this;}
	GPU_CPU float length() const { return v3_len(*this); }
};
template<CoordinateSystem CS>
struct direction : v3
{
	GPU_CPU direction() : v3(0) { }
	GPU_CPU direction(const position<CS>& from, const position<CS>& to)
		: v3(v3_normalized(v3_sub(to, from))) { }
	GPU_CPU direction(float x, float y, float z) : v3(x,y,z) { }
	GPU_CPU direction(const v3& v) : v3(v3_normalized(v)) { }
	GPU_CPU offset<CS> operator*(float s) const { return offset<CS>(v3_mul(*this, s)); }
	GPU_CPU direction<CS> operator-() const { return direction<CS>(v3_neg(*this)); }
	GPU_CPU v3 operator*(const v3& s) const { return v3_mul(*this, s); }
	GPU_CPU direction<CS> reflect(const direction<CS>& n) const
	{
		return v3_sub(*this, n * dot(n, *this) * 2);
	}
	GPU_CPU bool is_normalized() const { return abs(v3_len(*this) - 1) < 0.00001f;}
};
template<CoordinateSystem CS>
GPU_CPU direction<CS> cross(const direction<CS>& x, const direction<CS>& y)
{
	return direction<CS>(
		x.y * y.z - y.y * x.z,
		x.z * y.x - y.z * x.x,
		x.x * y.y - y.x * x.y);
}
template<CoordinateSystem CS>
GPU_CPU offset<CS> operator-(const position<CS>& lhs, const position<CS>& rhs)
{
	return offset<CS>(v3_sub(lhs, rhs)); 
} 
template<CoordinateSystem CS>
GPU_CPU position<CS> operator+(const position<CS>& lhs, const offset<CS>& rhs)
{
	return position<CS>(v3_add(lhs, rhs)); 
} 
template<CoordinateSystem CS>
GPU_CPU position<CS> operator-(const position<CS>& lhs, const offset<CS>& rhs)
{
	return lhs + -rhs;
} 


typedef ref::glm::vec2 NormalizedSphericalCS;
typedef ref::glm::uvec2 RandomKey;
typedef ref::glm::uvec2 RandomCounter;
typedef ref::glm::uvec2 SreenPosition;
typedef ref::glm::vec2 Ndc;
typedef ref::glm::vec2 Size;
typedef float InversePdf;
typedef float InverseProjectedPdf;
typedef color InverseProjectedPdf3;

typedef ref::glm::vec2 RandomPair;
#ifndef LW_CPU 
GPU
#endif
RandomPair rand2(RandomKey key, RandomCounter counter)
{
	philox4x32_ctr_t c={{}};
	philox4x32_key_t k={{}};
	
	k.v[0] = key.x;
	k.v[1] = key.y;
	c.v[0] = counter.x;
	c.v[1] = counter.y;
	philox4x32_ctr_t r = philox4x32(c, k);

	return RandomPair(u01_open_open_32_24(r.v[0]), u01_open_open_32_24(r.v[1]));
}
struct Camera
{
	Camera() { }
	Camera(position<World> eye, position<World> target, float aspect_ratio)
	{
		fovy = 60;

		this->eye = eye;
		view = ref::glm::lookAt(to_glm(eye), to_glm(target), to_glm(v3(0.f, 1.f, 0.f)));
		inv_view = ref::glm::inverse(view);
		proj = ref::glm::perspective(fovy, aspect_ratio, 1.f, 1000.f);
		inv_proj = ref::glm::inverse(proj);
		forward = direction<World>(eye, target);
	}
	float fovy;
	position<World> eye;
	direction<World> forward;
	ref::glm::mat4x4 inv_view;
	ref::glm::mat4x4 inv_proj;
	ref::glm::mat4x4 view;
	ref::glm::mat4x4 proj;
};
enum MaterialType
{
	eSpecular,
	eEmissive,
	eDiffuse
};
struct Material
{
	GPU_CPU static GPU_CPU Material make_specular(color albedo)
	{
		Material material;
		material.type = eSpecular;
		material.specular.albedo = albedo;
		return material;
	}
	GPU_CPU static GPU_CPU Material make_diffuse(color albedo) 
	{
		Material material;
		material.type = eDiffuse;
		material.diffuse.albedo = albedo;
		return material;
	}
	GPU_CPU static GPU_CPU Material make_emissive(color emission)
	{
		Material material;
		material.type = eEmissive;
		material.emissive.emission = emission;
		return material;
	}

	MaterialType type;
	struct 
	{
		color albedo;
	} specular;
	struct
	{
		color albedo;
	} diffuse;
	struct  
	{
		color emission;
	} emissive;
	GPU_CPU color brdf() const
	{		
		if(type == eEmissive) { return 1; }
		else if(type == eDiffuse) { return this->diffuse.albedo / PI; }
		else if(type == eSpecular) { return 1; }
		
	}
	GPU Material shuffle(unsigned int idx) const 
	{
		Material material;
		material.type = (MaterialType)__shfl((int)type, idx);
		material.diffuse.albedo = diffuse.albedo.shuffle(idx);
		material.specular.albedo = specular.albedo.shuffle(idx);
		material.emissive.emission = emissive.emission.shuffle(idx);
		return material;
	}
	GPU Material shuffle_up(unsigned int delta) const 
	{
		if(delta == 0) return *this;
		Material material;
		material.type = (MaterialType)shuffle_up_wrap((int)type, delta);
		material.diffuse.albedo = diffuse.albedo.shuffle_up(delta);
		material.specular.albedo = specular.albedo.shuffle_up(delta);
		material.emissive.emission = emissive.emission.shuffle_up(delta);
		return material;
	}
};
struct Sphere
{
	GPU_CPU Sphere() { }
	GPU_CPU Sphere(position<World> p_origin, float p_radius, Material p_material) 
		: origin(p_origin), radius(p_radius), material(p_material) { }
	position<World> origin;
	float radius;
	Material material;
};
struct Ring
{
	position<World> origin;
	float radius;
	float height;
	Material material;
	GPU_CPU Ring() { }
	GPU_CPU Ring(position<World> origin, float radius, float height, Material p_material) 
		: origin(origin), radius(radius), height(height), material(p_material) { }

};
struct InfiniteHorizontalPlane
{
	GPU_CPU InfiniteHorizontalPlane() { }
	GPU_CPU InfiniteHorizontalPlane(float p_y, Material p_material) 
		: y(p_y),  material(p_material) { }
	float y;
	Material material;
};
GPU_CPU direction<World> changeCoordSys(direction<World> n, direction<ZUp> dir)
{	
	direction<World> c = cross(direction<World>(0.f,0.f,1.f), n);
	ref::glm::vec4 q_content = ref::glm::normalize(ref::glm::vec4(to_glm(c), 1 + dot(n, direction<World>(0.f,0.f,1.f))));
	ref::glm::quat q(q_content.w, ref::glm::vec3(q_content.x, q_content.y, q_content.z));
	auto result = ref::glm::vec3(ref::glm::rotate(q, to_glm(dir)));
	direction<World> out_dir(result.x, result.y, result.z);
	return out_dir;
}
GPU_CPU position<World> sample_sphere_light(const Sphere& sphere, RandomPair u, 
	color* spatial_throughput)
{
	//assert(sphere.material.type == eEmissive);
	float a = sqrtf(u.y * (1 - u.y));
	*spatial_throughput = sphere.material.emissive.emission * PI 
		* 4 * PI * sphere.radius * sphere.radius;
	return position<World>(
		2 * sphere.radius * cosf(2 * PI * u.x) * a + sphere.origin.x, 
		2 * sphere.radius * sinf(2 * PI * u.x) * a + sphere.origin.y,
		sphere.radius * (1 - 2 * u.y) + sphere.origin.z);
}
GPU_CPU direction<World> sampleUniformHemi(direction<World> n, ref::glm::vec2 u, InverseProjectedPdf *inv_pdf)
{
	float a = sqrt(1 - u.y * u.y);

	direction<ZUp> wi(cosf(2 * PI * u.x) * a, sinf(2 * PI * u.x) * a, u.y);
	*inv_pdf = 2. * PI * wi.z;
	return changeCoordSys(n, wi);
}
template<CoordinateSystem CS>
struct Hit
{
	direction<CS> normal;
	position<CS> position;
	Material material;
	float t;

	GPU Hit<CS> shuffle(unsigned int idx)
	{
		Hit<CS> result;
		result.normal = normal.shuffle(idx);
		result.position = position.shuffle(idx);
		result.t = __shfl(t, idx);
		result.material = material.shuffle(idx);
		return result;
	}
	GPU Hit<CS> shuffle_up(unsigned int delta)
	{
		Hit<CS> result;
		result.normal = normal.shuffle_up(delta);
		result.position = position.shuffle_up(delta);
		result.t = shuffle_up_wrap(t, delta);
		result.material = material.shuffle_up(delta);
		return result;
	}
};
#define NUM_SPHERES 4
#define NUM_SPHERE_LIGHTS 1
#define NUM_PLANES 1
#define NUM_RINGS 1
struct Scene
{
	Sphere spheres[NUM_SPHERES];
	Sphere sphere_lights[NUM_SPHERE_LIGHTS];
	InfiniteHorizontalPlane planes[NUM_PLANES];
	Ring rings[NUM_RINGS];
	GPU_CPU Scene( ) 
	{
		spheres[0] = Sphere(position<World>(3,0,0), 1, Material::make_diffuse(color(1,0,0)));
		spheres[1] = Sphere(position<World>(1,0,0), 1, Material::make_diffuse(color(0,1,0)));
		spheres[2] = Sphere(position<World>(2, 2, 0), 1.5f, Material::make_diffuse(color(0,0,1)));
		spheres[3] = Sphere(position<World>(5, 5, 0), 3.f, Material::make_diffuse(color(1.f,1,1)));
		planes[0] = InfiniteHorizontalPlane(0,Material::make_diffuse(color(1,1,1)));
		rings[0] = Ring(position<World>(0,0,0), 5, 1,Material::make_specular(color(1,1,1)));

		sphere_lights[0] = Sphere(position<World>(10, 6, 0), 2,Material::make_emissive(color(20,20,20)));
	}
	GPU_CPU Hit<World> sample_light(position<World> pos, RandomPair u, color* spatial_throughput) const
	{
		float hemi_inv_pdf;
		direction<World> wi = sampleUniformHemi(direction<World>(sphere_lights[0].origin, pos), u, &hemi_inv_pdf);
		float r_squared = sphere_lights[0].radius * sphere_lights[0].radius;
		position<World> sample_pos = sphere_lights[0].origin + wi * sphere_lights[0].radius;

		*spatial_throughput = sphere_lights[0].material.emissive.emission * PI
			* 2 * PI * r_squared;

		Hit<World> hit;
		hit.material = sphere_lights[0].material;
		hit.normal = direction<World>(sphere_lights[0].origin, sample_pos);
		hit.position = sample_pos;
		return hit;
	}
};
template<CoordinateSystem CS>
struct ray
{
	GPU_CPU ray() { }
	GPU_CPU ray(const position<CS>& p_origin, const direction<CS>& p_direction)
		: origin(p_origin), dir(p_direction) { } 
	GPU_CPU ray(const position<CS>& from, const position<CS>& to)
		: origin(from), dir(from, to) { } 
	position<CS> origin;
	direction<CS> dir;
	GPU_CPU position<CS> at(float t) const
	{
		return origin + dir * t;
	}
	GPU_CPU ray<CS> offset_by(float t) const
	{
		return ray<CS>(at(t), dir);
	}
	GPU_CPU bool intersect_ring(const Ring& ring, Hit<CS>* hit) const
	{
		auto rd = dir;
		auto ro = origin - ring.origin;
		float r = ring.radius;
		float rd_xz_dot = (rd.x * rd.x + rd.z * rd.z);
		float discrim = r * r * rd_xz_dot// dot(rd.xz, rd.xz) 
			- rd.x * rd.x * ro.z * ro.z 
			+ 2 * rd.x * rd.z * ro.x * ro.z - rd.z * rd.z * ro.x * ro.x;
		if(discrim > 0)
		{
			float neg_b = - rd.x * ro.x - rd.z * ro.z;
			float root = sqrt(discrim);
			float inv = 1/rd_xz_dot;

			float t0 = (neg_b - root) * inv;
			float t1 = (neg_b + root) * inv;

			bool t0_valid = t0 > 0 && (abs(ro.y + t0 * dir.y) < ring.height);
			bool t1_valid = t1 > 0 && (abs(ro.y + t1 * dir.y) < ring.height);

			if(!t0_valid && !t1_valid)
			{
				return false;
			}
			float t = min(t0 + (!t0_valid ? 10000 : 0), t1 + (!t1_valid ? 10000 : 0));
			hit->position = origin + dir * t;
			bool outside = dot(direction<CS>(ring.origin, hit->position), dir) < 0;
			hit->normal = direction<CS>((hit->position - ring.origin) * v3(1,0,1))
				* (outside ? 1 : -1);

			hit->material = ring.material;
			hit->t = t;
			return true;
		}
		//don't care about glance
		return false;
	}
	GPU_CPU bool intersect_plane(const InfiniteHorizontalPlane& plane, Hit<CS>* hit) const
	{
		position<CS> p0(0, plane.y, 0);
		position<CS> l0 = this->origin;
		direction<CS> n(0, -1, 0);
		direction<CS> l = this->dir;
		float d = dot(p0 - l0, n) / dot(l, n);
		if(d > 0)
		{
			hit->normal = -n;
			hit->position = l0 + l * d;
			hit->t = d;
			hit->material = plane.material;
			return true;
		}
		return false;
	}
	GPU_CPU bool intersect_sphere(const Sphere& sphere, Hit<CS>* hit) const
	{	
		offset<CS> l = origin - sphere.origin;
		float a = 1;
		float b = 2*dot(dir, l);
		float c = dot(l, l)-sphere.radius*sphere.radius;

		float discriminant = b*b - 4*a*c;
		if(discriminant > 0)
		{	
			float det = sqrt(discriminant);		
			float t0 = (-b-det) * .5;
			float t1 = (-b+det) * .5;
			if(t0 > 0 || t1 > 0)
			{
				float t0_clamped = t0 < 0 ? 100000000 : t0;
				float t1_clamped = t1 < 0 ? 100000000 : t1;

				float t = min(t0_clamped, t1_clamped);

				position<World> hit_pos = at(t);
				//don't do normal <-> ray direction check	

				hit->normal = direction<World>(sphere.origin, hit_pos);
				hit->position = hit_pos;
				hit->material = sphere.material;
				hit->t = t;
				return true;
			}
		}
		return false;		
	}
#define SHADOW_EPSILON 0.01f
	GPU_CPU bool intersect_shadow(const Scene& scene, const position<CS> target) const
	{
		float max_t = (target - origin).length() - SHADOW_EPSILON;
		for(int i = 0; i < NUM_SPHERES; i++)
		{
			Hit<CS> tempHit;
			if(intersect_sphere(scene.spheres[i], &tempHit) && tempHit.t < max_t)
			{
				return true;
			}
		}
		for(int i = 0; i < NUM_PLANES; i++)
		{
			Hit<CS> tempHit;
			if(intersect_plane(scene.planes[i], &tempHit) && tempHit.t < max_t)
			{
				return true;
			}
		}
		for(int i = 0; i < NUM_RINGS; i++)
		{
			Hit<CS> tempHit;
			if(intersect_ring(scene.rings[i], &tempHit) && tempHit.t < max_t)
			{
				return true;
			}
		}
		return false;
	}
	GPU_CPU bool intersect(const Scene& scene, Hit<CS>* hit, bool include_light = false) const
	{
		bool has_hit = false;
		for(int i = 0; i < NUM_SPHERES; i++)
		{
			Hit<CS> tempHit;
			if(intersect_sphere(scene.spheres[i], &tempHit) && (!has_hit || tempHit.t < hit->t))
			{
				has_hit = true;
				*hit = tempHit;
			}
		}
		
		for(int i = 0; i < NUM_PLANES; i++)
		{
			Hit<CS> tempHit;
			if(intersect_plane(scene.planes[i], &tempHit) && (!has_hit || tempHit.t < hit->t))
			{
				has_hit = true;
				*hit = tempHit;
			}
		}
		for(int i = 0; i < NUM_RINGS; i++)
		{
			Hit<CS> tempHit;
			if(intersect_ring(scene.rings[i], &tempHit) && (!has_hit || tempHit.t < hit->t))
			{
				has_hit = true;
				*hit = tempHit;
			}
		}
		if(include_light)
		{
			for(int i = 0; i < NUM_SPHERE_LIGHTS; i++)
			{
				Hit<CS> tempHit;
				if(intersect_sphere(scene.sphere_lights[i], &tempHit) && (!has_hit || tempHit.t < hit->t))
				{
					has_hit = true;
					*hit = tempHit;
				}
			}
		}
		return has_hit;
	}
};
#ifndef LW_CPU
GPU SreenPosition screen_xy()
{	
	return SreenPosition(
		blockIdx.x * blockDim.x + threadIdx.x, 
		blockIdx.y * blockDim.y + threadIdx.y);
}
#endif
GPU_CPU Ndc ndc(ref::glm::uvec2 screen_size, ref::glm::uvec2 screen_pos)
{
	return (ref::glm::vec2(screen_pos) / ref::glm::vec2(screen_size) - 0.5f) * ref::glm::vec2(2, -2);
}
GPU_CPU ray<World> camera_ray(const Camera& camera, ref::glm::uvec2 screen_pos, ref::glm::uvec2 screen_size)
{
	ref::glm::vec4 view = camera.inv_proj * ref::glm::vec4(ndc(screen_size, screen_pos), -1, 1);
	view = ref::glm::vec4(ref::glm::vec3(view) / view.w, 1.f);
	auto result = ref::glm::vec3(camera.inv_view * view);
	position<World> world(result.x, result.y, result.z);
	return ray<World>(world, direction<World>(camera.eye, world));
}
GPU_CPU direction<World> sampleCosWeightedHemi(direction<World> n, ref::glm::vec2 u, InverseProjectedPdf *inv_pdf)
{
	float a = sqrt(1 - u.y);
	float b = sqrt(u.y);
	
	direction<ZUp> wi;
	wi.z = b;//(cos(2 * PI * u.x) * a, sin(2 * PI * u.x) * a, b);
	sincospif(2 * u.x, &wi.y, &wi.x);
	wi = wi * v3(a, a, 1);
	//wi = direction<ZUp>(cos(2 * PI * u.x) * a, sin(2 * PI * u.x) * a, b);
	*inv_pdf = PI;// / b; //cos(acos(sqrt(u.y))) / pi
	return changeCoordSys(n, wi);
}
GPU_CPU float pdfCosWeightedHemi(float theta)
{
	return cosf(theta)/PI;
}
template<CoordinateSystem CS>
GPU_CPU NormalizedSphericalCS spherical(direction<CS> xyz) //returns theta, phi
{
	return NormalizedSphericalCS(acosf(xyz.z), atan2f(xyz.x, xyz.y));
}

GPU_CPU bool bit_set(unsigned int x, int bit)
{
	return (x & (1 << bit));
}