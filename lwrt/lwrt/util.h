#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"
#include "math_constants.h"
#include "Random123/philox.h"
#include "Random123/u01.h"
namespace ref
{
#define GLM_FORCE_CUDA
#include "glm/glm.hpp"
#include "glm/gtx/quaternion.hpp"
#include "glm/gtx/matrix_operation.hpp"
#include "glm/gtc/matrix_transform.hpp"
}
#define PI CUDART_PI
#define RAY_EPSILON 0.00001f
#define GPU_CPU __host__ __device__
#define GPU __device__

enum CoordinateSystem
{
	World,
	Local,
	ZUp
};


GPU_CPU bool close_to(float test, float expected, float epsilon = 0.000001f)
{
	return abs(test - expected) < epsilon;
}

struct v3
{
	float x,y,z;	
	GPU_CPU v3() : x(0), y(0), z(0) { }
	GPU_CPU v3(float xyz) : x(xyz), y(xyz), z(xyz) { }
	GPU_CPU v3(float x, float y, float z) : x(x), y(y), z(z) { }
	GPU_CPU void set(float v) { x=v;y=v;z=v; }
};
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
GPU_CPU v3 saturate(const v3& v)
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

GPU_CPU float v3_len(const v3& v) { return sqrt(dot(v, v)); }
GPU_CPU v3 v3_normalized(const v3& v) { return v3_mul(v, (1.f / v3_len(v))); }

struct color : v3
{
	GPU_CPU color() : v3(1) { }
	GPU_CPU color(const v3& v) : v3(v) { }
	GPU_CPU color(float x, float y, float z) : v3(x,y,z) { }
	GPU_CPU color operator*(v3 s) const { return v3_mul(*this, s); }
	GPU_CPU color& operator=(float v) { set(v); return *this;}

	GPU_CPU color operator+(const color& rhs) const { return v3_add(*this, rhs); }
	GPU_CPU color operator-(const color& rhs) const { return v3_sub(*this, rhs); }
	GPU_CPU color operator*(float rhs) const { return v3_mul(*this, rhs); }
	GPU_CPU color operator/(float rhs) const { return *this * (1.f/rhs);	}
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
	GPU_CPU offset<CS> operator*(float s) const { return (offset<CS>)v3_mul(*this, s); }
	GPU_CPU offset<CS> operator+(const offset<CS>& rhs) const { return v3_add(*this, rhs); }
	GPU_CPU offset<CS> operator-(const offset<CS>& rhs) const { return v3_sub(*this, rhs); }
	GPU_CPU offset& operator=(float v) { set(v); return *this;}
	GPU_CPU float length() const { return v3_len(*this); }
};
template<CoordinateSystem CS>
struct direction : v3
{
	GPU_CPU direction() : v3(0) { }
	GPU_CPU direction(const position<CS>& from, const position<CS>& to) : v3(v3_normalized(v3_sub(to, from))) { }
	GPU_CPU direction(float x, float y, float z) : v3(x,y,z) { }
	GPU_CPU direction(const v3& v) : v3(v) { }
	GPU_CPU offset<CS> operator*(float s) const { return offset<CS>(v3_mul(*this, s)); }
	GPU_CPU direction<CS> operator-() const { return direction<CS>(v3_neg(*this)); }
	GPU_CPU v3 operator*(const v3& s) const { return v3_mul(*this, s); }
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

#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

typedef ref::glm::vec2 RandomPair;
GPU RandomPair rand2(RandomKey key, RandomCounter counter)
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
	Camera(position<World> eye, position<World> target)
	{
		this->eye = eye;
		view = ref::glm::lookAt(to_glm(eye), to_glm(target), to_glm(v3(0.f, 1.f, 0.f)));
		inv_view = ref::glm::inverse(view);
		proj = ref::glm::perspective(60.f, 1.f, 1.f, 1000.f);
		inv_proj = ref::glm::inverse(proj);
	}
	position<World> eye;
	ref::glm::mat4x4 inv_view;
	ref::glm::mat4x4 inv_proj;
	ref::glm::mat4x4 view;
	ref::glm::mat4x4 proj;
};
struct Material
{
	GPU_CPU Material() : albedo() { }
	GPU_CPU Material(const color& p_albedo) : albedo(p_albedo) { }
	color albedo;	
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
struct InfiniteHorizontalPlane
{
	GPU_CPU InfiniteHorizontalPlane() { }
	GPU_CPU InfiniteHorizontalPlane(float p_y, Material p_material) 
		: y(p_y),  material(p_material) { }
	float y;
	Material material;
};
#define NUM_SPHERES 2
#define NUM_PLANES 1
struct Scene
{
	Sphere spheres[NUM_SPHERES];
	InfiniteHorizontalPlane planes[NUM_PLANES];
	GPU_CPU Scene( ) 
	{
		spheres[0] = Sphere(position<World>(-1,0,0), 1, Material(color(.7f,1.f, .8f)));
		spheres[1] = Sphere(position<World>(1,0,0), 1, Material(color(1,.7f, .8f)));

		planes[0] = InfiniteHorizontalPlane(0, Material(color(1,1,1)));
	}
};
template<CoordinateSystem CS>
struct Hit
{
	direction<CS> normal;
	position<CS> position;
	Material material;
	float t;
};
template<CoordinateSystem CS>
struct Ray
{
	GPU_CPU Ray() { }
	GPU_CPU Ray(const position<CS>& p_origin, const direction<CS>& p_direction)
		: origin(p_origin), dir(p_direction) { } 
	position<CS> origin;
	direction<CS> dir;
	GPU_CPU position<CS> at(float t)
	{
		return origin + dir * t;
	}
	GPU_CPU Ray<CS> offset_by(float t)
	{
		return Ray<CS>(at(t), dir);
	}
	GPU_CPU bool intersect_plane(const InfiniteHorizontalPlane& plane, Hit<CS>* hit)
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
	GPU_CPU bool intersect_sphere(const Sphere& sphere, Hit<CS>* hit)
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

				position<World> hit_pos = this->at(t);
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
	GPU_CPU bool intersect_shadow(const Scene& scene, float expected_t)
	{
		for(int i = 0; i < NUM_SPHERES; i++)
		{
			Hit<CS> tempHit;
			if(intersect_sphere(scene.spheres[i], &tempHit) && !close_to(tempHit.t, expected_t))
			{
				return true;
			}
		}
		for(int i = 0; i < NUM_PLANES; i++)
		{
			Hit<CS> tempHit;
			if(intersect_plane(scene.planes[i], &tempHit) && !close_to(tempHit.t, expected_t))
			{
				return true;
			}
		}
		return false;
	}
	GPU_CPU bool intersect(const Scene& scene, Hit<CS>* hit)
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
		return has_hit;
	}
};
GPU SreenPosition screen_xy()
{	
	return SreenPosition(
		blockIdx.x * blockDim.x + threadIdx.x, 
		blockIdx.y * blockDim.y + threadIdx.y);
}
GPU_CPU Ndc ndc(ref::glm::uvec2 screen_size, ref::glm::uvec2 screen_pos)
{
	return (ref::glm::vec2(screen_pos) / ref::glm::vec2(screen_size) - 0.5f) * ref::glm::vec2(2, -2);
}
GPU_CPU Ray<World> camera_ray(const Camera& camera, ref::glm::uvec2 screen_pos, ref::glm::uvec2 screen_size)
{
	ref::glm::vec4 view = camera.inv_proj * ref::glm::vec4(ndc(screen_size, screen_pos), -1, 1);
	view = ref::glm::vec4(ref::glm::vec3(view) / view.w, 1.f);
	auto result = ref::glm::vec3(camera.inv_view * view);
	position<World> world(result.x, result.y, result.z);
	return Ray<World>(world, direction<World>(camera.eye, world));
}
GPU_CPU direction<World> changeCoordSys(direction<World> n, direction<ZUp> dir)
{	
	direction<World> c = cross(direction<World>(0.f,0.f,1.f), n);
	ref::glm::vec4 q_content = ref::glm::normalize(ref::glm::vec4(to_glm(c), 1 + dot(n, direction<World>(0.f,0.f,1.f))));
	ref::glm::quat q(q_content.w, ref::glm::vec3(q_content.x, q_content.y, q_content.z));
	auto result = ref::glm::vec3(ref::glm::rotate(q, to_glm(dir)));
	direction<World> out_dir(result.x, result.y, result.z);
	return out_dir;
}
GPU_CPU direction<World> sampleUniformHemi(direction<World> n, ref::glm::vec2 u, float *inv_pdf)
{
	float a = sqrt(1 - u.y * u.y);
	
	direction<ZUp> wi(cos(2 * PI * u.x) * a, sin(2 * PI * u.x) * a, u.y);
	*inv_pdf = 2 * PI; 
	return changeCoordSys(n, wi);
}
GPU_CPU direction<World> sampleCosWeightedHemi(direction<World> n, ref::glm::vec2 u, float *inv_pdf)
{
	float a = sqrt(1 - u.y);
	float b = sqrt(u.y);
	
	direction<ZUp> wi;
	wi.z = b;//(cos(2 * PI * u.x) * a, sin(2 * PI * u.x) * a, b);
	sincospif(2 * u.x, &wi.y, &wi.x);
	wi = wi * v3(a, a, 1);
	//wi = direction<ZUp>(cos(2 * PI * u.x) * a, sin(2 * PI * u.x) * a, b);
	*inv_pdf = PI / b; //cos(acos(sqrt(u.y))) / pi
	return changeCoordSys(n, wi);
}
GPU_CPU float pdfCosWeightedHemi(float theta)
{
	return cos(theta)/PI;
}
template<CoordinateSystem CS>
GPU_CPU NormalizedSphericalCS spherical(direction<CS> xyz) //returns theta, phi
{
	return NormalizedSphericalCS(acos(xyz.z), atan2(xyz.x, xyz.y));
}
