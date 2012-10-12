#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"
#include "math_constants.h"
#include "Random123/philox.h"
#include "Random123/u01.h"
#define GLM_FORCE_CUDA
#include "glm/glm.hpp"
#include "glm/gtx/quaternion.hpp"
#include "glm/gtx/matrix_operation.hpp"
#include "glm/gtc/matrix_transform.hpp"
#define PI CUDART_PI
#define RAY_EPSILON 0.00001f
#define GPU_CPU __host__ __device__
#define GPU __device__

template<typename T>
GPU_CPU T saturate(T val)
{
	return glm::clamp(val, 0, 1);
}
enum CoordinateSystem
{
	World,
	Local,
	ZUp,
	Unknown
};
struct color : glm::vec3 
{
	GPU_CPU color() : glm::vec3(0.f,0.f,0.f) { }
	GPU_CPU color(float x, float y, float z) : glm::vec3(x,y,z) { }
};


template<CoordinateSystem CS>
struct position : glm::vec3
{
	GPU_CPU position() : glm::vec3() { }
	GPU_CPU position(glm::vec3 v) : glm::vec3(v) { }
	GPU_CPU position(float x, float y, float z) : glm::vec3(x,y,z) { }
};

template<CoordinateSystem CS>
struct direction : glm::vec3
{
	GPU_CPU direction() : glm::vec3() { }
	GPU_CPU direction(glm::vec3 v) : glm::vec3(v) { }
	GPU_CPU direction(const position<CS>& a, const position<CS>& b)
		: glm::vec3(glm::normalize(b - a)) { }
	GPU_CPU direction(float x, float y, float z) : glm::vec3(x,y,z) { }
	GPU_CPU direction<CS> negated() const
	{
		return (direction<CS>)(*this * -1.f);
	}
	GPU_CPU float dot(const direction<CS>& other) const
	{
		glm::vec3 otherv3 = other;
		return glm::dot(*this, otherv3);
	}
	GPU_CPU bool valid() const
	{
		return glm::abs(this->length() - 1.f) < 0.0000001f;
	}
};
typedef glm::mat4x4 mat4x4;
typedef glm::uvec2 uvec2;
typedef glm::uvec4 uvec4;
typedef glm::vec2 vec2;
typedef glm::vec4 vec4;
typedef glm::vec2 NormalizedSphericalCS;
typedef glm::vec2 RandomPair;
typedef float InversePdf;

#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }
GPU vec2 rand2(uvec2 key, uvec2 counter)
{
	philox4x32_ctr_t c={{}};
	philox4x32_key_t k={{}};
	
	k.v[0] = key.x;
	k.v[1] = key.y;
	c.v[0] = counter.x;
	c.v[1] = counter.y;
	philox4x32_ctr_t r = philox4x32(c, k);

	return vec2(u01_open_open_32_24(r.v[0]), u01_open_open_32_24(r.v[1]));
}
struct Camera
{
	Camera() { }
	Camera(position<World> eye, position<World> target)
	{
		this->eye = eye;
		view = glm::lookAt(eye, target, glm::vec3(0.f, 1.f, 0.f));
		inv_view = glm::inverse(view);
		proj = glm::perspective(60.f, 1.f, 1.f, 1000.f);
		inv_proj = glm::inverse(proj);
	}
	position<World> eye;
	mat4x4 inv_view;
	mat4x4 inv_proj;
	mat4x4 view;
	mat4x4 proj;
};
struct Material
{
	GPU_CPU Material() { }
	GPU_CPU Material(color p_albedo) : albedo(p_albedo) { }
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
		spheres[0] = Sphere(position<World>(-1,0,0), 1, Material(color(.7,1.f, .8f)));
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
		return (position<CS>)(origin + dir * t);
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
		float d = glm::dot(p0 - l0, n) / l.dot(n);
		if(d > 0)
		{
			hit->normal = n.negated();
			hit->position = l0 + l * d;// l0 + d * l;
			hit->t = d;
			hit->material = plane.material;
			return true;
		}
		return false;
	}
	GPU_CPU bool intersect_sphere(const Sphere& sphere, Hit<CS>* hit)
	{	
		glm::vec3 l = origin - sphere.origin;
		float a = 1;
		float b = 2*this->dir.dot(l);
		float c = glm::dot(l, l)-sphere.radius*sphere.radius;

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
	GPU_CPU bool intersect_scene(const Scene& scene, Hit<CS>* hit)
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
GPU uvec2 screen_xy()
{	
	return uvec2(
		blockIdx.x * blockDim.x + threadIdx.x, 
		blockIdx.y * blockDim.y + threadIdx.y);
}
GPU_CPU vec2 ndc(uvec2 screen_size, uvec2 screen_pos)
{
	return (vec2(screen_pos) / vec2(screen_size) - 0.5f) * vec2(2, -2);
}
GPU_CPU Ray<World> camera_ray(const Camera& camera, uvec2 screen_pos, uvec2 screen_size)
{
	vec4 view = camera.inv_proj * vec4(ndc(screen_size, screen_pos), -1, 1);
	view = vec4(glm::vec3(view) / view.w, 1.f);
	position<World> world(glm::vec3(camera.inv_view * view));
	return Ray<World>(world, direction<World>(camera.eye, world));
}
GPU_CPU direction<World> changeCoordSys(direction<World> n, direction<ZUp> dir)
{	
	glm::vec3 v3n = n;
	glm::vec3 c = glm::cross(glm::vec3(0.f,0.f,1.f), v3n);
	glm::vec4 q_content = glm::normalize(glm::vec4(c, 1 + n.dot(direction<World>(0.f,0.f,1.f))));
	glm::quat q(q_content.w, glm::vec3(q_content));
	direction<World> out_dir(glm::vec3(glm::rotate(q, dir)));
	assert(out_dir.valid());
	return out_dir;
}
GPU_CPU direction<World> sampleUniformHemi(direction<World> n, vec2 u, float *inv_pdf)
{
	float a = sqrt(1 - u.y * u.y);
	
	direction<ZUp> wi(cos(2 * PI * u.x) * a, sin(2 * PI * u.x) * a, u.y);
	*inv_pdf = 2 * PI; 
	assert(wi.valid());
	return changeCoordSys(n, wi);
}
GPU_CPU direction<World> sampleCosWeightedHemi(direction<World> n, vec2 u, float *inv_pdf)
{
	float a = sqrt(1 - u.y);
	float b = sqrt(u.y);
	
	direction<ZUp> wi;
	wi.z = b;//(cos(2 * PI * u.x) * a, sin(2 * PI * u.x) * a, b);
	sincospif(2 * u.x, &wi.y, &wi.x);
	wi *= glm::vec3(a, a, 1);
	//wi = direction<ZUp>(cos(2 * PI * u.x) * a, sin(2 * PI * u.x) * a, b);
	*inv_pdf = PI / b; //cos(acos(sqrt(u.y))) / pi
	assert(wi.valid());
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


GPU_CPU bool close_to(float test, float expected, float epsilon = 0.000001)
{
	return abs(test - expected) < epsilon;
}
