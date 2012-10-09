#include <amp.h>
#include <amp_math.h>
#include <amp_graphics.h>
#include "glm/glm.hpp"
#include "glm/ext.hpp"
using namespace concurrency;
using namespace concurrency::fast_math;
using namespace concurrency::graphics;
using namespace concurrency::graphics::direct3d;

struct float4x4
{
	float4x4() { }
	float4x4(const glm::mat4x4& mat) restrict(cpu)
	{
		memcpy(data, glm::value_ptr(glm::transpose(mat)), 16 * sizeof(float));
	}
	float4 data[4];
};
float min(float a, float b) restrict(cpu, amp)
{
	return a < b ? a : b;
}
float3 min(float3 a, float3 b) restrict(cpu, amp)
{
	return float3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}
float dot(float3 a, float3 b) restrict(cpu, amp)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}
float dot(float4 a, float4 b) restrict(cpu, amp)
{
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
float3 normalize(float3 f3) restrict(cpu, amp)
{
	return f3 / sqrt(dot(f3, f3));
}
float4 mul(float4x4 a, float4 b) restrict(cpu, amp)
{
	return float4(
		dot(a.data[0], b),
		dot(a.data[1], b),
		dot(a.data[2], b),
		dot(a.data[3], b)
		);
}


struct CameraParams
{
	float4x4 view;
	float4x4 proj;
	float4x4 invView;
	float4x4 invProj;
	float3 cameraPos;
	CameraParams(int w, int h, glm::vec3 cameraPos, glm::vec3 target)
	{		
		auto glmView = glm::lookAt(cameraPos, target, glm::vec3(0, 1, 0));
		auto glmProj = glm::perspective(60.f, (float)w/h, 1.f, 1000.f);
		view = float4x4(glmView);
		proj = float4x4(glmProj);
		invView = float4x4(glm::inverse(glmView));
		invProj = float4x4(glm::inverse(glmProj));
		this->cameraPos = float3(cameraPos.x, cameraPos.y, cameraPos.z);
	}
};

struct Material
{
	float3 albedo;
};
struct Sphere
{
	float3 origin;
	float radius;
	Material material;
};
struct InfiniteHorizontalPlane
{
	float y;
	Material material;
};
struct Hit
{
	float3 normal;
	float3 position;
	Material material;
	float t;
};
const int MAX_NUM_SPHERES = 6;
const int MAX_NUM_PLANES = 6;
struct Scene
{
	int numSpheres;
	int numPlanes;
	Sphere spheres[MAX_NUM_SPHERES];
	InfiniteHorizontalPlane planes[MAX_NUM_SPHERES];
	void init()
	{
		numSpheres = 1;
		numPlanes = 1;
		spheres[0].material.albedo = 1;
		spheres[0].origin = float3(0, 0, 0);
		spheres[0].radius = .7;
		planes[0].material.albedo = 1;
		planes[0].y = 0;
	}
};
#define SHADOW_T_EPSILON 0.0001f
struct Ray
{
	float3 origin;
	float3 direction;
	bool intersect(const InfiniteHorizontalPlane& plane, Hit* hit) restrict(amp)
	{
		float3 p0(0, plane.y, 0);
		float3 l0 = this->origin;
		float3 n(0, -1, 0);
		float3 l = this->direction;
		float d = dot(p0 - l0, n) / dot(l, n);
		if(d > 0)
		{
			hit->normal = -n;
			hit->position = l0 + d * l;
			hit->t = d;
			hit->material = plane.material;
			return true;
		}
		return false;
	}
	bool intersect(const Sphere& sphere, Hit* hit) restrict(amp)
	{
		float3 l = origin - sphere.origin;
		float a = 1;
		float b = 2*dot(direction, l);
		float c = dot(l, l)-sphere.radius*sphere.radius;

		float discriminant = b*b - 4*a*c;
		if(discriminant > 0)
		{	
			float det = sqrt(discriminant);		
			float t0 = (-b-det) * .5f;
			float t1 = (-b+det) * .5f;
			if(t0 > 0 || t1 > 0)
			{
				float t0_clamped = t0 < 0 ? 100000000 : t0;
				float t1_clamped = t1 < 0 ? 100000000 : t1;

				float t = min(t0_clamped, t1_clamped);

				float3 hit_pos = t * this->direction + this->origin;
				float3 normal = normalize(hit_pos - sphere.origin);		
				//don't do normal <-> ray direction check	

				hit->normal = normal;
				hit->position = hit_pos;
				hit->material = sphere.material;
				hit->t = t;
				return true;	

			}
		}
		return false;
	}
	bool intersect(const Scene& scene, Hit* hit) restrict(amp)
	{
		bool hasHit = false;
		for(int i = 0; i < scene.numSpheres; i++)
		{
			Hit tempHit;
			if(intersect(scene.spheres[i], &tempHit) && (!hasHit || (hit->t > tempHit.t)))
			{
				hasHit = true;
				*hit = tempHit;
			}
		}
		for(int i = 0; i < scene.numPlanes; i++)
		{
			Hit tempHit;
			if(intersect(scene.planes[i], &tempHit) && (!hasHit || (hit->t > tempHit.t)))
			{
				hasHit = true;
				*hit = tempHit;
			}
		}
		return hasHit;
	}
	bool intersectShadow(const Scene& scene, Hit* hit, float expectedT) restrict(amp)
	{
		bool hasHit = false;
		for(int i = 0; i < scene.numSpheres; i++)
		{
			Hit tempHit;
			if(intersect(scene.spheres[i], &tempHit) && concurrency::fast_math::fabs(tempHit.t - expectedT) < SHADOW_T_EPSILON)
			{
				hasHit = true;
				*hit = tempHit;
			}
		}
		for(int i = 0; i < scene.numPlanes; i++)
		{
			Hit tempHit;
			if(intersect(scene.planes[i], &tempHit) && concurrency::fast_math::fabs(tempHit.t - expectedT) < SHADOW_T_EPSILON)
			{
				hasHit = true;
				*hit = tempHit;
			}
		}
		return hasHit;
	}
};

float2 getNdc(int2 screenPos, int2 screenSize) restrict(amp)
{	
	return (float2(screenPos.x, screenPos.y) / float2(screenSize) - float2(0.5f, 0.5f)) * float2(2.f, -2.f);
}
Ray getCameraRay(const CameraParams& viewParam, int2 screenPos, int2 screenSize) restrict(amp)
{	
	float2 ndc = getNdc(screenPos, screenSize);
	float4 pView = mul(viewParam.invProj, float4(ndc.x, ndc.y, -1, 1));
	pView /= pView.w;
	pView.w = 1;
	float4 pWorld = mul(viewParam.invView, pView);

	Ray cameraRay;
	cameraRay.direction = normalize(pWorld.xyz - viewParam.cameraPos.xyz);
	cameraRay.origin = pWorld.xyz;
	return cameraRay;
}