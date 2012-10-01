#include "C:\Users\zhaoz3\Documents\lightway2\lightwayrt\python\Random123-1.06\include\Random123\philox.h"
#include "C:\Users\zhaoz3\Documents\lightway2\lightwayrt\python\Random123-1.06\include\Random123\u01.h"
typedef struct 
{
	float4 data[4];
} f4x4;
float4 saturate4(float4 v)
{
	return clamp(v, 0, 1);
}

float3 saturate3(float3 v)
{
	return clamp(v, 0, 1);
}
float4 mul(f4x4 m, float4 v)
{
	return (float4)(
		dot(m.data[0], v),
		dot(m.data[1], v),
		dot(m.data[2], v),
		dot(m.data[3], v)
	);
}

// Group size
#define size_x 32
#define size_y 32

#define VIEWPORT_W 1600
#define VIEWPORT_H 1000


typedef struct
{
	float3 albedo;
} Material;



Material getMaterial(int materialId)
{
	if(materialId == 0) 
	{
		return (Material){(float3)(.7, .8, .6)};
	}
	else if(materialId == 1)
	{
		return (Material){(float3)(.4, .5, .8)};
	}
	else if(materialId == 2)
	{
		return (Material){(float3)(.8, .1, .2)};
	}
	else if(materialId == 3)
	{
		return (Material){(float3)(.7, .7,.7)};
	}
	return (Material){(float3)(1, 0, 0)};
}
typedef struct
{
	float3 normal;
	float3 position;
	int materialId;
	float t;
} Hit;
typedef struct
{
	float3 origin;
	float radius;
	int materialId;
} Sphere;
typedef struct
{
	float y;
	int materialId;
} InfiniteHorizontalPlane;
typedef struct
{
	float3 direction;
	float3 origin;	
} Ray;
Ray makeRay(float3 origin, float3 direction)
{
	Ray ray;
	ray.origin = origin;
	ray.direction = direction;
	return ray;
}
#define INVALID_MATERIAL_ID -1
bool intersectInfiniteHorizontalPlaneG(Ray* this, const global InfiniteHorizontalPlane* plane, Hit* hit)
{
	hit->materialId = INVALID_MATERIAL_ID;
	float3 p0 = (float3)(0, plane->y, 0);
	float3 l0 = this->origin;
	float3 n = (float3)(0, -1, 0);
	float3 l = this->direction;
	float d = dot(p0 - l0, n) / dot(l, n);
	if(d > 0)
	{
		hit->normal = -n;
		hit->position = l0 + d * l;
		hit->t = d;
		hit->materialId = plane->materialId;
		return true;
	}
	return false;
}
bool intersectSphereG(Ray* this, const global Sphere* sphere, Hit* hit)
{	
	hit->materialId = INVALID_MATERIAL_ID;
	float3 l = this->origin - sphere->origin;
	float a = 1;
	float b = 2*dot(this->direction, l);
	float c = dot(l, l)-sphere->radius*sphere->radius;

	float discriminant = b*b - 4*a*c;
	if(discriminant > 0)
	{
		float t0 = (-b-sqrt(discriminant))/(2*a);
		float t1 = (-b+sqrt(discriminant))/(2*a);
		if(t0 > 0 || t1 > 0)
		{
			float t0_clamped = t0 < 0 ? 100000000 : t0;
			float t1_clamped = t1 < 0 ? 100000000 : t1;

			float t = min(t0_clamped, t1_clamped);


			float3 hit_pos = t * this->direction + this->origin;
			float3 normal = normalize(hit_pos - sphere->origin);		
			//don't do normal <-> ray direction check	
			
			hit->normal = normal;
			hit->position = hit_pos;
			hit->materialId = sphere->materialId;
			hit->t = t;
			return true;	
			
		}
	}
	return false;
}
bool intersectInfiniteHorizontalPlane(Ray* this, InfiniteHorizontalPlane* plane, Hit* hit)
{
	hit->materialId = INVALID_MATERIAL_ID;
	float3 p0 = (float3)(0, plane->y, 0);
	float3 l0 = this->origin;
	float3 n = (float3)(0, -1, 0);
	float3 l = this->direction;
	float d = dot(p0 - l0, n) / dot(l, n);
	if(d > 0)
	{
		hit->normal = -n;
		hit->position = l0 + d * l;
		hit->t = d;
		hit->materialId = plane->materialId;
		return true;
	}
	return false;
}
bool intersectSphere(Ray* this, Sphere* sphere, Hit* hit)
{	

	hit->materialId = INVALID_MATERIAL_ID;
	float3 l = this->origin - sphere->origin;
	float a = 1;
	float b = 2*dot(this->direction, l);
	float c = dot(l, l)-sphere->radius*sphere->radius;

	float discriminant = b*b - 4*a*c;
	if(discriminant > 0)
	{
		float t0 = (-b-sqrt(discriminant))/(2*a);
		float t1 = (-b+sqrt(discriminant))/(2*a);
		if(t0 > 0 || t1 > 0)
		{
			float t0_clamped = t0 < 0 ? 100000000 : t0;
			float t1_clamped = t1 < 0 ? 100000000 : t1;

			float t = min(t0_clamped, t1_clamped);


			float3 hit_pos = t * this->direction + this->origin;
			float3 normal = normalize(hit_pos - sphere->origin);		
			//don't do normal <-> ray direction check	
			
			hit->normal = normal;
			hit->position = hit_pos;
			hit->materialId = sphere->materialId;
			hit->t = t;
			return true;	
			
		}
	}
	return false;
}

#define LIGHT_POS ((float3)(0, 10, 0))
float3 brdf(Hit* hit)
{
	float3 lightDir = normalize(LIGHT_POS - hit->position);
	float3 ndotl = dot(hit->normal, lightDir);
	return getMaterial(hit->materialId).albedo * saturate3(ndotl);
}
float3 brdf2(float3 position, float3 normal, int materialId)
{
	float3 lightDir = normalize(LIGHT_POS - position);
	float3 ndotl = dot(normal, lightDir);
	return getMaterial(materialId).albedo * saturate3(ndotl);
}
#define RED (float3)(1, 0, 0)
#define HIT_NEXT_RAY_EPSILON 0.0001f
#define SHADOW_RAY_EPSILON 0.00001f
//LDDE = 2 bounces, LE = 0 bounces, LDDDE = 3 bounces
#define NUM_SPHERES 3
#define NUM_INF_HORIZ_PLANES 1

typedef struct {
	Sphere sphere[NUM_SPHERES];
	InfiniteHorizontalPlane infHorizPlanes[NUM_INF_HORIZ_PLANES];
} Scene;

bool intersectAllGeom(Ray* ray, Scene* scene, Hit* hit)
{
	bool hasHit = false;
	for(int i = 0; i < NUM_SPHERES; i++)
	{
		Hit tempHit;
		if(intersectSphere(ray, &scene->sphere[i], &tempHit)
			&& (!hasHit || (hit->t > tempHit.t)))
		{
			hasHit = true;
			*hit = tempHit;				
		}
	}
	
	for(int i = 0; i < NUM_INF_HORIZ_PLANES; i++)
	{
		Hit tempHit;
		if(intersectInfiniteHorizontalPlane(ray, &scene->infHorizPlanes[i], &tempHit)
			&& (!hasHit || hit->t > tempHit.t))
		{
			hasHit = true;
			*hit = tempHit;				
		}
	}
	
	return hasHit;
}

bool intersectAllGeomG(Ray* ray, const global Scene* scene, Hit* hit)
{
	bool hasHit = false;
	for(int i = 0; i < NUM_SPHERES; i++)
	{
		Hit tempHit;
		if(intersectSphereG(ray, &scene->sphere[i], &tempHit)
			&& (!hasHit || (hit->t > tempHit.t)))
		{
			hasHit = true;
			*hit = tempHit;				
		}
	}
	
	for(int i = 0; i < NUM_INF_HORIZ_PLANES; i++)
	{
		Hit tempHit;
		if(intersectInfiniteHorizontalPlaneG(ray, &scene->infHorizPlanes[i], &tempHit)
			&& (!hasHit || hit->t > tempHit.t))
		{
			hasHit = true;
			*hit = tempHit;				
		}
	}
	
	return hasHit;
}
float2 rand2(uint key, uint2 counter)
{
	philox2x32_ctr_t c;
	philox2x32_key_t k;

	k.v[0] = key;
	c.v[0] = counter.x;
	c.v[1] = counter.y;

	philox2x32_ctr_t rand = philox2x32_R(5, c, k);

	return (float2)(
		u01_open_open_32_24(rand.v[0]),
		u01_open_open_32_24(rand.v[1]),
	);
}
float4 qconj(float4 q)
{
	return q * (float4)(-1, -1, -1, 1);
}
float4 qmul(float4 q1, float4 q2)
{
	float a = q1.w; 
	float c = q2.w;
	float3 b = q1.xyz;
	float3 d = q2.xyz;
	return (float4)(a * d + b * c + cross(b, d), a * c - dot(b, d));
}
float3 changeCoordSys(float3 newUp, float3 oldUp, float3 dir)
{	
	float3 c = cross(oldUp, newUp);
	float4 q = normalize((float4)(c, 1 + dot(newUp, oldUp)));
	return qmul(qmul(q, (float4)(dir, 0)), qconj(q)).xyz;	
}
float3 sampleCosWeightedHemi(float3 nWorld, float2 u, float *pdf)
{
	float a = sqrt(1 - u.y);
	float b = sqrt(u.y);
	float3 wi = (float3)(cos(2 * M_PI_F * u.x) * a, sin(2 * M_PI_F * u.x) * a, b);
	*pdf = b / M_PI_F; //cos(acos(sqrt(u.y))) / pi
	return changeCoordSys(nWorld, (float3)(0, 0, 1), wi);
}
#define NUM_BOUNCES 3
#define NUM_ITERATIONS 300
typedef struct {
	float4 cameraPos;
	f4x4 invView;
	f4x4 invProj;
} ViewParams;
