
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
	float specPower;
} Material;
typedef struct
{
	float3 normal;
	float3 position;
	Material material;
	float t;
} Hit;
typedef struct
{
	float3 origin;
	float radius;
	Material material;
} Sphere;
typedef struct
{
	float y;
	Material material;
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
bool intersectInfiniteHorizontalPlane(Ray* this, InfiniteHorizontalPlane* plane, Hit* hit)
{
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
		hit->material = plane->material;
		return true;
	}
	return false;
}
bool intersectSphere(Ray* this, Sphere* sphere, Hit* hit)
{	
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
			hit->material = sphere->material;
			hit->t = t;
			return true;	
			
		}
	}
	return false;
}

#define LIGHT_POS ((float3)(0, 10, 8))
float3 brdf(Hit* hit)
{
	float3 lightDir = normalize(LIGHT_POS - hit->position);
	float3 ndotl = dot(hit->normal, lightDir);
	return hit->material.albedo * saturate3(ndotl);
}

#define RED (float3)(1, 0, 0)
#define HIT_NEXT_RAY_EPSILON 0.0001f
//LDDE = 2 bounces, LE = 0 bounces, LDDDE = 3 bounces
#define NUM_SPHERES 3
#define NUM_INF_HORIZ_PLANES 1

typedef struct {
	Sphere sphere[NUM_SPHERES];
	InfiniteHorizontalPlane infHorizPlanes[NUM_INF_HORIZ_PLANES];
} Scene;
void initScene(Scene* scene)
{
	scene->sphere[0].origin = (float3)(0, .6, 3);
	scene->sphere[0].radius = .7;
	scene->sphere[0].material.albedo = (float3)(.7, .8, .6);
	
	scene->sphere[1].origin = (float3)(-1, .5, 3);
	scene->sphere[1].radius = .25;
	scene->sphere[1].material.albedo = (float3)(.4, .5, .8);

	scene->sphere[2].origin = (float3)(-1, -1, 3);
	scene->sphere[2].radius = 1;
	scene->sphere[2].material.albedo = (float3)(.8, .1, .2);

	scene->infHorizPlanes[0].y = -1;
	scene->infHorizPlanes[0].material.albedo = (float3)(.7, .7,.7);
}
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
kernel void part1(
	constant ViewParams* viewParam,
	global float* color)
{	 
	uint2 pixelXy = (uint2)(get_global_id(0), get_global_id(1));
	uint2 viewportSize = (uint2)(get_global_size(0), get_global_size(1)); 

	float2 ndc = ((convert_float2(pixelXy) / ((float2)(viewportSize.x, viewportSize.y))) - (float2)(0.5, 0.5))
		* (float2)(2, -2);
	float4 pView = mul(viewParam->invProj, (float4)(ndc, -1, 1));
	pView /= pView.w;
	pView.w = 1;
	float4 pWorld = mul(viewParam->invView, pView);
	
	Ray cameraRay;
	cameraRay.direction = normalize(pWorld.xyz - viewParam->cameraPos.xyz);
	cameraRay.origin = pWorld.xyz;

	Scene scene;
	initScene(&scene);

	float3 value = 0;
	for(uint iterationIdx = 0; iterationIdx < NUM_ITERATIONS; iterationIdx++)
	{
		Ray ray = cameraRay;
		float3 throughput = 1;
		for(uint bounceIdx = 0; bounceIdx < NUM_BOUNCES; bounceIdx++)
		{
			Hit hit;
			if(intersectAllGeom(&ray, &scene, &hit))
			{
				float3 lightDir = normalize(LIGHT_POS - hit.position);
				Ray shadowRay = makeRay(
					hit.position + lightDir * HIT_NEXT_RAY_EPSILON,
					lightDir);
				Hit shadowHit;
				if(!intersectAllGeom(&shadowRay, &scene, &shadowHit))
				{		
					value += throughput * brdf(&hit) * M_PI_F; //normalization doesn't look right	
				}

				if(bounceIdx < NUM_BOUNCES - 1)
				{
					float2 u = rand2(pixelXy.x + pixelXy.y * viewportSize.x, 
						(uint2)(bounceIdx, iterationIdx));

					float pdf;
					float3 wiWorld = sampleCosWeightedHemi(hit.normal, u, &pdf);

					throughput *= M_PI_F * hit.material.albedo;
					
					ray = makeRay(
						hit.position + wiWorld * HIT_NEXT_RAY_EPSILON,
						wiWorld);
				}
			}
			else
			{
				break;
			}
		}
	}
	
	value /= NUM_ITERATIONS;
	value = value / (1 + value);
	/*
	int existingNumSamples = g_time_sampleCount.y;
	float existingRatio = (float)existingNumSamples / (existingNumSamples + 1);
	float newRatio = 1 - existingRatio;
	float4 existing = InputMap.Load(int3(pixelXy, 0));
	OutputMap[pixelXy] = existingRatio * existing + newRatio * float4(value, 1);
	*/	
	
	vstore4((float4)(value, 1), viewportSize.x * pixelXy.y + pixelXy.x, color);
}