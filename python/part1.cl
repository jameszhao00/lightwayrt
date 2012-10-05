
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
	float3 emission;
	float3 albedo;
	bool is_specular;
	bool is_emissive;
} Material;
typedef struct
{
	float3 origin;
	float radius;
	Material material;
	float height;
} Ring;
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
bool intersectRing(Ray* ray, Ring* ring, Hit* hit)
{
	float3 rd = ray->direction;
	float3 ro = ray->origin - ring->origin;
	float r = ring->radius;
	float discrim = r * r * dot(rd.xz, rd.xz) 
		- rd.x * rd.x * ro.z * ro.z 
		+ 2 * rd.x * rd.z * ro.x * ro.z - rd.z * rd.z * ro.x * ro.x;
	if(discrim > 0)
	{
		float neg_b = - rd.x * ro.x - rd.z * ro.z;
		float root = sqrt(discrim);
		float inv = 1/dot(rd.xz, rd.xz);

		float t0 = (neg_b - root) * inv;
		float t1 = (neg_b + root) * inv;

		bool t0_valid = t0 > 0 && (fabs(ro.y + t0 * ray->direction.y) < ring->height);
		bool t1_valid = t1 > 0 && (fabs(ro.y + t1 * ray->direction.y) < ring->height);

		if(!t0_valid && !t1_valid)
		{
			return false;
		}
		float t = min(t0 + (!t0_valid ? 10000 : 0), t1 + (!t1_valid ? 10000 : 0));
		hit->position = ray->origin + t * ray->direction;
		bool outside = dot(normalize(hit->position - ring->origin), ray->direction) < 0;
		hit->normal = (outside ? 1 : -1) 
			* normalize((float3)(hit->position.xyz - ring->origin.xyz) * (float3)(1,0,1));

		hit->material = ring->material;
		hit->t = t;
		return true;
	}
	//don't care about glance
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
		float det = sqrt(discriminant);		
		float t0 = (-b-det) * .5;
		float t1 = (-b+det) * .5;
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

float3 brdf(Hit* hit)
{
	return hit->material.albedo;
}

#define RED (float3)(1, 0, 0)
#define HIT_NEXT_RAY_EPSILON 0.0001f
//LDDE = 2 bounces, LE = 0 bounces, LDDDE = 3 bounces
#define NUM_SPHERES 3
#define NUM_INF_HORIZ_PLANES 1
#define NUM_RINGS 1
typedef struct {
	Sphere sphere[NUM_SPHERES];
	InfiniteHorizontalPlane infHorizPlanes[NUM_INF_HORIZ_PLANES];
	Ring ring[NUM_RINGS];
} Scene;
void initScene(Scene* scene)
{
	scene->sphere[0].origin = (float3)(0, .6, 3);
	scene->sphere[0].radius = .7;
	scene->sphere[0].material.albedo = (float3)(.7, .8, .6);
	scene->sphere[0].material.is_specular = false;
	scene->sphere[0].material.is_emissive = false;
	
	scene->sphere[1].origin = (float3)(-1, .5, 3);
	scene->sphere[1].radius = .25;
	scene->sphere[1].material.albedo = (float3)(.4, .5, .8);
	scene->sphere[1].material.is_specular = false;
	scene->sphere[1].material.is_emissive = false;

	scene->sphere[2].origin = (float3)(-1, -1, 3);
	scene->sphere[2].radius = 1;
	scene->sphere[2].material.albedo = (float3)(.5, .5, .5);
	scene->sphere[2].material.is_specular = true;
	scene->sphere[2].material.is_emissive = false;

	scene->infHorizPlanes[0].y = -1;
	scene->infHorizPlanes[0].material.albedo = (float3)(.7, .7,.7);
	scene->infHorizPlanes[0].material.is_specular = false;
	scene->infHorizPlanes[0].material.is_emissive = false;

	scene->ring[0].origin = (float3)(0,-.6,3);
	scene->ring[0].radius = 3;
	scene->ring[0].material.albedo = 1;
	scene->ring[0].material.is_specular = true;
	scene->ring[0].material.is_emissive = false;
	scene->ring[0].height = .8;

}
bool intersectAllGeomWithLight(Ray* ray, Scene* scene, Sphere* light, Hit* hit)
{
	bool hasHit = false;
	{
		Hit tempHit;
		if(intersectSphere(ray, light, &tempHit) && (!hasHit || (hit->t > tempHit.t)))
		{
			hasHit = true;
			*hit = tempHit;
		}
	}
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
	for(int i = 0; i < NUM_RINGS; i++)
	{
		Hit tempHit;
		if(intersectRing(ray, &scene->ring[i], &tempHit)
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
	for(int i = 0; i < NUM_RINGS; i++)
	{
		Hit tempHit;
		if(intersectRing(ray, &scene->ring[i], &tempHit)
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

bool shadowIntersectAllGeom(Ray* ray, Scene* scene, float maxT)
{
	for(int i = 0; i < NUM_SPHERES; i++)
	{
		Hit tempHit;
		if(intersectSphere(ray, &scene->sphere[i], &tempHit)
			&& tempHit.t < maxT)
		{
			return true;			
		}
	}
	for(int i = 0; i < NUM_RINGS; i++)
	{
		Hit tempHit;
		if(intersectRing(ray, &scene->ring[i], &tempHit)
			&& tempHit.t < maxT)
		{
			return true;			
		}
	}
	
	for(int i = 0; i < NUM_INF_HORIZ_PLANES; i++)
	{
		Hit tempHit;
		if(intersectInfiniteHorizontalPlane(ray, &scene->infHorizPlanes[i], &tempHit)
			&& tempHit.t < maxT)
		{
			return true;		
		}
	}
	return false;
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
//returns r, theta
float2 sampleDisc(float2 u, float radius)
{
	return (float2)(sqrt(u.x), 2 * M_PI_F * u.y);
}
//don't think this is actually correct, as you could be really close
//to the sphere... and pick points that are occluded by the curvature
float3 sampleSphere(float3 incident, float2 u, Sphere* sphere, float* invPdf)
{
	*invPdf = 4 * M_PI_F * sphere->radius * sphere->radius;
	float xyMultiplier = 2 * sqrt(u.y * (1 - u.y));
	float inner = 2 * M_PI_F * u.x;
	float3 direction = (float3)(cos(inner), sin(inner), 1 - 2 * u.y) 
		* (float3)(xyMultiplier, xyMultiplier, 1);
	direction = dot(direction, -incident) > 0 ? direction : -direction;
	return direction * sphere->radius + sphere->origin;
}
#define NUM_BOUNCES 4
#define NUM_ITERATIONS 100
typedef struct {
	float4 cameraPos;
	f4x4 invView;
	f4x4 invProj;
} ViewParams;
float3 reflect(float3 normal, float3 incident)
{
	return incident - 2 * normal * dot(normal, incident);
}
kernel void part1(
	constant ViewParams* viewParam,
	global float* color)
{	 

    int ltid = get_local_id(0) + get_local_size(0) * get_local_id(1);
	uint2 pixelXy = (uint2)(get_global_id(0), get_global_id(1));
	uint2 viewportSize = (uint2)(get_global_size(0), get_global_size(1)); 
	Scene scene;
	initScene(&scene);
	
	float2 ndc = ((convert_float2(pixelXy) / ((float2)(viewportSize.x, viewportSize.y))) - (float2)(0.5, 0.5))
		* (float2)(2, -2);
	float4 pView = mul(viewParam->invProj, (float4)(ndc, -1, 1));
	pView /= pView.w;
	pView.w = 1;
	float4 pWorld = mul(viewParam->invView, pView);
	
	Ray cameraRay;
	cameraRay.direction = normalize(pWorld.xyz - viewParam->cameraPos.xyz);
	cameraRay.origin = pWorld.xyz;
	
	Sphere light;
	light.origin = (float3)(6,3,2);
	light.radius = .6;
	light.material.is_emissive = true;
	light.material.emission = 100;


	float3 value = 0;
	for(uint iterationIdx = 0; iterationIdx < NUM_ITERATIONS; iterationIdx++)
	{
		Ray ray = cameraRay;
		float3 throughput = 1;
		bool previousBounceSpecular = false;
		for(uint bounceIdx = 0; bounceIdx < NUM_BOUNCES; bounceIdx++)
		{
			Hit hit;
			bool has_hit;
			if(previousBounceSpecular || bounceIdx == 0)
			{
				has_hit = intersectAllGeomWithLight(&ray, &scene, &light, &hit);
			}
			else
			{
				has_hit = intersectAllGeom(&ray, &scene, &hit);
			}
			if(has_hit)
			{
				if((previousBounceSpecular || bounceIdx == 0) && hit.material.is_emissive)
				{
					value += throughput * hit.material.emission;
				}

				if(hit.material.is_specular)
				{
					previousBounceSpecular = true;
				}
				else
				{
					previousBounceSpecular = false;
					float invLightPdf;
					float3 lightSamplePosition;
					{
						float3 lightDir = normalize(light.origin - hit.position);
						float2 u = rand2(pixelXy.x + pixelXy.y * viewportSize.x, 
								(uint2)(bounceIdx + NUM_BOUNCES, iterationIdx));
						lightSamplePosition = sampleSphere(lightDir, u, &light, &invLightPdf);
					}
					float3 lightSampleDir = normalize(lightSamplePosition - hit.position);
					Ray shadowRay = makeRay(
						hit.position + lightSampleDir * HIT_NEXT_RAY_EPSILON,
						lightSampleDir);
					float shadowMaxT = length(lightSamplePosition - shadowRay.origin);
					if(!shadowIntersectAllGeom(&shadowRay, &scene, shadowMaxT))
					{		
						float cos0 = clamp(dot(normalize(lightSamplePosition - light.origin), -lightSampleDir), 0.f, 1.f);
						float cos1 = clamp(dot(hit.normal, lightSampleDir), 0.f, 1.f);
						value += throughput 
							* light.material.emission
							* brdf(&hit)
							* cos0
							* cos1
							* invLightPdf
							/ (shadowMaxT * shadowMaxT);
					}
				}

				if(bounceIdx < NUM_BOUNCES - 1)
				{
					float3 wiWorld;

					if(hit.material.is_specular)
					{
						wiWorld = reflect(hit.normal, ray.direction);
						//no cos() here
						throughput *= hit.material.albedo;
					}
					else
					{				
						float pdf;		
						float2 u = rand2(pixelXy.x + pixelXy.y * viewportSize.x, 
							(uint2)(bounceIdx, iterationIdx));
						wiWorld = sampleCosWeightedHemi(hit.normal, u, &pdf);
						//pdf = cos(theta)/pi for diffuse
						//invPdf = pi / cos(theta)
						float invPdf = M_PI_F / dot(wiWorld, hit.normal); 
						throughput *= hit.material.albedo; //pi canceled out (pdf, lambert)
					}

					
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