//
// InvertColorCS.hlsl
//
// Copyright (C) 2010  Jason Zink 

Texture2D<float4>		InputMap : register( t0 );           
RWTexture2D<float4>		OutputMap : register( u0 );

cbuffer Camera 
{
	float4x4 g_invProj;
	float4x4 g_invView;
	float4 g_cameraPosition;
};
cbuffer MC
{
	float4 g_time_sampleCount;
};


#define PI 3.141592654
// Group size
#define size_x 32
#define size_y 32

#define VIEWPORT_W 1000
#define VIEWPORT_H 1000

struct Material
{
	float3 albedo;
};
struct Hit
{
	float3 normal;
	float3 position;
	Material material;
	float t;
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
struct Ray
{
	static Ray make(float3 origin, float3 direction)
	{
		Ray ray;
		ray.origin = origin;
		ray.direction = direction;
		return ray;
	}
	float3 direction;
	float3 origin;
	bool intersectInfiniteHorizontalPlane(InfiniteHorizontalPlane plane, out Hit hit)
	{
		hit = (Hit)0;
		float3 p0 = float3(0, plane.y, 0);
		float3 l0 = this.origin;
		float3 n = float3(0, -1, 0);
		float3 l = this.direction;
		float d = dot(p0 - l0, n) / dot(l, n);
		if(d > 0)
		{
			hit.normal = -n;
			hit.position = l0 + d * l;
			hit.t = d;
			hit.material = plane.material;
			return true;
		}
		return false;
	}
	bool intersectSphere(Sphere sphere, out Hit hit)
	{	
		hit = (Hit) 0;
		float3 l = this.origin - sphere.origin;
		float a = 1;
		float b = 2*dot(this.direction, l);
		float c = dot(l, l)-sphere.radius*sphere.radius;

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


				float3 hit_pos = t * this.direction + this.origin;
				float3 normal = normalize(hit_pos - sphere.origin);
				if(dot(normal, -this.direction) > 0)
				{
					hit.normal = normal;
					hit.position = hit_pos;
					hit.material = sphere.material;
					hit.t = t;
					return true;
				}
			}
		}
		return false;
	}
};

static float3 lightPos = float3(0, 10, 0);
float3 shade(Hit hit)
{
	return hit.material.albedo * saturate(dot(hit.normal, normalize(lightPos - hit.position)));
}
float2 rand_2_0004(in float2 uv)
{
    float noiseX = (frac(sin(dot(uv, float2(12.9898,78.233)      )) * 43758.5453));
    float noiseY = (frac(sin(dot(uv, float2(12.9898,78.233) * 2.0)) * 43758.5453));
    return saturate(float2(noiseX, noiseY));// * 0.004;
}

class Rand
{
	float next()
	{
		v++;
		return rand_2_0004(this.v).x;
	}
	float2 nextf2()
	{
		v++;
		return rand_2_0004(this.v);
	}
	static Rand make(float2 seed)
	{
		Rand rand;
		rand.v = seed;
		return rand;
	}
	float2 v;
};

static const float3 RED = float3(1, 0, 0);
static const float HIT_NEXT_RAY_EPSILON = 0.00001f;
//LDDE = 2 bounces, LE = 0 bounces, LDDDE = 3 bounces
static const int NUM_BOUNCES = 3;
static const int NUM_SPHERES = 3;
static const int NUM_INF_HORIZ_PLANES = 1;
static Sphere sphere[NUM_SPHERES];
static InfiniteHorizontalPlane infHorizPlanes[NUM_INF_HORIZ_PLANES];

void initializeGeom()
{
	sphere[0].origin = float3(1, 1, -5);
	sphere[0].radius = 1;
	sphere[0].material.albedo = float3(.7, .8, .6);
	
	sphere[1].origin = float3(-1, 1.3, -6);
	sphere[1].radius = 1;
	sphere[1].material.albedo = float3(.4, .5, .8);

	sphere[2].origin = float3(-1, -1, -6);
	sphere[2].radius = 1;
	sphere[2].material.albedo = float3(.8, .5, .8);

	infHorizPlanes[0].y = -1;
	infHorizPlanes[0].material.albedo = float3(.7, .5, .4);
}

bool intersectAllGeom(Ray ray, out Hit hit)
{
	bool hasHit = false;
	for(int i = 0; i < NUM_SPHERES; i++)
	{
		Hit tempHit = (Hit)0;
		if(ray.intersectSphere(sphere[i], tempHit)
			&& (!hasHit || hit.t > tempHit.t))
		{
			hasHit = true;
			hit = tempHit;				
		}
	}
		
	for(int i2 = 0; i2 < NUM_INF_HORIZ_PLANES; i2++)
	{
		Hit tempHit = (Hit)0;
		if(ray.intersectInfiniteHorizontalPlane(infHorizPlanes[i2], tempHit)
			&& (!hasHit || hit.t > tempHit.t))
		{
			hasHit = true;
			hit = tempHit;				
		}
	}
	return hasHit;
}
// Declare one thread for each texel of the input texture.
[numthreads(size_x, size_y, 1)]
void CSMAIN( uint3 GroupID : SV_GroupID, uint3 DispatchThreadID : SV_DispatchThreadID, uint3 GroupThreadID : SV_GroupThreadID, uint GroupIndex : SV_GroupIndex )
{		
	initializeGeom();

	int2 pixelXy = DispatchThreadID.xy;

	float2 ndc = ((pixelXy / float2(VIEWPORT_W, VIEWPORT_H)) - 0.5) * float2(2, -2);
	float4 pView = mul(g_invProj, float4(ndc, -1, 1));
	pView /= pView.w;
	pView.w = 1;
	float4 pWorld = mul(g_invView, pView);
	
	Ray ray;
	ray.direction = normalize(pWorld.xyz - g_cameraPosition.xyz);
	ray.origin = pWorld.xyz;

	float3 value = 0;
	
	float2 seed = ndc.xy * 0.5 + 1;
	Rand rand = Rand::make(seed.xy * seed.yx + g_time_sampleCount.x);
	
	for(int bounceIdx = 0; bounceIdx < NUM_BOUNCES; bounceIdx++)
	{
		Hit hit = (Hit)0;
		bool hasHit = intersectAllGeom(ray, hit);

		if(hasHit)
		{
			float3 lightDir = normalize(lightPos - hit.position);
			Ray shadowRay = Ray::make(
				hit.position + lightDir * HIT_NEXT_RAY_EPSILON,
				lightDir);
			Hit shadowHit;
			if(!intersectAllGeom(shadowRay, shadowHit))
			{
				value += shade(hit);
			}
			if(bounceIdx < NUM_BOUNCES - 1)
			{
				float2 u = float2(rand.next(), rand.next());
				float a = sqrt(1 - u.y * u.y);
				float3 wi = float3(cos(2 * PI * u.x) * a, sin(2 * PI * u.x) * a, u.y);
				
				float3 wiWorld = dot(wi, hit.normal) < 0 ? -wi : wi;
				ray = Ray::make(
					hit.position + wiWorld * HIT_NEXT_RAY_EPSILON,
					wiWorld);
			}
		}
		else
		{
			break;
		}
	}
	int existingNumSamples = g_time_sampleCount.y;
	float existingRatio = (float)existingNumSamples / (existingNumSamples + 1);
	float newRatio = 1 - existingRatio;
	float4 existing = InputMap.Load(int3(pixelXy, 0));
	OutputMap[pixelXy] = existingRatio * existing + newRatio * float4(value, 1);
}
