#include "C:\Users\zhaoz3\Documents\lightway2\lightwayrt\python\shared.h"

typedef struct {
	int iterationIdx;
	int bounceIdx;
} BounceParams;

kernel void bounce(
	constant BounceParams* bounceParams,
	const global uint* obstructed,
	const global float* normalX,
	const global float* normalY,
	const global float* normalZ,
	const global float* positionX, 
	const global float* positionY,
	const global float* positionZ,
	const global int* materialId,

	global float* rayOriginX,
	global float* rayOriginY,
	global float* rayOriginZ,
	global float* rayDirectionX,
	global float* rayDirectionY,
	global float* rayDirectionZ,

	global float* throughputR, //read write
	global float* throughputG, //read write
	global float* throughputB, //read write
	global float* color)
{	 

	uint2 pixelXy = (uint2)(get_global_id(0), get_global_id(1));
	uint2 viewportSize = (uint2)(get_global_size(0), get_global_size(1));
	uint linid = pixelXy.x + pixelXy.y * viewportSize.x;
	if(materialId[linid] == -1) //hit nothing
	{
		return;
	}
	
	Hit hit;
	hit.normal = (float3)(normalX[linid], normalY[linid], normalZ[linid]);	
	hit.position = (float3)(positionX[linid], positionY[linid], positionZ[linid]);;
	hit.materialId = materialId[linid];
	float3 throughputVal = (float3)(throughputR[linid], throughputG[linid], throughputB[linid]);
	if(obstructed[linid] == 0)
	{		
		float3 value = throughputVal * brdf(&hit) * M_PI_F; //normalization doesn't look right
		//TODO: This isn't right
		vstore4((float4)(value, 1) + vload4(linid, color), linid, color);
	}

	float2 u = rand2(pixelXy.x + pixelXy.y * viewportSize.x, 
		(uint2)(bounceParams->bounceIdx, bounceParams->iterationIdx));

	float pdf;
	float3 wiWorld = sampleCosWeightedHemi(hit.normal, u, &pdf);

	float3 newThroughput = throughputVal * M_PI_F * getMaterial(hit.materialId).albedo;
	throughputR[linid] = newThroughput.x;
	throughputG[linid] = newThroughput.y;
	throughputB[linid] = newThroughput.z;
	
	float3 rayDirection = wiWorld;
	float3 rayOrigin = hit.position + rayDirection * HIT_NEXT_RAY_EPSILON;

	rayOriginX[linid] = rayOrigin.x;
	rayOriginY[linid] = rayOrigin.y;
	rayOriginZ[linid] = rayOrigin.z;

	rayDirectionX[linid] = rayDirection.x;
	rayDirectionY[linid] = rayDirection.y;
	rayDirectionZ[linid] = rayDirection.z;

}

kernel void bounceFinal(
	const global uint* obstructed,
	const global float* normalX,
	const global float* normalY,
	const global float* normalZ,
	const global float* positionX, 
	const global float* positionY,
	const global float* positionZ,
	const global uint* materialId,	
	const global float* throughputR, //read write
	const global float* throughputG, //read write
	const global float* throughputB, //read write

	global float* color)
{	 

	uint2 pixelXy = (uint2)(get_global_id(0), get_global_id(1));
	uint2 viewportSize = (uint2)(get_global_size(0), get_global_size);
	uint linid = pixelXy.x + pixelXy.y * viewportSize.x;

	float3 value;
	if(materialId[linid] == -1 || obstructed[linid] == 1) //hit nothing
	{
		value = 0;
	}
	else
	{	
		Hit hit;
		hit.normal = (float3)(normalX[linid], normalY[linid], normalZ[linid]);	
		hit.position = (float3)(positionX[linid], positionY[linid], positionZ[linid]);;
		hit.materialId = materialId[linid];
		float3 throughputVal = (float3)(throughputR[linid], throughputG[linid], throughputB[linid]);
		value = throughputVal * brdf(&hit) * M_PI_F; //normalization doesn't look right
	}
	value = value / (1 + value);
	//TODO: This isn't right
	vstore4((float4)(value, 1), linid, color);
}