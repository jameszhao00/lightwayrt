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

	float3 value = vload4(linid, color).xyz;

	if(materialId[linid] == -1) //hit nothing
	{
		//vstore4((float4)(value, 1), linid, color);
		return;
	}

	Hit hit;
	hit.normal = (float3)(normalX[linid], normalY[linid], normalZ[linid]);	
	hit.position = (float3)(positionX[linid], positionY[linid], positionZ[linid]);;
	hit.materialId = materialId[linid];		
	float3 throughputVal = (float3)(throughputR[linid], throughputG[linid], throughputB[linid]);

	if(obstructed[linid] != 1)
	{		
		value += throughputVal * brdf(&hit) * M_PI_F; //normalization doesn't look right
	}
	vstore4((float4)(value, 1), linid, color);

	float2 u = rand2(linid, 
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
	constant BounceParams* bounceParams,
	const global uint* obstructed,
	const global float* normalX,
	const global float* normalY,
	const global float* normalZ,
	const global float* positionX, 
	const global float* positionY,
	const global float* positionZ,
	const global int* materialId,	
	const global float* throughputR,
	const global float* throughputG,
	const global float* throughputB,
	const global float* iteration_color,
	global float* color)
{	 

	uint2 pixelXy = (uint2)(get_global_id(0), get_global_id(1));
	uint2 viewportSize = (uint2)(get_global_size(0), get_global_size(1));
	uint linid = pixelXy.x + pixelXy.y * viewportSize.x;

	float3 bounceValue;
	if(materialId[linid] == -1 || obstructed[linid] == 1) //hit nothing
	{
		bounceValue = 0;
	}
	else
	{	
		Hit hit;
		hit.normal = (float3)(normalX[linid], normalY[linid], normalZ[linid]);	
		hit.position = (float3)(positionX[linid], positionY[linid], positionZ[linid]);;
		hit.materialId = materialId[linid];
		float3 throughputVal = (float3)(throughputR[linid], throughputG[linid], throughputB[linid]);
		bounceValue = throughputVal * brdf(&hit) * M_PI_F; //normalization doesn't look right
	}
	float4 iteration = (float4)(bounceValue, 0) + vload4(linid, iteration_color);
	iteration = iteration / (1 + iteration);
	float4 existing = vload4(linid, color);

	float iterationRatio = 1.f / (1 + bounceParams->iterationIdx);
	float4 composite = existing * (1-iterationRatio) + iteration * (iterationRatio);
	//TODO: This isn't right
	vstore4((float4)(composite.xyz, 1), linid, color);
}