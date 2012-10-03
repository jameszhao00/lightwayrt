#include "C:\Users\zhaoz3\Documents\lightway2\lightwayrt\python\shared.h"

typedef struct {
	int iterationIdx;
	int bounceIdx;
} BounceParams;

kernel void bounce(	
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
	global half* throughput,
	global half* color,
	constant BounceParams* bounceParams)
{	 

	uint2 pixelXy = (uint2)(get_global_id(0), get_global_id(1));
	uint2 viewportSize = (uint2)(get_global_size(0), get_global_size(1));
	uint linid = pixelXy.x + pixelXy.y * viewportSize.x;

	float4 value = vload_half4(linid, color);

	if(materialId[linid] == -1) //hit nothing
	{
		//vstore4((float4)(value, 1), linid, color);
		return;
	}

	Hit hit;
	hit.normal = (float3)(normalX[linid], normalY[linid], normalZ[linid]);	
	hit.position = (float3)(positionX[linid], positionY[linid], positionZ[linid]);;
	hit.materialId = materialId[linid];	
	float4 throughputVal = vload_half4(linid, throughput);

	if(obstructed[linid] != 1)
	{		
		value += throughputVal * (float4)(brdf(&hit), 1) * M_PI_F; //normalization doesn't look right
	}
	vstore_half4((float4)(value.xyz, 1), linid, color);

	float2 u = rand2(linid, (uint2)(bounceParams->bounceIdx, bounceParams->iterationIdx));

	float pdf;
	float3 wiWorld = sampleCosWeightedHemi(hit.normal, u, &pdf);

	//TODO: this needs to depend on the brdf!
	float4 newThroughput = throughputVal * M_PI_F * getMaterial(hit.materialId).albedo.xyzz;
	vstore_half4((float4)(newThroughput.xyz, 1), linid, throughput);

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
	const global int* materialId,	
	const global half* throughput,
	const global half* iteration_color,
	global float* color,
	constant BounceParams* bounceParams)
{	 

	uint2 pixelXy = (uint2)(get_global_id(0), get_global_id(1));
	uint2 viewportSize = (uint2)(get_global_size(0), get_global_size(1));
	uint linid = pixelXy.x + pixelXy.y * viewportSize.x;

	float4 bounceValue;
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
		float4 throughputVal = vload_half4(linid, throughput);		
		bounceValue = throughputVal * (float4)(brdf(&hit), 1) * M_PI_F; //normalization doesn't look right
	}
	float4 iteration = bounceValue + vload_half4(linid, iteration_color);
	iteration = iteration / (1 + iteration);
	float4 existing = vload4(linid, color);

	float iterationRatio = 1.f / (1 + bounceParams->iterationIdx);
	float4 composite = existing * (1-iterationRatio) + iteration * (iterationRatio);
	//TODO: This isn't right
	vstore4((float4)(composite.xyz, 1), linid, color);

}

kernel void afterEachBounce(
	global BounceParams* bounceParams)
{
	bounceParams[0].bounceIdx++;	
}
kernel void afterFinalBounce(
	global BounceParams* bounceParams)
{
	bounceParams[0].bounceIdx = 0;
	bounceParams[0].iterationIdx++;
}