#include "C:\Users\zhaoz3\Documents\lightway2\lightwayrt\python\shared.h"
kernel void genShadowRay(	
	//we could use ray params to derive this in the following stages
	const global float* positionX, 
	const global float* positionY,
	const global float* positionZ,

	global float* rayOriginX,
	global float* rayOriginY,
	global float* rayOriginZ,
	global float* rayDirectionX,
	global float* rayDirectionY,
	global float* rayDirectionZ,
	global float* tExpected)
{
	uint2 pixelXy = (uint2)(get_global_id(0), get_global_id(1));
	uint2 viewportSize = (uint2)(get_global_size(0), get_global_size(1)); 
	uint linid = pixelXy.x + pixelXy.y * viewportSize.x;

	float3 position = (float3)(positionX[linid], positionY[linid], positionZ[linid]);

	float3 rayDirection = normalize(LIGHT_POS - position);
	float3 rayOrigin = position + rayDirection * HIT_NEXT_RAY_EPSILON;

	rayOriginX[linid] = rayOrigin.x;
	rayOriginY[linid] = rayOrigin.y;
	rayOriginZ[linid] = rayOrigin.z;

	rayDirectionX[linid] = rayDirection.x;
	rayDirectionY[linid] = rayDirection.y;
	rayDirectionZ[linid] = rayDirection.z;

	//take into account epsilon
	tExpected[linid] = length(LIGHT_POS - rayOrigin);
}