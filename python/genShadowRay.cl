#include "C:\Users\zhaoz3\Documents\lightway2\lightwayrt\python\shared.h"
kernel void genShadowRay(	
	//we could use ray params to derive this in the following stages
	const global Hit* hits,
	global Ray* rays,
	global float* tExpected)
{
	uint2 pixelXy = (uint2)(get_global_id(0), get_global_id(1));
	uint2 viewportSize = (uint2)(get_global_size(0), get_global_size(1)); 
	uint linid = pixelXy.x + pixelXy.y * viewportSize.x;

	float3 position = hits[linid].position;

	float3 rayDirection = normalize(LIGHT_POS - position);
	float3 rayOrigin = position + rayDirection * HIT_NEXT_RAY_EPSILON;

	//take into account epsilon
	tExpected[linid] = length(LIGHT_POS - rayOrigin);
	
	rays[linid] = makeRay(rayOrigin, rayDirection);
}