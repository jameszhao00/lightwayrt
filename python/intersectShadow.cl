#include "C:\Users\zhaoz3\Documents\lightway2\lightwayrt\python\shared.h"
kernel void intersectShadow(
	const global float* rayOriginX,
	const global float* rayOriginY,
	const global float* rayOriginZ,
	const global float* rayDirectionX,
	const global float* rayDirectionY,
	const global float* rayDirectionZ,
	const global float* tExpected,
	global uint* obstructed)
{
	uint2 pixelXy = (uint2)(get_global_id(0), get_global_id(1));
	uint2 viewportSize = (uint2)(get_global_size(0), get_global_size(1)); 
	uint linid = pixelXy.x + pixelXy.y * viewportSize.x;

	float3 origin = (float3)(rayOriginX[linid], rayOriginY[linid], rayOriginZ[linid]);
	float3 direction = (float3)(rayDirectionX[linid], rayDirectionY[linid], rayDirectionZ[linid]);

	Ray ray = makeRay(origin, direction);
	Scene scene;
	initScene(&scene);

	Hit hit;
	if(intersectAllGeom(&ray, &scene, &hit) && 
		(tExpected[linid] - hit.t) > SHADOW_RAY_EPSILON)
	{
		obstructed[linid] = 1;
	}
	else
	{
		obstructed[linid] = 0;
	}
}