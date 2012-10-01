#include "C:\Users\zhaoz3\Documents\lightway2\lightwayrt\python\shared.h"
kernel void intersectShadow(
	const global Scene* scene,
	const global Ray* rays,
	const global float* tExpected,
	global uint* obstructed)
{
	uint2 pixelXy = (uint2)(get_global_id(0), get_global_id(1));
	uint2 viewportSize = (uint2)(get_global_size(0), get_global_size(1)); 
	uint linid = pixelXy.x + pixelXy.y * viewportSize.x;

	Ray ray = rays[linid];

	Hit hit;
	if(intersectAllGeomG(&ray, scene, &hit) && 
		(tExpected[linid] - hit.t) > SHADOW_RAY_EPSILON)
	{
		obstructed[linid] = 1;
	}
	else
	{
		obstructed[linid] = 0;
	}
}