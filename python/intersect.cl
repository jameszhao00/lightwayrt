#include "C:\Users\zhaoz3\Documents\lightway2\lightwayrt\python\shared.h"
kernel void intersect(
	const global Scene* scene,
	const global Ray* rays,
	global Hit* hits)
{
	uint2 pixelXy = (uint2)(get_global_id(0), get_global_id(1));
	uint2 viewportSize = (uint2)(get_global_size(0), get_global_size(1)); 
	uint linid = pixelXy.x + pixelXy.y * viewportSize.x;

	Ray ray = rays[linid];

	Hit hit;
	intersectAllGeomG(&ray, scene, &hit);
	hits[linid] = hit;
}