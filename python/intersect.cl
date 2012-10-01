#include "C:\Users\zhaoz3\Documents\lightway2\lightwayrt\python\shared.h"
kernel void intersect(
	const global Scene* scene,
	const global float* rayOriginX,
	const global float* rayOriginY,
	const global float* rayOriginZ,
	const global float* rayDirectionX,
	const global float* rayDirectionY,
	const global float* rayDirectionZ,
	global float* normalX,
	global float* normalY,
	global float* normalZ,
	//we could use ray params to derive this in the following stages
	global float* positionX, 
	global float* positionY,
	global float* positionZ,	
	global int* materialId)
{
	uint2 pixelXy = (uint2)(get_global_id(0), get_global_id(1));
	uint2 viewportSize = (uint2)(get_global_size(0), get_global_size(1)); 
	uint linid = pixelXy.x + pixelXy.y * viewportSize.x;

	float3 origin = (float3)(rayOriginX[linid], rayOriginY[linid], rayOriginZ[linid]);
	float3 direction = (float3)(rayDirectionX[linid], rayDirectionY[linid], rayDirectionZ[linid]);

	Ray ray = makeRay(origin, direction);

	Hit hit;
	if(intersectAllGeomG(&ray, scene, &hit))
	{
		normalX[linid] = hit.normal.x;
		normalY[linid] = hit.normal.y;
		normalZ[linid] = hit.normal.z;

		positionX[linid] = hit.position.x;
		positionY[linid] = hit.position.y;
		positionZ[linid] = hit.position.z;

		materialId[linid] = hit.materialId;
	}
	else
	{
		materialId[linid] = -1;
	}
}