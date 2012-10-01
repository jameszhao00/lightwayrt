#include "C:\Users\zhaoz3\Documents\lightway2\lightwayrt\python\shared.h"

kernel void emitInitialRays(
	constant ViewParams* viewParam,
	global Ray* rays)
{	 
	uint2 pixelXy = (uint2)(get_global_id(0), get_global_id(1));
	uint2 viewportSize = (uint2)(get_global_size(0), get_global_size(1)); 
	uint linid = pixelXy.x + pixelXy.y * viewportSize.x;

	float2 ndc = ((convert_float2(pixelXy) / ((float2)(viewportSize.x, viewportSize.y))) - (float2)(0.5, 0.5))
		* (float2)(2, -2);
	float4 pView = mul(viewParam->invProj, (float4)(ndc, -1, 1));
	pView /= pView.w;
	pView.w = 1;
	float4 pWorld = mul(viewParam->invView, pView);
	
	float3 direction = normalize(pWorld.xyz - viewParam->cameraPos.xyz);
	float3 origin = pWorld.xyz;

	rays[linid] = makeRay(origin, direction);
}