#include "C:\Users\zhaoz3\Documents\lightway2\lightwayrt\python\shared.h"
kernel void buildScene(global Scene* scene)
{

    scene->sphere[0].origin = (float3)(0, .6, 3);
    scene->sphere[0].radius = .7;
    scene->sphere[0].materialId = 0;
    
    scene->sphere[1].origin = (float3)(-1, .5, 3);
    scene->sphere[1].radius = .25;
    scene->sphere[1].materialId = 1;

    scene->sphere[2].origin = (float3)(-1, -1, 3);
    scene->sphere[2].radius = 1;
    scene->sphere[2].materialId = 2;

    scene->infHorizPlanes[0].y = -1;
    scene->infHorizPlanes[0].materialId = 3;
    return;
}