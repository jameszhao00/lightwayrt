#include "util.h"
#include "bitmap_image.hpp"
const int WIDTH = 1000;
const int HEIGHT = 1000;
const int NUM_PIXELS = WIDTH * HEIGHT;
const int NUM_ITERATIONS = 1;
int main()
{
	auto av = accelerator(accelerator::default_accelerator).default_view;
	float4* image = new float4[NUM_PIXELS];
	CameraParams params(WIDTH, HEIGHT, glm::vec3(0,0,-1), glm::vec3(0,0,1));
	Scene scene;
	scene.init();
	array_view<float4, 2> output(HEIGHT, WIDTH, image); 
	for(int i = 0; i < 10; i++)
	{
		parallel_for_each(output.extent, [=](index<2> idx) restrict(amp) 
		{ 			
			int2 screenPos(idx[1], idx[0]);
			int2 screenSize(WIDTH, HEIGHT);
			auto eyeRay = getCameraRay(params, screenPos, screenSize);
			float3 value = 0;
			for(int iteration = 0; iteration < NUM_ITERATIONS; iteration++)
			{
				Hit hit;
				if(eyeRay.intersect(scene, &hit))
				{				
					value += dot(-eyeRay.direction, hit.normal)* hit.material.albedo;
				}
			}
			value /= NUM_ITERATIONS;

			output[idx] = float4(eyeRay.origin.x, eyeRay.origin.y, eyeRay.origin.z, 1);
			
		});

		av.flush();
	}

	output.synchronize();
	bitmap_image bmp(WIDTH,HEIGHT);
	for(int i = 0; i < HEIGHT; i++)
	{
		for(int j = 0; j < WIDTH; j++)
		{
			auto f4 = image[i * WIDTH + j] * 255;
			bmp.set_pixel(j, i, (char)f4.x, (char)f4.y, (char)f4.z);
		}
	}
	bmp.save_image("out.bmp");
}