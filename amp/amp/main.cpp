#include "util.h"
struct Material
{
	float3 albedo;
};
struct Sphere
{
	float3 origin;
	float radius;
	Material material;
};
struct Hit
{

};
struct Ray
{
	float3 origin;
	float3 direction;
	bool intersect(Sphere* sphere, Hit* hit) restrict(amp)
	{
		float3 l = origin - sphere->origin;
		float a = 1;
		float b = 2*dot(direction, l);
		float c = dot(l, l)-sphere->radius*sphere->radius;

		float discriminant = b*b - 4*a*c;
		if(discriminant > 0)
		{	
			float det = sqrt(discriminant);		
			float t0 = (-b-det) * .5;
			float t1 = (-b+det) * .5;
			if(t0 > 0 || t1 > 0)
			{
				float t0_clamped = t0 < 0 ? 100000000 : t0;
				float t1_clamped = t1 < 0 ? 100000000 : t1;

				float t = min(t0_clamped, t1_clamped);


				float3 hit_pos = t * this->direction + this->origin;
				float3 normal = normalize(hit_pos - sphere->origin);		
				//don't do normal <-> ray direction check	

				hit->normal = normal;
				hit->position = hit_pos;
				hit->material = sphere->material;
				hit->t = t;
				return true;	

			}
		}
		return false;
	}
};
int main()
{
	float* image = new float[NUM_PIXELS];
	CameraParams params(600, 600);
	array_view<float4, 2> output(WIDTH, HEIGHT, image); 
	parallel_for_each(output.extent, [=](index<2> idx) restrict(amp) 
	{ 			
		for(int i = 0; i < 50; i++)
	});
	output.synchronize();
}