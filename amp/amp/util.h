#include <amp.h>
#include <amp_math.h>
#include <amp_graphics.h>
#include "glm/glm.hpp"
#include "glm/ext.hpp"
using namespace concurrency;
using namespace concurrency::fast_math;
using namespace concurrency::graphics;
using namespace concurrency::graphics::direct3d;
const int WIDTH = 600;
const int HEIGHT = 600;
const int NUM_PIXELS = WIDTH * HEIGHT;

struct float4x4
{
	float4x4() { }
	float4x4(const glm::mat4x4& mat) restrict(cpu)
	{
		memcpy(data, glm::value_ptr(mat), 16 * sizeof(float));
	}
	float4 data[4];
};

float dot(float3 a, float3 b) restrict(cpu, amp)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}
float dot(float4 a, float4 b) restrict(cpu, amp)
{
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
float4 mul(float4x4 a, float4 b) restrict(cpu, amp)
{
	return float4(
		dot(a.data[0], b),
		dot(a.data[1], b),
		dot(a.data[2], b),
		dot(a.data[3], b)
		);
}


struct CameraParams
{
	float4x4 view;
	float4x4 proj;
	float4x4 invView;
	float4x4 invProj;
	CameraParams(int w, int h)
	{		
		auto glmView = glm::lookAt(glm::vec3(0,0,0), glm::vec3(0, 0, -1), glm::vec3(0, 1, 0));
		auto glmProj = glm::perspective(60.f, (float)w/h, 1.f, 1000.f);
		view = float4x4(glmView);
		proj = float4x4(glmProj);
		invView = float4x4(glm::inverse(glmView));
		invProj = float4x4(glm::inverse(glmProj));
	}
};