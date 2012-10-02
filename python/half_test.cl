
kernel void half3_test(
	const global half* input,
	global half* output )
{
	uint linid = get_global_id(0);
	float3 v = vload_half3(linid, input);
	for(int i = 0; i < 20; i++)
	{
		v = v * v - v;
	}
	vstore_half3(v * v, linid, output);
}
kernel void half_test(
	const global half* input,
	global half* output )
{
	uint linid = get_global_id(0);
	float4 v = vload_half4(linid, input);
	for(int i = 0; i < 20; i++)
	{
		v = v * v - v;
	}
	vstore_half4(v * v, linid, output);
}
kernel void half_raw_test(
	const global half* input,
	global half* output )
{
	uint linid = get_global_id(0);
	float4 v = (float4)(input[linid * 4], input[linid * 4 + 1], input[linid * 4 + 2], input[linid * 4 + 3]);
	for(int i = 0; i < 20; i++)
	{
		v = v * v - v;
	}
	vstore_half4(v * v, linid, output);
}

kernel void float_raw_test(
	const global float* input,
	global float* output )
{
	uint linid = get_global_id(0);
	float4 v = (float4)(input[linid * 4], input[linid * 4 + 1], input[linid * 4 + 2], input[linid * 4 + 3]);
	for(int i = 0; i < 20; i++)
	{
		v = v * v - v;
	}
	vstore4(v * v, linid, output);
}

kernel void float_test(
	const global float* input,
	global float* output )
{
	uint linid = get_global_id(0);
	float4 v = vload4(linid, input);
	for(int i = 0; i < 20; i++)
	{
		v = v * v - v;
	}
	vstore4(v * v, linid, output);
}
kernel void float3_test(
	const global float* input,
	global float* output )
{
	uint linid = get_global_id(0);
	float3 v = vload3(linid, input);
	for(int i = 0; i < 20; i++)
	{
		v = v * v - v;
	}
	vstore3(v * v, linid, output);
}


kernel void float_aos_test(
	const global float* inputA,
	const global float* inputB,
	const global float* inputC,
	const global float* inputD,
	global float* output )
{
	uint linid = get_global_id(0);
	float4 v = (float4)(inputA[linid], inputB[linid], inputC[linid], inputD[linid]);
	for(int i = 0; i < 20; i++)
	{
		v = v * v - v;
	}
	vstore4(v * v, linid, output);
}

