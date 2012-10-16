//--------------------------------------------------------------------------------
// FinalPS.hlsl
//
// This pixel shader samples the ambient occlusion buffer and uses its value to 
// modify the ambient term of a standard lighting equation.
//
// Copyright (C) 2009 Jason Zink.  All rights reserved.
//--------------------------------------------------------------------------------
Texture2D<float4> Input;// : register(t0);
//Texture2D       ColorMap00 : register( t0 );   
cbuffer DisplayParams 
{
	float4 g_width_height;
};
struct VS_OUTPUT
{
	float4 position : SV_Position;
};

float4 PSMAIN( in VS_OUTPUT input ) : SV_Target
{
	int3 screenspace = int3( input.position.x, input.position.y, 0 );
	float4 color = Input.Load(screenspace);
	return float4(color.xyz, 1);
}

