//--------------------------------------------------------------------------------
// This file is a portion of the Hieroglyph 3 Rendering Engine.  It is distributed
// under the MIT License, available in the root of this distribution and
// at the following URL:
//
// http://www.opensource.org/licenses/mit-license.php
//
// Copyright (c) 2003-2010 Jason Zink
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
#include "PCH.h"
#include "SpriteRendererDX11.h"
#include "RendererDX11.h"
#include "Log.h"
#include "BlendStateConfigDX11.h"
#include "RasterizerStateConfigDX11.h"
#include "DepthStencilStateConfigDX11.h"
#include "SpriteFontDX11.h"
#include "PipelineManagerDX11.h"
#include "ParameterManagerDX11.h"
#include "BufferConfigDX11.h"
#include "Texture2dConfigDX11.h"
#include "ViewPortDX11.h"
//--------------------------------------------------------------------------------
using namespace Glyph3;
//--------------------------------------------------------------------------------
SpriteRendererDX11::SpriteRendererDX11() :
									m_iLinearSamplerState(-1),
									m_iPointSamplerState(-1),
									m_bInitialized(false)
{

}
//--------------------------------------------------------------------------------
SpriteRendererDX11::~SpriteRendererDX11()
{

}
//--------------------------------------------------------------------------------
bool SpriteRendererDX11::Initialize()
{
	// Get the renderer
	RendererDX11* renderer = RendererDX11::Get();

	// Load the shaders
	m_effect.SetVertexShader( renderer->LoadShader( VERTEX_SHADER,
		std::wstring( L"Sprite.hlsl" ),
		std::wstring( L"SpriteVS" ),
		std::wstring( L"vs_4_0" ) ) );

	if ( m_effect.GetVertexShader() == -1 )
	{
		Log::Get().Write( L"Failed to load sprite vertex shader" );
		return false;
	}

	m_effect.SetPixelShader( renderer->LoadShader( PIXEL_SHADER,
		std::wstring( L"Sprite.hlsl" ),
		std::wstring( L"SpritePS" ),
		std::wstring( L"ps_4_0" ) ) );

	if ( m_effect.GetPixelShader() == -1 )
	{
		Log::Get().Write( L"Failed to load sprite vertex shader" );
		return false;
	}


	// Create our states
	RasterizerStateConfigDX11 rsConfig;
	rsConfig.AntialiasedLineEnable = FALSE;
	rsConfig.CullMode = D3D11_CULL_NONE;
	rsConfig.DepthBias = 0;
	rsConfig.DepthBiasClamp = 1.0f;
	rsConfig.DepthClipEnable = false;
	rsConfig.FillMode = D3D11_FILL_SOLID;
	rsConfig.FrontCounterClockwise = false;
	rsConfig.MultisampleEnable = true;
	rsConfig.ScissorEnable = false;
	rsConfig.SlopeScaledDepthBias = 0;
	m_effect.m_iRasterizerState = renderer->CreateRasterizerState( &rsConfig );

	if ( m_effect.m_iRasterizerState == -1 )
	{
		Log::Get().Write( L"Failed to create sprite rasterizer state" );
		return false;
	}

	BlendStateConfigDX11 blendConfig;
	blendConfig.AlphaToCoverageEnable = false;
	blendConfig.IndependentBlendEnable = false;
	for ( int i = 0; i < 8; ++i )
	{
		blendConfig.RenderTarget[i].BlendEnable = true;
		blendConfig.RenderTarget[i].BlendOp = D3D11_BLEND_OP_ADD;
		blendConfig.RenderTarget[i].SrcBlend = D3D11_BLEND_SRC_ALPHA;
		blendConfig.RenderTarget[i].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
		blendConfig.RenderTarget[i].BlendOpAlpha = D3D11_BLEND_OP_ADD;
		blendConfig.RenderTarget[i].SrcBlendAlpha = D3D11_BLEND_ONE;
		blendConfig.RenderTarget[i].DestBlendAlpha = D3D11_BLEND_ONE;
		blendConfig.RenderTarget[i].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
	}

	m_effect.m_iBlendState = renderer->CreateBlendState( &blendConfig );

	if ( m_effect.m_iBlendState == -1 )
	{
		Log::Get().Write( L"Failed to create sprite blend state" );
		return false;
	}

	DepthStencilStateConfigDX11 dsConfig;
	dsConfig.DepthEnable = FALSE;
	dsConfig.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
	dsConfig.DepthFunc = D3D11_COMPARISON_LESS;
	dsConfig.StencilEnable = false;
	dsConfig.StencilReadMask = D3D11_DEFAULT_STENCIL_READ_MASK;
	dsConfig.StencilWriteMask = D3D11_DEFAULT_STENCIL_WRITE_MASK;
	dsConfig.FrontFace.StencilDepthFailOp = D3D11_STENCIL_OP_KEEP;
	dsConfig.FrontFace.StencilFailOp = D3D11_STENCIL_OP_KEEP;
	dsConfig.FrontFace.StencilPassOp = D3D11_STENCIL_OP_REPLACE;
	dsConfig.FrontFace.StencilFunc = D3D11_COMPARISON_ALWAYS;
	dsConfig.BackFace = dsConfig.FrontFace;

	m_effect.m_iDepthStencilState = renderer->CreateDepthStencilState( &dsConfig );

	if ( m_effect.m_iDepthStencilState == -1 )
	{
		Log::Get().Write( L"Failed to create sprite depth stencil state" );
		return false;
	}

	// Linear filtering sampler state
	D3D11_SAMPLER_DESC samplerConfig;
	samplerConfig.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
	samplerConfig.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
	samplerConfig.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
	samplerConfig.BorderColor[0] = 0;
	samplerConfig.BorderColor[1] = 0;
	samplerConfig.BorderColor[2] = 0;
	samplerConfig.BorderColor[3] = 0;
	samplerConfig.ComparisonFunc = D3D11_COMPARISON_ALWAYS;
	samplerConfig.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
	samplerConfig.MaxAnisotropy = 1;
	samplerConfig.MaxLOD = D3D11_FLOAT32_MAX;
	samplerConfig.MinLOD = 0;
	samplerConfig.MipLODBias = 0;

	m_iLinearSamplerState = renderer->CreateSamplerState( &samplerConfig );

	if ( m_iLinearSamplerState == -1 )
	{
		Log::Get().Write( L"Failed to create sprite sampler state" );
		return false;
	}

	// Point filtering sampler state
	samplerConfig.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;

	m_iPointSamplerState = renderer->CreateSamplerState( &samplerConfig );

	if ( m_iLinearSamplerState == -1 )
	{
		Log::Get().Write( L"Failed to create sprite sampler state" );
		return false;
	}

	m_pGeometry = InstancedQuadGeometryPtr( new InstancedQuadGeometryDX11() );

	m_bInitialized = true;

	return true;
}
//--------------------------------------------------------------------------------
void SpriteRendererDX11::Render( PipelineManagerDX11* pipeline,
								 IParameterManager* parameters,
								 ResourcePtr texture,
								 const SpriteDrawData* drawData,
								 UINT numSprites, FilterMode filterMode )
{
	_ASSERT(m_bInitialized);

	RendererDX11::PIXBeginEvent( L"SpriteRendererDX11 Render" );

	m_pGeometry->ResetGeometry();

	// Make sure the draw rects are all valid
	D3D11_TEXTURE2D_DESC desc = texture->m_pTexture2dConfig->GetTextureDesc();
	for ( UINT i = 0; i < numSprites; ++i )
	{
		SpriteDrawRect drawRect = drawData[i].DrawRect;
		_ASSERT( drawRect.X >= 0 && drawRect.X < desc.Width );
		_ASSERT( drawRect.Y >= 0 && drawRect.Y < desc.Height );
		_ASSERT( drawRect.Width > 0 && drawRect.X + drawRect.Width <= desc.Width );
		_ASSERT( drawRect.Height > 0 && drawRect.Y + drawRect.Height <= desc.Height );

		m_pGeometry->AddQuad( drawData[i] );
	}

	// Set the constants
	Vector4f texAndViewportSize;
	texAndViewportSize.x = static_cast<float>( desc.Width );
	texAndViewportSize.y = static_cast<float>( desc.Height );

	int viewportID = pipeline->RasterizerStage.DesiredState.GetViewport( 0 );
	ViewPortDX11* vp = RendererDX11::Get()->GetViewPort( viewportID );
	texAndViewportSize.z = static_cast<float>( vp->GetWidth() );
	texAndViewportSize.w = static_cast<float>( vp->GetHeight() );

	parameters->SetVectorParameter( L"TexAndViewportSize", &texAndViewportSize );

	// Set the texture
	parameters->SetShaderResourceParameter( L"SpriteTexture", texture );

	// Set the sampler
	if ( filterMode == Linear )
		parameters->SetSamplerParameter( L"SpriteSampler", &m_iLinearSamplerState );
	else if ( filterMode == Point )
		parameters->SetSamplerParameter( L"SpriteSampler", &m_iPointSamplerState );

	pipeline->ClearPipelineResources();
	m_effect.ConfigurePipeline( pipeline, parameters );
	pipeline->ApplyPipelineResources();

	m_pGeometry->Execute( pipeline, parameters );

	RendererDX11::PIXEndEvent();
}
//--------------------------------------------------------------------------------
void SpriteRendererDX11::Render( PipelineManagerDX11* pipeline, 
								 IParameterManager* parameters,
							     ResourcePtr texture, 
								 const Matrix4f& transform,
								 const Vector4f& color, FilterMode filterMode,
								 const SpriteDrawRect* drawRect )
{
	SpriteDrawData data;
	data.Color = color;
	data.Transform = transform;

	if ( drawRect )
		data.DrawRect = *drawRect;
	else
	{
		// Draw the full texture
		D3D11_TEXTURE2D_DESC textureDesc = texture->m_pTexture2dConfig->GetTextureDesc();
		data.DrawRect.X = 0;
		data.DrawRect.Y = 0;
		data.DrawRect.Width = static_cast<float>( textureDesc.Width );
		data.DrawRect.Height = static_cast<float>( textureDesc.Height );
	}

	Render( pipeline, parameters, texture, &data, 1, filterMode );
}
//--------------------------------------------------------------------------------
void SpriteRendererDX11::RenderText( PipelineManagerDX11* pipeline, 
									 IParameterManager* parameters,
									 const SpriteFontDX11& font, const wchar_t* text,
									 const Matrix4f& transform, const Vector4f& color )
{
	RendererDX11::PIXBeginEvent( L"SpriteRenderer RenderText" );

	SpriteDrawData m_TextDrawData [MaxBatchSize];

	size_t length = wcslen( text );

	Matrix4f textTransform = Matrix4f::Identity();

	UINT numCharsToDraw = min( length, MaxBatchSize );
	UINT currentDraw = 0;
	for (UINT i = 0; i < numCharsToDraw; ++i)
	{
		wchar_t character = text[i];
		if(character == ' ')
			textTransform[12] += font.SpaceWidth();
		else if(character == '\n')
		{
			textTransform[13] += font.CharHeight();
			textTransform[12] = 0;
		}
		else
		{
			SpriteFontDX11::CharDesc desc = font.GetCharDescriptor(character);

			m_TextDrawData[currentDraw].Transform = textTransform * transform;
			m_TextDrawData[currentDraw].Color = color;
			m_TextDrawData[currentDraw].DrawRect.X = desc.X;
			m_TextDrawData[currentDraw].DrawRect.Y = desc.Y;
			m_TextDrawData[currentDraw].DrawRect.Width = desc.Width;
			m_TextDrawData[currentDraw].DrawRect.Height = desc.Height;
			currentDraw++;

			textTransform[12] += desc.Width + 1;
		}
	}

	// Submit a batch
	Render( pipeline, parameters, font.TextureResource(), m_TextDrawData, currentDraw, Point );

	RendererDX11::PIXEndEvent();

	if( length > numCharsToDraw )
		RenderText( pipeline, parameters, font, text + numCharsToDraw, textTransform, color );
}
//--------------------------------------------------------------------------------