//--------------------------------------------------------------------------------
// This file is a portion of the Hieroglyph 3 Rendering Engine.  It is distributed
// under the MIT License, available in the root of this distribution and 
// at the following URL:
//
// http://www.opensource.org/licenses/mit-license.php
//
// Copyright (c) 2003-2010 Jason Zink 
//--------------------------------------------------------------------------------
#include "App.h"
#include "Log.h"

#include <sstream>

#include "EventManager.h"
#include "EvtFrameStart.h"
#include "EvtChar.h"
#include "EvtKeyUp.h"
#include "EvtKeyDown.h"

#include "ScriptManager.h"

#include "SwapChainConfigDX11.h"
#include "Texture2dConfigDX11.h"
#include "BufferConfigDX11.h"
#include "GeometryGeneratorDX11.h"
#include "RenderEffectDX11.h"

#include "IParameterManager.h"

#include <iostream>
#include "glm/glm.hpp"
#include "glm/ext.hpp"
using namespace Glyph3;

#include "cuda_runtime_api.h"
#include "cuda_d3d11_interop.h"
#include "kernel.h"

//--------------------------------------------------------------------------------
App AppInstance; // Provides an instance of the application
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
App::App() : m_numFrames(0)
{
	// Register for window based events here.
	m_pEventMgr->AddEventListener( WINDOW_RESIZE, this );
}
//--------------------------------------------------------------------------------
bool App::ConfigureEngineComponents()
{
	m_width = 700;
	m_height = 700;
	bool windowed = true;

	// Set the render window parameters and initialize the window
	m_pWindow = new Win32RenderWindow();
	m_pWindow->SetPosition( 25, 25 );
	m_pWindow->SetSize( m_width, m_height );
	m_pWindow->SetCaption( GetName() );
	m_pWindow->Initialize( this ); 


	// Create the renderer and initialize it for the desired device
	// type and feature level.

	m_pRenderer11 = new RendererDX11();

	if ( !m_pRenderer11->Initialize( D3D_DRIVER_TYPE_HARDWARE, D3D_FEATURE_LEVEL_11_0 ) )
	{
		Log::Get().Write( L"Could not create hardware device, trying to create the reference device..." );

		if ( !m_pRenderer11->Initialize( D3D_DRIVER_TYPE_REFERENCE, D3D_FEATURE_LEVEL_11_0 ) )
		{
			ShowWindow( m_pWindow->GetHandle(), SW_HIDE );
			MessageBox( m_pWindow->GetHandle(), L"Could not create a hardware or software Direct3D 11 device - the program will now abort!", L"Hieroglyph 3 Rendering", MB_ICONEXCLAMATION | MB_SYSTEMMODAL );
			RequestTermination();			
			return( false );
		}

		// If using the reference device, utilize a fixed time step for any animations.
		m_pTimer->SetFixedTimeStep( 1.0f / 10.0f );
	}


	// Create a swap chain for the window that we started out with.  This
	// demonstrates using a configuration object for fast and concise object
	// creation.

	SwapChainConfigDX11 Config;
	Config.SetWidth( m_pWindow->GetWidth() );
	Config.SetHeight( m_pWindow->GetHeight() );
	Config.SetOutputWindow( m_pWindow->GetHandle() );
	m_iSwapChain = m_pRenderer11->CreateSwapChain( &Config );
	m_pWindow->SetSwapChain( m_iSwapChain );

	// We'll keep a copy of the render target index to use in later examples.

	m_RenderTarget = m_pRenderer11->GetSwapChainResource( m_iSwapChain );


	// Next we create a depth buffer for use in the traditional rendering
	// pipeline.

	Texture2dConfigDX11 DepthConfig;
	DepthConfig.SetDepthBuffer( m_width, m_height);
	m_DepthTarget = m_pRenderer11->CreateTexture2D( &DepthConfig, 0 );


	// Bind the swap chain render target and the depth buffer for use in 
	// rendering.  

	m_pRenderer11->pImmPipeline->ClearRenderTargets();
	m_pRenderer11->pImmPipeline->OutputMergerStage.DesiredState.SetRenderTarget( 0, m_RenderTarget->m_iResourceRTV );
	m_pRenderer11->pImmPipeline->OutputMergerStage.DesiredState.SetDepthStencilTarget( m_DepthTarget->m_iResourceDSV );
	m_pRenderer11->pImmPipeline->ApplyRenderTargets();


	// Create a view port to use on the scene.  This basically selects the 
	// entire floating point area of the render target.

	D3D11_VIEWPORT viewport;
	viewport.Width = static_cast< float >( m_width );
	viewport.Height = static_cast< float >( m_height );
	viewport.MinDepth = 0.0f;
	viewport.MaxDepth = 1.0f;
	viewport.TopLeftX = 0;
	viewport.TopLeftY = 0;

	int ViewPort = m_pRenderer11->CreateViewPort( viewport );
	m_pRenderer11->pImmPipeline->RasterizerStage.DesiredState.SetViewportCount( 1 );
	m_pRenderer11->pImmPipeline->RasterizerStage.DesiredState.SetViewport( 0, ViewPort );

	m_scene = new Scene();
	m_camera = new FirstPersonCamera();
	m_camera->GetBody()->Position()[1] = 1;

	m_camera->SetProjectionParams(.1, 50, ((float)m_width) / m_height, GLYPH_PI / 4.f);
	m_camera->GetBody()->Rotation();
	m_scene->AddCamera( m_camera );

	return( true );
}
//--------------------------------------------------------------------------------
void App::ShutdownEngineComponents()
{
	cudaDeviceReset();
	if ( m_pRenderer11 )
	{
		m_pRenderer11->Shutdown();
		delete m_pRenderer11;
	}

	if ( m_pWindow )
	{
		m_pWindow->Shutdown();
		delete m_pWindow;
	}

	
}
//--------------------------------------------------------------------------------
void App::Initialize()
{
	// Here we load our desired texture, and create a shader resource view to 
	// use for input to the compute shader.  By specifying null for the 
	// configuration, the view is created from the default resource configuration.


	// Create the texture for output of the compute shader.
	Texture2dConfigDX11 FilteredConfig;
	FilteredConfig.SetColorBuffer( m_width, m_height ); 
	FilteredConfig.SetBindFlags(D3D11_BIND_SHADER_RESOURCE);
	FilteredConfig.SetFormat(DXGI_FORMAT_R32G32B32A32_FLOAT);
	//FilteredConfig.SetFormat(DXGI_FORMAT_R16G16B16A16_FLOAT);
	//FilteredConfig.SetFormat(DXGI_FORMAT_R8G8B8A8_UNORM);
	m_Output1 = m_pRenderer11->CreateTexture2D( &FilteredConfig, 0 );

	m_pTextureEffect = new RenderEffectDX11();
	m_pTextureEffect->SetVertexShader( m_pRenderer11->LoadShader( VERTEX_SHADER,
		std::wstring( L"LightwayRT/TextureVS.hlsl" ),
		std::wstring( L"VSMAIN" ),
		std::wstring( L"vs_5_0" ) ) );

	m_pTextureEffect->SetPixelShader( m_pRenderer11->LoadShader( PIXEL_SHADER,
		std::wstring( L"LightwayRT/TexturePS.hlsl" ),
		std::wstring( L"PSMAIN" ),
		std::wstring( L"ps_5_0" ) ) );

	// Create a full screen quad for rendering the texture to the backbuffer.

	m_pFullScreen = GeometryPtr( new GeometryDX11() );
	GeometryGeneratorDX11::GenerateFullScreenQuad( m_pFullScreen );

	m_pFullScreen->GenerateInputLayout( m_pTextureEffect->GetVertexShader() );
	m_pFullScreen->LoadToBuffers();

	// Specify the bindings for the resources.  These take as input a parameter
	// name and a resource proxy object created above.  This will connect these
	// resources with the appropriate shader variables at rendering time.  The
	// resource proxy object contains the needed 'ResourceView' instances.

	cudaGraphicsResource* cuda_tex;
	this->m_pRenderer11->PIXBeginEvent(L"cuda");
	CUDA_CHECK_RETURN(cudaGraphicsD3D11RegisterResource(&cuda_tex, 
		m_pRenderer11->GetResourceByIndex(m_Output1->m_iResource)->GetResource(), 
		cudaGraphicsRegisterFlagsSurfaceLoadStore));
	CUDA_CHECK_RETURN(cudaGraphicsResourceSetMapFlags(cuda_tex, cudaGraphicsMapFlagsWriteDiscard));
	this->m_pRenderer11->PIXEndEvent();
	kernel = new Kernel();
	kernel->setup(cuda_tex, m_width, m_height);
}
std::wstring ToString(const float value)
{
	std::wstringstream converter;
	std::wstring  wstr;

	converter << value;
	converter >> wstr;
	return wstr;
}

glm::vec3 toGlm(Vector3f v3)
{
	return glm::vec3(v3[0], v3[1], v3[2]);
}
//--------------------------------------------------------------------------------
void App::Update()
{
	// Update the timer to determine the elapsed time since last frame.  This can 
	// then used for animation during the frame.

	m_pTimer->Update();

	auto even = m_pTimer->FrameCount() % 2 == 0;

	this->m_pRenderer11->PIXBeginEvent(L"cuda");
	//if(m_numFrames < 100)//if(m_pTimer->Runtime() < .3)
	Stats stats;
	{
		kernel->execute(m_numFrames, m_width, m_height, true, &stats);
	}

	this->m_pRenderer11->PIXEndEvent();
	// Send an event to everyone that a new frame has started.  This will be used
	// in later examples for using the material system with render views.

	EventManager::Get()->ProcessEvent( new EvtFrameStart( *m_pTimer ) );

	// Perform the filtering with the compute shader.  The assumption in this case
	// is that the texture is 640x480 - if there is a different size then the 
	// dispatch call can be modified accordingly.
	m_camera->GetBody()->Update(0);

	//auto invView = m_camera->GetBody()->LocalMatrix().Inverse();
	//auto invProj = m_camera->ProjMatrix().Inverse();	
	//Vector3f pos = m_camera->GetBody()->Position();	
	//Vector4f pos4(pos, 1);

	m_pWindow->SetCaption(L"Frame " + ToString(m_numFrames) + L" MC/S: " + ToString(stats.combinations_per_second() / 1000000));

	m_numFrames++;
	if(m_camera->pollMoved())
	{
		m_numFrames = 0;
	}
	Vector4f width_height(m_width, m_height, 0, 0);
	auto timeData = Vector4f(m_pTimer->FrameCount() / 100.f, m_numFrames, 0, 0);
	m_pRenderer11->m_pParamMgr->SetVectorParameter(L"g_width_height", &width_height);

	m_pRenderer11->pImmPipeline->ClearPipelineResources();
	m_pRenderer11->pImmPipeline->ApplyPipelineResources();
	m_pRenderer11->m_pParamMgr->SetShaderResourceParameter( L"Input", m_Output1);

	// Render the texture to the backbuffer.

	m_pRenderer11->pImmPipeline->ClearBuffers( Vector4f( 0.0f, 0.0f, 0.0f, 0.0f ), 1.0f );
	m_pRenderer11->pImmPipeline->Draw( *m_pTextureEffect, m_pFullScreen, m_pRenderer11->m_pParamMgr );

	// Present the results to the window.

	m_pRenderer11->Present( m_pWindow->GetHandle(), m_pWindow->GetSwapChain() );

}
//--------------------------------------------------------------------------------
void App::Shutdown()
{
	m_pFullScreen = NULL;

	// Print the framerate out for the log before shutting down.

	std::wstringstream out;
	out << L"Max FPS: " << m_pTimer->MaxFramerate();
	Log::Get().Write( out.str() );
}
//--------------------------------------------------------------------------------
void App::TakeScreenShot()
{
	if ( m_bSaveScreenshot  )
	{
		m_bSaveScreenshot = false;
		m_pRenderer11->pImmPipeline->SaveTextureScreenShot( 0, GetName(), D3DX11_IFF_BMP );
	}
}
//--------------------------------------------------------------------------------
bool App::HandleEvent( IEvent* pEvent )
{
	eEVENT e = pEvent->GetEventType();
	if ( e == WINDOW_RESIZE )
	{
		EvtWindowResize* pResize = (EvtWindowResize*)pEvent;

		this->HandleWindowResize( pResize->GetWindowHandle(), pResize->NewWidth(), pResize->NewHeight() );

		return( true );
	}
	else if ( e == SYSTEM_KEYBOARD_KEYDOWN )
	{
		EvtKeyDown* pKeyDown = (EvtKeyDown*)pEvent;

		unsigned int key = pKeyDown->GetCharacterCode();
	}
	else if ( e == SYSTEM_KEYBOARD_KEYUP )
	{
		EvtKeyUp* pKeyUp = (EvtKeyUp*)pEvent;

		unsigned int key = pKeyUp->GetCharacterCode();
	}

	// Call the parent class's event handler if we haven't handled the event.

	return( Application::HandleEvent( pEvent ) );
}

//--------------------------------------------------------------------------------
void App::HandleWindowResize( HWND handle, UINT width, UINT height )
{
	return;

	// TODO: are these local width and height members needed???
	if ( width < 1 ) width = 1;
	if ( height < 1 ) height = 1;

	m_width = width;
	m_height = height;

	// Resize our rendering window state if the handle matches
	if ( ( m_pWindow != 0 ) && ( m_pWindow->GetHandle() == handle ) ) {
		m_pWindow->ResizeWindow( width, height );
		m_pRenderer11->ResizeSwapChain( m_pWindow->GetSwapChain(), width, height );

	}

	// Update the projection matrix of our camera
	if ( m_camera != 0 ) {
		m_camera->SetAspectRatio( static_cast<float>(width) / static_cast<float>(height) );
	}


	Texture2dConfigDX11 FilteredConfig;
	FilteredConfig.SetColorBuffer( width, height); 
	FilteredConfig.SetBindFlags( D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE );

	m_Output1 = m_pRenderer11->CreateTexture2D( &FilteredConfig, 0 );

}
//--------------------------------------------------------------------------------
std::wstring App::GetName( )
{
	return( std::wstring( L"BasicComputeShader" ) );
}
//--------------------------------------------------------------------------------