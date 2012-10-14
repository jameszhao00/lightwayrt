//--------------------------------------------------------------------------------
// This file is a portion of the Hieroglyph 3 Rendering Engine.  It is distributed
// under the MIT License, available in the root of this distribution and 
// at the following URL:
//
// http://www.opensource.org/licenses/mit-license.php
//
// Copyright (c) 2003-2010 Jason Zink 
//--------------------------------------------------------------------------------
#include "Application.h"

#include "Win32RenderWindow.h"
#include "RendererDX11.h"
#include "FirstPersonCamera.h"


using namespace Glyph3;

class App : public Application
{

public:
	App();
	
public:
	virtual void Initialize();
	virtual void Update();
	virtual void Shutdown();

	virtual bool ConfigureEngineComponents();
	virtual void ShutdownEngineComponents();

	virtual void TakeScreenShot();

	virtual bool HandleEvent( IEvent* pEvent );
	virtual std::wstring GetName( );

	void HandleWindowResize( HWND handle, UINT width, UINT height );

protected:
	RendererDX11*			m_pRenderer11;
	Win32RenderWindow*		m_pWindow;
	
	int						m_iSwapChain;
	ResourcePtr				m_RenderTarget;
	ResourcePtr				m_DepthTarget;

	// Output Texture IDs
	ResourcePtr				m_Output1;
	ResourcePtr				m_Output2;

	// Geometry for full screen pass
	GeometryPtr				m_pFullScreen;

	// RenderEffects for running the compute shader and rendering
	// the resulting texture to the backbuffer.
	RenderEffectDX11*		m_pFilterEffect;
	RenderEffectDX11*		m_pTextureEffect;

	Scene* m_scene;
	FirstPersonCamera* m_camera;
	UINT m_width;
	UINT m_height;
	int m_numFrames;
};
