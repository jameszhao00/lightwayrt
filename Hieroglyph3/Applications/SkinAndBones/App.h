//--------------------------------------------------------------------------------
// This file is a portion of the Hieroglyph 3 Rendering Engine.  It is distributed
// under the MIT License, available in the root of this distribution and 
// at the following URL:
//
// http://www.opensource.org/licenses/mit-license.php
//
// Copyright (c) 2003-2010 Jason Zink 
//--------------------------------------------------------------------------------
#include "RenderApplication.h"

#include "GeometryDX11.h"
#include "MaterialDX11.h"
#include "Camera.h"
#include "Scene.h"
#include "SkinnedActor.h"
#include "VectorParameterDX11.h"

using namespace Glyph3;

class App : public RenderApplication
{

public:
	App();
	
public:
	virtual bool ConfigureEngineComponents();
	virtual void ShutdownEngineComponents();

	virtual void Initialize();
	virtual void Update();
	virtual void Shutdown();

	virtual bool HandleEvent( IEvent* pEvent );
	virtual std::wstring GetName( );

protected:

	Node3D*					m_pNode;
	
	SkinnedActor*			m_pSkinnedActor;
	SkinnedActor*			m_pDisplacedActor;
	Actor*					m_pStaticActor;
	
	VectorParameterDX11*	m_pLightColor;
	VectorParameterDX11*	m_pLightPosition;
};
