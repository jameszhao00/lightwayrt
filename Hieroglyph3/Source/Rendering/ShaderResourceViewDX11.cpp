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
#include "ShaderResourceViewDX11.h"
//--------------------------------------------------------------------------------
using namespace Glyph3;
//--------------------------------------------------------------------------------
ShaderResourceViewDX11::ShaderResourceViewDX11( ID3D11ShaderResourceView* pView )
{
	m_pShaderResourceView = pView;
}
//--------------------------------------------------------------------------------
ShaderResourceViewDX11::~ShaderResourceViewDX11()
{
	SAFE_RELEASE( m_pShaderResourceView );
}
//--------------------------------------------------------------------------------