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
#include "VectorParameterWriterDX11.h"
//--------------------------------------------------------------------------------
using namespace Glyph3;
//--------------------------------------------------------------------------------
VectorParameterWriterDX11::VectorParameterWriterDX11()
{
	m_Value.MakeZero();
}
//--------------------------------------------------------------------------------
VectorParameterWriterDX11::~VectorParameterWriterDX11()
{
}
//--------------------------------------------------------------------------------
void VectorParameterWriterDX11::SetRenderParameterRef( VectorParameterDX11* pParam )
{
	m_pParameter = pParam;
}
//--------------------------------------------------------------------------------
void VectorParameterWriterDX11::WriteParameter( IParameterManager* pParamMgr )
{
	pParamMgr->SetVectorParameter( m_pParameter, &m_Value );
}
//--------------------------------------------------------------------------------
void VectorParameterWriterDX11::SetValue( Vector4f& Value )
{
	m_Value = Value;
}
//--------------------------------------------------------------------------------