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
// BufferConfigDX11
//
//--------------------------------------------------------------------------------
#include "RendererDX11.h"
//--------------------------------------------------------------------------------
#ifndef BufferConfigDX11_h
#define BufferConfigDX11_h
//--------------------------------------------------------------------------------
namespace Glyph3
{
	class BufferConfigDX11
	{
	public:
		BufferConfigDX11();
		virtual ~BufferConfigDX11();

		void SetDefaults();

		void SetDefaultConstantBuffer( UINT size, bool dynamic );
		void SetDefaultVertexBuffer( UINT size, bool dynamic );
		void SetDefaultIndexBuffer( UINT size, bool dynamic );
		void SetDefaultStructuredBuffer( UINT size, UINT structsize );
		void SetDefaultByteAddressBuffer( UINT size );
		void SetDefaultIndirectArgsBuffer( UINT size );
		void SetDefaultStagingBuffer( UINT size );

		void SetByteWidth( UINT state );
		void SetUsage( D3D11_USAGE state );
	    void SetBindFlags( UINT state );
	    void SetCPUAccessFlags( UINT state );
	    void SetMiscFlags( UINT state );	
	    void SetStructureByteStride( UINT state );

		D3D11_BUFFER_DESC GetBufferDesc();

	protected:
		D3D11_BUFFER_DESC 		m_State;

		friend RendererDX11;
	};
};
//--------------------------------------------------------------------------------
#endif // BufferConfigDX11_h
//--------------------------------------------------------------------------------

