/*
   Copyright (c) 2010, Jack Poulson
   All rights reserved.

   This file is part of ButterflyFIO.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are met:

    - Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    - Neither the name of the owner nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
   ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
   LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
   CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
   SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
   INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
   CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
   ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
   POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once
#ifndef BFIO_FLATTEN_HTREE_INDEX_HPP
#define BFIO_FLATTEN_HTREE_INDEX_HPP 1

#include "bfio/structures/data.hpp"
#include "bfio/tools/twiddle.hpp"

namespace bfio {

template<unsigned d>
unsigned
FlattenHTreeIndex
( const Array<unsigned,d>& x )
{
    // We will accumulate the index into this variable
    unsigned index = 0;

    // Compute the maximum recursion height reached by searching for the
    // maximum log2 of the coordinates
    unsigned maxLog2 = 0;
    for( unsigned j=0; j<d; ++j )
        maxLog2 = std::max( Log2(x[j]), maxLog2 );

    // Now unroll the coordinates into the index
    for( unsigned i=0; i<=maxLog2; ++i )
        for( unsigned j=0; j<d; ++j )
            index |= ((x[j]>>i)&1)<<(i*d+j);

    return index;
}

template<unsigned d>
unsigned
FlattenCHTreeIndex
( const Array<unsigned,d>& x, const Array<unsigned,d>& log2BoxesPerDim )
{
    // We will accumulate the index into this variable
    unsigned index = 0;

    // Compute the maximum recursion height reached by searching for the
    // maximum log2 of the coordinates
    unsigned maxLog2 = 0;
    for( unsigned j=0; j<d; ++j )
        maxLog2 = std::max( Log2(x[j]), maxLog2 );

    // Now unroll the coordinates into the index
    for( unsigned i=0; i<=maxLog2; ++i )
    {
        // Sum the total number of levels i is 'above' the maximum for 
        // each dimension
        unsigned log2BoxesUpToLevel = 0;
        for( unsigned j=0; j<d; ++j )
            log2BoxesUpToLevel += std::min( i, log2BoxesPerDim[j] );
        
        // Now unroll for each dimension
        unsigned unfilledBefore = 0;
        for( unsigned j=0; j<d; ++j )
        {
            index |= ((x[j]>>i)&1)<<(log2BoxesUpToLevel+unfilledBefore);
            if( log2BoxesPerDim[j] > i )
                ++unfilledBefore;
        }
    }

    return index;
}

} // bfio

#endif /* BFIO_FLATTEN_HTREE_INDEX_HPP */

