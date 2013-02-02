/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of ButterflyFIO and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef BFIO_TOOLS_FLATTEN_CONSTRAINED_HTREE_INDEX_HPP
#define BFIO_TOOLS_FLATTEN_CONSTRAINED_HTREE_INDEX_HPP

#include <array>
#include <cstddef>

#include "bfio/tools/twiddle.hpp"

namespace bfio {

using std::array;
using std::size_t;

template<size_t d>
size_t
FlattenConstrainedHTreeIndex
( const array<size_t,d>& x, 
  const array<size_t,d>& log2BoxesPerDim )
{
    // We will accumulate the index into this variable
    size_t index = 0;

    // Compute the maximum recursion height reached by searching for the
    // maximum log2 of the coordinates
    size_t maxLog2 = 0;
    for( size_t j=0; j<d; ++j )
        maxLog2 = std::max( Log2(x[j]), maxLog2 );

    // Now unroll the coordinates into the index
    for( size_t i=0; i<=maxLog2; ++i )
    {
        // Sum the total number of levels i is 'above' the maximum for 
        // each dimension
        size_t log2BoxesUpToLevel = 0;
        for( size_t j=0; j<d; ++j )
            log2BoxesUpToLevel += std::min( i, log2BoxesPerDim[j] );
        
        // Now unroll for each dimension
        size_t unfilledBefore = 0;
        for( size_t j=0; j<d; ++j )
        {
            index |= ((x[j]>>i)&1)<<(log2BoxesUpToLevel+unfilledBefore);
            if( log2BoxesPerDim[j] > i )
                ++unfilledBefore;
        }
    }

    return index;
}

} // bfio

#endif // ifndef BFIO_TOOLS_FLATTEN_CONSTRAINED_HTREE_INDEX_HPP
