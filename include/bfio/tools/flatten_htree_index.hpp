/*
   ButterflyFIO: a distributed-memory fast algorithm for applying FIOs.
   Copyright (C) 2010 Jack Poulson <jack.poulson@gmail.com>
 
   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
 
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
 
   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
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

#endif // BFIO_FLATTEN_HTREE_INDEX_HPP

