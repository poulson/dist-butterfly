/*
   ButterflyFIO: a distributed-memory fast algorithm for applying FIOs.
   Copyright (C) 2010-2011 Jack Poulson <jack.poulson@gmail.com>
 
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
#ifndef BFIO_TOOLS_FLATTEN_HTREE_INDEX_HPP
#define BFIO_TOOLS_FLATTEN_HTREE_INDEX_HPP 1

#include <cstddef>
#include "bfio/structures/array.hpp"

#include "bfio/tools/twiddle.hpp"

namespace bfio {

template<std::size_t d>
std::size_t
FlattenHTreeIndex
( const Array<std::size_t,d>& x )
{
    // We will accumulate the index into this variable
    std::size_t index = 0;

    // Compute the maximum recursion height reached by searching for the
    // maximum log2 of the coordinates
    std::size_t maxLog2 = 0;
    for( std::size_t j=0; j<d; ++j )
        maxLog2 = std::max( Log2(x[j]), maxLog2 );

    // Now unroll the coordinates into the index
    for( std::size_t i=0; i<=maxLog2; ++i )
        for( std::size_t j=0; j<d; ++j )
            index |= ((x[j]>>i)&1)<<(i*d+j);

    return index;
}

} // bfio

#endif // BFIO_TOOLS_FLATTEN_HTREE_INDEX_HPP

