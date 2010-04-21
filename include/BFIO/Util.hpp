/*
   Copyright 2010 Jack Poulson

   This file is part of ButterflyFIO.

   This program is free software: you can redistribute it and/or modify it under
   the terms of the GNU Lesser General Public License as published by the
   Free Software Foundation; either version 3 of the License, or 
   (at your option) any later version.

   This program is distributed in the hope that it will be useful, but 
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with this program. If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef BFIO_UTIL_HPP
#define BFIO_UTIL_HPP 1

// C standard headers
#include <cmath>
#include <cstdlib>

// C++ standard headers
#include <complex>
#include <iostream>
#include <vector>

// Additional library headers
#include "mpi.h"

#include "BFIO/Data.hpp"
#include "BFIO/MPI.hpp"

namespace BFIO
{
    inline bool
    IsPowerOfTwo( unsigned N )
    { return N && !(N & (N-1)); }

    inline unsigned
    Log2( unsigned N )
    { 
        unsigned j = 0;
        while( (N>>j) > 1 )
            ++j;
        return j;
    }

    // This is a modification of Sean Eron Anderson's binary search algorithm
    // for counting the trailing zeros of a 32-bit integer. It was found at
    //     http://graphics.stanford.edu/~seander/bithacks.html
    // and is in the public domain.
    //
    // The main difference is that I switched the algorithm to count ones and 
    // ignore the case where N=2^32-1 rather than N=0
    inline unsigned
    NumberOfTrailingOnes( unsigned N )
    {
        unsigned int c;
        if( (N & 0x1)==0 )
        {
            c = 0;
        }
        else
        {
            c = 1; 
            if( (N & 0xffff)==0xffff )
            {
                N >>= 16; 
                c += 16;
            }
            if( (N & 0xff) == 0xff )
            {
                N >>= 8;
                c += 8;
            }
            if( (N & 0xf) == 0xf )
            {
                N >>= 4;
                c += 4;
            }
            if( (N & 0x3) == 0x3 )
            {
                N >>= 2;
                c += 2;
            }
            c -= !(N & 0x1);
        }
        return c;
    }
}

#endif /* BFIO_UTIL_HPP */

