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
#ifndef BFIO_UTIL_H
#define BFIO_UTIL_H 1

// C standard headers
#include <cmath>
#include <cstdlib>

// C++ standard headers
#include <complex>
#include <iostream>
#include <vector>

// Additional library headers
#include "mpi.h"

#include "BFIOMPI.h"

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

    inline unsigned
    TwiddleBit
    ( unsigned N, unsigned l )
    { return N ^ (1<<l); }
}

#endif /* BFIO_UTIL_H */

