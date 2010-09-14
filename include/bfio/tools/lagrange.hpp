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
#ifndef BFIO_TOOLS_LAGRANGE_HPP
#define BFIO_TOOLS_LAGRANGE_HPP 1

#include <vector>
#include "bfio/structures/data.hpp"

namespace bfio {

template<typename R,unsigned d,unsigned q>
R
Lagrange
( const unsigned t, const Array<R,d>& z )
{
    const unsigned q_to_d = Pow<q,d>::val;

    static bool initialized = false;
    static Array<R,q> chebyNodes;
    static std::vector< Array<unsigned,d> > chebyIndex( q_to_d );

    if( !initialized )
    {
        for( unsigned i=0; i<q; ++i )
            chebyNodes[i] = 0.5*cos(i*Pi/(q-1));
        for( unsigned tp=0; tp<q_to_d; ++tp )
        {
            unsigned qToThej = 1;
            for( unsigned j=0; j<d; ++j )
            {
                unsigned i = (tp/qToThej) % q;
                chebyIndex[tp][j] = i;
                qToThej *= q;
            }
        }
        initialized = true;
    }

    R product = static_cast<R>(1);
    for( unsigned j=0; j<d; ++j )
    {
        unsigned i = chebyIndex[t][j];
        for( unsigned k=0; k<q; ++k )
        {
            if( i != k )
            {
                product *= 
                    (z[j]-chebyNodes[k]) / (chebyNodes[i]-chebyNodes[k]);
            }
        }
    }
    return product;
}

} // bfio

#endif // BFIO_TOOLS_LAGRANGE_HPP

