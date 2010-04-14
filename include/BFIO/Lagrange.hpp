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
#ifndef BFIO_LAGRANGE_HPP
#define BFIO_LAGRANGE_HPP 1

#include "BFIO/Data.hpp"

namespace BFIO
{
    template<typename R,unsigned d,unsigned q>
    inline R
    Lagrange
    ( const unsigned t, const Array<R,d>& z, const Array<R,q>& chebyNodes )
    {
        R product = static_cast<R>(1);
        unsigned qToThej = 1;
        for( unsigned j=0; j<d; ++j )
        {
            const unsigned i = (t/qToThej) % q;
            for( unsigned k=0; k<q; ++k )
            {
                if( i != k )
                {
                    product *= (z[j]-chebyNodes[k]) / 
                               (chebyNodes[i]-chebyNodes[k]);
                }
            }
            qToThej *= q;
        }
        return product;
    }
}

#endif /* BFIO_LAGRANGE_HPP */

