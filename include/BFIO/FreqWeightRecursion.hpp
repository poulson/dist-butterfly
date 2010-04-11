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
#ifndef BFIO_FREQ_WEIGHT_RECURSION_HPP
#define BFIO_FREQ_WEIGHT_RECURSION_HPP 1

#include "BFIO/Lagrange.hpp"

namespace BFIO
{
    using namespace std;

    template<typename Psi,typename R,unsigned d,unsigned q>
    inline void
    FreqWeightRecursion
    ( const unsigned N, 
      const Array<R,q>& chebyNodes,
      const Array< Array<R,d>,Power<q,d>::value >& chebyGrid,
      const Array<R,d>& x0,
      const Array<R,d>& p0,
      const R wB,
      const unsigned parentOffset,
      const vector< Array<complex<R>,Power<q,d>::value> >& oldWeights,
            Array<complex<R>,Power<q,d>::value> weights               )
    {
        typedef complex<R> C;

        for( unsigned t=0; t<Power<q,d>::value-1; ++t )
        {
            // Compute the unscaled weight
            weights[t] = 0;
            for( unsigned c=0; c<(1u<<d); ++c )
            {
                const unsigned parentKey = parentOffset + c;
                for( unsigned tp=0; tp<Power<q,d>::value-1; ++tp )        
                {
                    // Map the tp'th reference point to its parent reference
                    Array<R,d> pRef;
                    for( unsigned j=0; j<d; ++j )
                    {
                        pRef[j] = ( (c>>j) & 1 ? 
                                    (2*chebyGrid[tp][j]+1)/4 :
                                    (2*chebyGrid[tp][j]-1)/4  );
                    }

                    // Scale and translate the the physical position
                    Array<R,d> p;
                    for( unsigned j=0; j<d; ++j )
                        p[j] = p0[j] + wB*pRef[j];

                    const R alpha = TwoPi*N*Psi::Eval(x0,p);
                    weights[t] += Lagrange<R,d,q>( t, pRef, chebyNodes ) *
                                  C( cos(alpha), sin(alpha) ) * 
                                  oldWeights[parentKey][tp];
                }
            }

            // Scale the weight
            Array<R,d> p;
            for( unsigned j=0; j<d; ++j )
                p[j] = p0[j] + wB*chebyGrid[t][j];
            const R alpha = -TwoPi*N*Psi::Eval(x0,p);
            weights[t] *= C( cos(alpha), sin(alpha) );
        }
    }
}

#endif /* BFIO_FREQ_WEIGHT_RECURSION_HPP */

