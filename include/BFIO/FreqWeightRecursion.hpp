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
      const vector< Array<R,d> >& chebyGrid,
      const Array<R,d>& x0A,
      const Array<R,d>& p0B,
      const R wB,
      const unsigned parentOffset,
      const vector< vector< complex<R> > >& oldWeights,
            vector< complex<R> >& weights               )
    {
        typedef complex<R> C;

        for( unsigned t=0; t<Power<q,d>::value; ++t )
        {
            // Compute the unscaled weight
            weights[t] = 0;
            for( unsigned c=0; c<(1u<<d); ++c )
            {
                const unsigned parentKey = parentOffset + c;
                for( unsigned tp=0; tp<Power<q,d>::value; ++tp )        
                {
                    // Map p_t'(Bc) to the reference domain of B
                    Array<R,d> ptpBcRefB;
                    for( unsigned j=0; j<d; ++j )
                    {
                        ptpBcRefB[j] = ( (c>>j) & 1 ? 
                                         (2*chebyGrid[tp][j]+1)/4 :
                                         (2*chebyGrid[tp][j]-1)/4  );
                    }

                    // Scale and translate p_t'(Bc) on ref of B to p_t'
                    Array<R,d> ptp;
                    for( unsigned j=0; j<d; ++j )
                        ptp[j] = p0B[j] + wB*ptpBcRefB[j];

                    const R alpha = TwoPi*N*Psi::Eval(x0A,ptp);
                    weights[t] += Lagrange<R,d,q>( t, ptpBcRefB, chebyNodes ) *
                                  C( cos(alpha), sin(alpha) ) * 
                                  oldWeights[parentKey][tp];
                }
            }

            // Scale the weight
            Array<R,d> ptB;
            for( unsigned j=0; j<d; ++j )
                ptB[j] = p0B[j] + wB*chebyGrid[t][j];
            const R alpha = -TwoPi*N*Psi::Eval(x0A,ptB);
            weights[t] *= C( cos(alpha), sin(alpha) );
        }
    }
}

#endif /* BFIO_FREQ_WEIGHT_RECURSION_HPP */

