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
#ifndef BFIO_FREQ_TO_SPATIAL_FREQ_WEIGHT_RECURSION_HPP
#define BFIO_FREQ_TO_SPATIAL_FREQ_WEIGHT_RECURSION_HPP 1

#include "bfio/tools/lagrange.hpp"

namespace bfio {
namespace freq_to_spatial {

template<typename R,unsigned d,unsigned q>
void
FreqWeightRecursion
( const PhaseFunctor<R,d>& Phi,
  const unsigned log2Procs,
  const unsigned myTeamRank,
  const unsigned N, 
  const std::vector< Array<R,d> >& chebyGrid,
  const Array<R,d>& x0A,
  const Array<R,d>& p0B,
  const R wB,
  const unsigned parentOffset,
  const WeightSetList<R,d,q>& oldWeightSetList,
        WeightSet<R,d,q>& weightSet
)
{
    typedef std::complex<R> C;
    const unsigned q_to_d = Pow<q,d>::val;
    const unsigned q_to_2d = Pow<q,2*d>::val;

    static bool initialized = false;
    static std::vector<R> pRefB( (q_to_d << d)*d );
    static std::vector<R> LFreq( q_to_2d << d );

    if( !initialized )
    {
        for( unsigned c=0; c<(1u<<d); ++c )
        {
            for( unsigned t=0; t<q_to_d; ++t )
            {
                for( unsigned tPrime=0; tPrime<q_to_d; ++tPrime )
                {
                    // Map p_t'(Bc) to the reference domain of B and 
                    // store the Lagrangian evaluation
                    Array<R,d> ptPrimeBcRefB;
                    for( unsigned j=0; j<d; ++j )
                    {
                        pRefB[c*q_to_d*d+tPrime*d+j] = ptPrimeBcRefB[j] = 
                            ( (c>>j)&1 ? (2*chebyGrid[tPrime][j]+1)/4 :
                                         (2*chebyGrid[tPrime][j]-1)/4  );
                    }
                    LFreq[c*q_to_2d+t+tPrime*q_to_d] = 
                        Lagrange<R,d,q>( t, ptPrimeBcRefB );
                }
            }
        }
        initialized = true;
    }

    // We seek performance by isolating the Lagrangian interpolation as
    // a matrix-vector multiplication
    //
    // To do so, the frequency weight recursion is broken into 3 steps.
    // 
    // For each child:
    //  1) scale the old weights with the appropriate exponentials
    //  2) accumulate the lagrangian matrix against the scaled weights
    // Finally:
    // 3) scale the accumulated weights

    for( unsigned t=0; t<q_to_d; ++t )
        weightSet[t] = 0;

    for( unsigned cLocal=0; cLocal<(1u<<(d-log2Procs)); ++cLocal )
    {
        // Step 1
        const unsigned c = (cLocal<<log2Procs) + myTeamRank;
        const unsigned key = parentOffset + cLocal;

        WeightSet<R,d,q> scaledWeightSet;
        for( unsigned tPrime=0; tPrime<q_to_d; ++tPrime )
        {
            Array<R,d> ptPrime;
            for( unsigned j=0; j<d; ++j )
                ptPrime[j] = p0B[j] + wB*pRefB[c*q_to_d*d+tPrime*d+j];

            const R alpha = TwoPi*Phi( x0A, ptPrime );
            scaledWeightSet[tPrime] = 
                C( cos(alpha), sin(alpha) )*oldWeightSetList[key][tPrime];
        }
        
        // Step 2
        RealMatrixComplexVec
        ( q_to_d, q_to_d, (R)1, &LFreq[c*q_to_2d], q_to_d, 
          &scaledWeightSet[0], (R)1, &weightSet[0] );
    }

    // Step 3
    for( unsigned t=0; t<q_to_d; ++t )
    {
        Array<R,d> ptB;
        for( unsigned j=0; j<d; ++j )
            ptB[j] = p0B[j] + wB*chebyGrid[t][j];

        const R alpha = -TwoPi*Phi( x0A, ptB );
        weightSet[t] *= C( cos(alpha), sin(alpha) );
    }
}

} // freq_to_spatial
} // bfio

#endif /* BFIO_FREQ_TO_SPATIAL_FREQ_WEIGHT_RECURSION_HPP */

