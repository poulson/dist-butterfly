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
#ifndef BFIO_FREQ_TO_SPATIAL_SPATIAL_WEIGHT_RECURSION_HPP
#define BFIO_FREQ_TO_SPATIAL_SPATIAL_WEIGHT_RECURSION_HPP 1

#include "bfio/tools/blas.hpp"
#include "bfio/tools/lagrange.hpp"

namespace bfio {
namespace freq_to_spatial {

template<typename R,unsigned d,unsigned q>
void
SpatialWeightRecursion
( const PhaseFunctor<R,d>& Phi,
  const unsigned log2Procs,
  const unsigned myTeamRank,
  const unsigned N, 
  const std::vector< Array<R,d> >& chebyGrid,
  const unsigned ARelativeToAp,
  const Array<R,d>& x0A,
  const Array<R,d>& x0Ap,
  const Array<R,d>& p0B,
  const R wA,
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
    static std::vector<R> LSpatial( q_to_2d << d );

    if( !initialized )
    {
        for( unsigned p=0; p<(1u<<d); ++p )
        {
            for( unsigned tPrime=0; tPrime<q_to_d; ++tPrime )
            {
                for( unsigned t=0; t<q_to_d; ++t )
                {
                    // Map x_t(A) to the reference domain of its parent
                    Array<R,d> xtARefAp;
                    for( unsigned j=0; j<d; ++j )
                    {
                        xtARefAp[j] = 
                            ( (p>>j)&1 ? (2*chebyGrid[t][j]+1)/4 :
                                         (2*chebyGrid[t][j]-1)/4  );
                    }

                    LSpatial[p*q_to_2d + t+tPrime*q_to_d] = 
                        Lagrange<R,d,q>( tPrime, xtARefAp );
                }
            }
        }
        initialized = true;
    }

    for( unsigned t=0; t<q_to_d; ++t )
        weightSet[t] = 0;

    // We seek performance by isolating the Lagrangian interpolation as 
    // a matrix-vector multiplication.
    //
    // To do so, the spatial weight recursion is broken into three updates:
    // For each child c:
    //  1) scale the old weights with the appropriate exponentials
    //  2) multiply the lagrangian matrix against the scaled weights
    //  3) scale and accumulate the result of the lagrangian mat-vec
    for( unsigned cLocal=0; cLocal<(1u<<(d-log2Procs)); ++cLocal )
    {
        // Step 1: scale the old weights
        WeightSet<R,d,q> scaledWeightSet;
        const unsigned c = (cLocal<<log2Procs) + myTeamRank;
        const unsigned key = parentOffset + cLocal;
        Array<R,d> p0Bc;
        for( unsigned j=0; j<d; ++j )
            p0Bc[j] = p0B[j] + ( (c>>j)&1 ? wB/4 : -wB/4 );
        for( unsigned tPrime=0; tPrime<q_to_d; ++tPrime )
        {
            Array<R,d> xtPrimeAp;
            for( unsigned j=0; j<d; ++j )
                xtPrimeAp[j] = x0Ap[j] + (2*wA)*chebyGrid[tPrime][j];
            const R alpha = -TwoPi*Phi( xtPrimeAp, p0Bc );
            scaledWeightSet[tPrime] = 
                C(cos(alpha),sin(alpha))*oldWeightSetList[key][tPrime];
        }

        // Step 2: perform the matrix-vector multiply
        WeightSet<R,d,q> expandedWeightSet;
        RealMatrixComplexVec
        ( q_to_d, q_to_d, 
          (R)1, &LSpatial[ARelativeToAp*q_to_2d], q_to_d, 
                &scaledWeightSet[0],
          (R)0, &expandedWeightSet[0] );

        // Step 3: scale the result
        for( unsigned t=0; t<Pow<q,d>::val; ++t )
        {
            Array<R,d> xtA;
            for( unsigned j=0; j<d; ++j )
                xtA[j] = x0A[j] + wA*chebyGrid[t][j];
            const R alpha = TwoPi*Phi( xtA, p0Bc );
            weightSet[t] += C(cos(alpha),sin(alpha))*expandedWeightSet[t];
        }
    }
}

} // freq_to_spatial
} // bfio

#endif /* BFIO_FREQ_TO_SPATIAL_SPATIAL_WEIGHT_RECURSION_HPP */

