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
#ifndef BFIO_FREQ_TO_SPATIAL_SPATIAL_WEIGHT_RECURSION_HPP
#define BFIO_FREQ_TO_SPATIAL_SPATIAL_WEIGHT_RECURSION_HPP 1

#include "bfio/tools/blas.hpp"

namespace bfio {
namespace freq_to_spatial {

template<typename R,unsigned d,unsigned q>
void
SpatialWeightRecursion
( const AmplitudeFunctor<R,d>& Amp,
  const PhaseFunctor<R,d>& Phi,
  const unsigned log2NumMergingProcesses,
  const unsigned myTeamRank,
  const unsigned N, 
  const Context<R,d,q>& context,
  const unsigned ARelativeToAp,
  const Array<R,d>& x0A,
  const Array<R,d>& x0Ap,
  const Array<R,d>& p0B,
  const Array<R,d>& wA,
  const Array<R,d>& wB,
  const unsigned parentOffset,
  const WeightGridList<R,d,q>& oldWeightGridList,
        WeightGrid<R,d,q>& weightGrid
)
{
    typedef std::complex<R> C;
    const unsigned q_to_d = Pow<q,d>::val;
    const unsigned q_to_2d = Pow<q,2*d>::val;

    for( unsigned t=0; t<q_to_d; ++t )
        weightGrid[t] = 0;

    // We seek performance by isolating the Lagrangian interpolation as 
    // a matrix-vector multiplication.
    //
    // To do so, the spatial weight recursion is broken into three updates:
    // For each child c:
    //  1) scale the old weights with the appropriate exponentials
    //  2) multiply the lagrangian matrix against the scaled weights
    //  3) scale and accumulate the result of the lagrangian mat-vec
    std::vector< Array<R,d> > pPoint( 1 );
    std::vector< Array<R,d> > xPoints( q_to_d );
    std::vector<R> phiResults( q_to_d );
    std::vector<C> imagExpResults( q_to_d );
    const std::vector<R>& spatialMaps = context.GetSpatialMaps();
    const std::vector< Array<R,d> >& chebyshevGrid = context.GetChebyshevGrid();
    for( unsigned cLocal=0; cLocal<(1u<<(d-log2NumMergingProcesses)); ++cLocal )
    {
        // Step 1: scale the old weights
        WeightGrid<R,d,q> scaledWeightGrid;
        const unsigned c = (cLocal<<log2NumMergingProcesses) + myTeamRank;
        const unsigned key = parentOffset + cLocal;
        for( unsigned j=0; j<d; ++j )
            pPoint[0][j] = p0B[j] + ( (c>>j)&1 ? wB[j]/4 : -wB[j]/4 );
        for( unsigned tPrime=0; tPrime<q_to_d; ++tPrime )
            for( unsigned j=0; j<d; ++j )
                xPoints[tPrime][j] = 
                    x0Ap[j] + (2*wA[j])*chebyshevGrid[tPrime][j];
        Phi.BatchEvaluate( xPoints, pPoint, phiResults );
        for( unsigned tPrime=0; tPrime<q_to_d; ++tPrime )
            phiResults[tPrime] *= TwoPi;
        ImagExpBatch<R>( phiResults, imagExpResults );
        for( unsigned tPrime=0; tPrime<q_to_d; ++tPrime )
            scaledWeightGrid[tPrime] = 
                oldWeightGridList[key][tPrime]/imagExpResults[tPrime];

        // Step 2: perform the matrix-vector multiply
        WeightGrid<R,d,q> expandedWeightGrid;
        RealMatrixComplexVec
        ( q_to_d, q_to_d, 
          (R)1, &spatialMaps[ARelativeToAp*q_to_2d], q_to_d, 
                &scaledWeightGrid[0],
          (R)0, &expandedWeightGrid[0] );

        // Step 3: scale the result
        for( unsigned t=0; t<Pow<q,d>::val; ++t )
            for( unsigned j=0; j<d; ++j )
                xPoints[t][j] = x0A[j] + wA[j]*chebyshevGrid[t][j];
        Phi.BatchEvaluate( xPoints, pPoint, phiResults );
        for( unsigned t=0; t<q_to_d; ++t )
            phiResults[t] *= TwoPi;
        ImagExpBatch<R>( phiResults, imagExpResults );
        for( unsigned t=0; t<q_to_d; ++t )
            weightGrid[t] += imagExpResults[t]*expandedWeightGrid[t];
    }
}

} // freq_to_spatial
} // bfio

#endif // BFIO_FREQ_TO_SPATIAL_SPATIAL_WEIGHT_RECURSION_HPP

