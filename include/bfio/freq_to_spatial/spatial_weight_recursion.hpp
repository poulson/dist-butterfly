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
#ifndef BFIO_FREQ_TO_SPATIAL_SPATIAL_WEIGHT_RECURSION_HPP
#define BFIO_FREQ_TO_SPATIAL_SPATIAL_WEIGHT_RECURSION_HPP 1

#include "bfio/tools/blas.hpp"

namespace bfio {
namespace freq_to_spatial {

template<typename R,std::size_t d,std::size_t q>
void
SpatialWeightRecursion
( const AmplitudeFunctor<R,d>& Amp,
  const PhaseFunctor<R,d>& Phi,
  const std::size_t log2NumMergingProcesses,
  const std::size_t myTeamRank,
  const std::size_t N, 
  const Context<R,d,q>& context,
  const std::size_t ARelativeToAp,
  const std::tr1::array<R,d>& x0A,
  const std::tr1::array<R,d>& x0Ap,
  const std::tr1::array<R,d>& p0B,
  const std::tr1::array<R,d>& wA,
  const std::tr1::array<R,d>& wB,
  const std::size_t parentOffset,
  const WeightGridList<R,d,q>& oldWeightGridList,
        WeightGrid<R,d,q>& weightGrid
)
{
    typedef std::complex<R> C;
    const std::size_t q_to_d = Pow<q,d>::val;
    const std::size_t q_to_2d = Pow<q,2*d>::val;

    for( std::size_t t=0; t<q_to_d; ++t )
    {
        weightGrid.RealWeight(t) = 0;
        weightGrid.ImagWeight(t) = 0;
    }

    // We seek performance by isolating the Lagrangian interpolation as 
    // a matrix-vector multiplication.
    //
    // To do so, the spatial weight recursion is broken into three updates:
    // For each child c:
    //  1) scale the old weights with the appropriate exponentials
    //  2) multiply the lagrangian matrix against the scaled weights
    //  3) scale and accumulate the result of the lagrangian mat-vec
    std::vector<R> phiResults;
    std::vector<R> sinResults;
    std::vector<R> cosResults;
    std::vector< std::tr1::array<R,d> > pPoint( 1 );
    std::vector< std::tr1::array<R,d> > xPoints( q_to_d );
    const std::vector<R>& spatialMaps = context.GetSpatialMaps();
    const std::vector< std::tr1::array<R,d> >& chebyshevGrid = 
        context.GetChebyshevGrid();
    for( std::size_t cLocal=0; 
         cLocal<(1u<<(d-log2NumMergingProcesses)); 
         ++cLocal )
    {
        // Step 1: scale the old weights
        WeightGrid<R,d,q> scaledWeightGrid;
        const std::size_t c = (cLocal<<log2NumMergingProcesses) + myTeamRank;
        const std::size_t key = parentOffset + cLocal;
        for( std::size_t j=0; j<d; ++j )
            pPoint[0][j] = p0B[j] + ( (c>>j)&1 ? wB[j]/4 : -wB[j]/4 );
        for( std::size_t tPrime=0; tPrime<q_to_d; ++tPrime )
            for( std::size_t j=0; j<d; ++j )
                xPoints[tPrime][j] = 
                    x0Ap[j] + (2*wA[j])*chebyshevGrid[tPrime][j];
        Phi.BatchEvaluate( xPoints, pPoint, phiResults );
        for( std::size_t tPrime=0; tPrime<q_to_d; ++tPrime )
            phiResults[tPrime] *= -TwoPi;
        SinCosBatch( phiResults, sinResults, cosResults );
        for( std::size_t tPrime=0; tPrime<q_to_d; ++tPrime )
        {
            const R realWeight = oldWeightGridList[key].RealWeight(tPrime);
            const R imagWeight = oldWeightGridList[key].ImagWeight(tPrime);
            scaledWeightGrid.RealWeight(tPrime) = 
                cosResults[tPrime]*realWeight - sinResults[tPrime]*imagWeight;
            scaledWeightGrid.ImagWeight(tPrime) = 
                sinResults[tPrime]*realWeight + cosResults[tPrime]*imagWeight;
        }

        // Step 2: perform the matrix-vector multiplies
        WeightGrid<R,d,q> expandedWeightGrid;
        Gemm
        ( 'N', 'N', q_to_d, 2, q_to_d,
          (R)1, &spatialMaps[ARelativeToAp*q_to_2d], q_to_d,
                scaledWeightGrid.Buffer(),           q_to_d,
          (R)0, expandedWeightGrid.Buffer(),         q_to_d );

        // Step 3: scale the result
        for( std::size_t t=0; t<Pow<q,d>::val; ++t )
            for( std::size_t j=0; j<d; ++j )
                xPoints[t][j] = x0A[j] + wA[j]*chebyshevGrid[t][j];
        Phi.BatchEvaluate( xPoints, pPoint, phiResults );
        for( std::size_t t=0; t<q_to_d; ++t )
            phiResults[t] *= TwoPi;
        SinCosBatch( phiResults, sinResults, cosResults );
        for( std::size_t t=0; t<q_to_d; ++t )
        {
            const R realWeight = expandedWeightGrid.RealWeight(t);
            const R imagWeight = expandedWeightGrid.ImagWeight(t);
            weightGrid.RealWeight(t) += 
                cosResults[t]*realWeight - sinResults[t]*imagWeight;
            weightGrid.ImagWeight(t) += 
                sinResults[t]*realWeight + cosResults[t]*imagWeight;
        }
    }
}

} // freq_to_spatial
} // bfio

#endif // BFIO_FREQ_TO_SPATIAL_SPATIAL_WEIGHT_RECURSION_HPP

