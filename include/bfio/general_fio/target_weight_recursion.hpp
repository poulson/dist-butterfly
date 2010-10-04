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
#ifndef BFIO_GENERAL_FIO_TARGET_WEIGHT_RECURSION_HPP
#define BFIO_GENERAL_FIO_TARGET_WEIGHT_RECURSION_HPP 1

#include <cstddef>
#include <cstring>
#include <vector>

#include "bfio/structures/array.hpp"
#include "bfio/structures/plan.hpp"
#include "bfio/structures/weight_grid.hpp"
#include "bfio/structures/weight_grid_list.hpp"

#include "bfio/functors/phase_functor.hpp"

#include "bfio/tools/blas.hpp"
#include "bfio/tools/special_functions.hpp"

#include "bfio/general_fio/context.hpp"

namespace bfio {
namespace general_fio {

template<typename R,std::size_t d,std::size_t q>
void
TargetWeightRecursion
( const general_fio::Context<R,d,q>& context,
  const Plan<d>& plan,
  const PhaseFunctor<R,d>& Phi,
  const std::size_t log2NumMergingProcesses,
  const std::size_t myClusterRank,
  const std::size_t ARelativeToAp,
  const Array<R,d>& x0A,
  const Array<R,d>& x0Ap,
  const Array<R,d>& p0B,
  const Array<R,d>& wA,
  const Array<R,d>& wB,
  const std::size_t parentInteractionOffset,
  const WeightGridList<R,d,q>& oldWeightGridList,
        WeightGrid<R,d,q>& weightGrid )
{
    const std::size_t q_to_d = Pow<q,d>::val;
    const std::size_t q_to_2d = Pow<q,2*d>::val;

    std::memset( weightGrid.Buffer(), 0, 2*q_to_d*sizeof(R) );

    // We seek performance by isolating the Lagrangian interpolation as 
    // a matrix-vector multiplication.
    //
    // To do so, the target weight recursion is broken into three updates:
    // For each child c:
    //  1) scale the old weights with the appropriate exponentials
    //  2) multiply the lagrangian matrix against the scaled weights
    //  3) scale and accumulate the result of the lagrangian mat-vec
    std::vector<R> phiResults;
    std::vector<R> sinResults;
    std::vector<R> cosResults;
    std::vector< Array<R,d> > pPoint( 1 );
    std::vector< Array<R,d> > xPoints( q_to_d );
    const std::vector<R>& targetMaps = context.GetTargetMaps();
    const std::vector< Array<R,d> >& chebyshevGrid = context.GetChebyshevGrid();
    for( std::size_t cLocal=0; 
         cLocal<(1u<<(d-log2NumMergingProcesses)); 
         ++cLocal )
    {
        // Step 1: scale the old weights
        WeightGrid<R,d,q> scaledWeightGrid;
        const std::size_t c = 
            ( plan.BackwardSourcePartitioning() ? 
              cLocal + (myClusterRank<<(d-log2NumMergingProcesses)) :
              (cLocal<<log2NumMergingProcesses) + myClusterRank );
        const std::size_t interactionIndex = parentInteractionOffset + cLocal;
        for( std::size_t j=0; j<d; ++j )
            pPoint[0][j] = p0B[j] + ( (c>>j)&1 ? wB[j]/4 : -wB[j]/4 );
        {
            R* xPointsBuffer = &(xPoints[0][0]);
            const R* wABuffer = &wA[0];
            const R* x0ApBuffer = &x0Ap[0];
            const R* chebyshevBuffer = &(chebyshevGrid[0][0]);
            for( std::size_t tPrime=0; tPrime<q_to_d; ++tPrime )
                for( std::size_t j=0; j<d; ++j )
                    xPointsBuffer[tPrime*d+j] = 
                        x0ApBuffer[j] + 
                        2*wABuffer[j]*chebyshevBuffer[tPrime*d+j];
        }
        Phi.BatchEvaluate( xPoints, pPoint, phiResults );
        {
            R* phiBuffer = &phiResults[0];
            for( std::size_t tPrime=0; tPrime<q_to_d; ++tPrime )
                phiBuffer[tPrime] *= -TwoPi;
        }
        SinCosBatch( phiResults, sinResults, cosResults );
        {
            R* scaledRealBuffer = scaledWeightGrid.RealBuffer();
            R* scaledImagBuffer = scaledWeightGrid.ImagBuffer();
            const R* cosBuffer = &cosResults[0];
            const R* sinBuffer = &sinResults[0];
            const R* oldRealBuffer = 
                oldWeightGridList[interactionIndex].RealBuffer();
            const R* oldImagBuffer = 
                oldWeightGridList[interactionIndex].ImagBuffer();
            for( std::size_t tPrime=0; tPrime<q_to_d; ++tPrime )
            {
                const R realPhase = cosBuffer[tPrime];
                const R imagPhase = sinBuffer[tPrime];
                const R realWeight = oldRealBuffer[tPrime];
                const R imagWeight = oldImagBuffer[tPrime];
                scaledRealBuffer[tPrime] = 
                    realPhase*realWeight - imagPhase*imagWeight;
                scaledImagBuffer[tPrime] = 
                    imagPhase*realWeight + realPhase*imagWeight;
            }
        }

        // Step 2: perform the matrix-vector multiplies
        WeightGrid<R,d,q> expandedWeightGrid;
        Gemm
        ( 'N', 'N', q_to_d, 2, q_to_d,
          (R)1, &targetMaps[ARelativeToAp*q_to_2d], q_to_d,
                scaledWeightGrid.Buffer(),          q_to_d,
          (R)0, expandedWeightGrid.Buffer(),        q_to_d );

        // Step 3: scale the result
        {
            R* xPointsBuffer = &(xPoints[0][0]);
            const R* wABuffer = &wA[0];
            const R* x0ABuffer = &x0A[0];
            const R* chebyshevBuffer = &(chebyshevGrid[0][0]);
            for( std::size_t t=0; t<Pow<q,d>::val; ++t )
                for( std::size_t j=0; j<d; ++j )
                    xPointsBuffer[t*d+j] = 
                        x0ABuffer[j] + wABuffer[j]*chebyshevBuffer[t*d+j];
        }
        Phi.BatchEvaluate( xPoints, pPoint, phiResults );
        {
            R* phiBuffer = &phiResults[0];
            for( std::size_t t=0; t<q_to_d; ++t )
                phiBuffer[t] *= TwoPi;
        }
        SinCosBatch( phiResults, sinResults, cosResults );
        {
            R* realBuffer = weightGrid.RealBuffer();
            R* imagBuffer = weightGrid.ImagBuffer();
            const R* cosBuffer = &cosResults[0];
            const R* sinBuffer = &sinResults[0];
            const R* expandedRealBuffer = expandedWeightGrid.RealBuffer();
            const R* expandedImagBuffer = expandedWeightGrid.ImagBuffer();
            for( std::size_t t=0; t<q_to_d; ++t )
            {
                const R realPhase = cosBuffer[t];
                const R imagPhase = sinBuffer[t];
                const R realWeight = expandedRealBuffer[t];
                const R imagWeight = expandedImagBuffer[t];
                realBuffer[t] += realPhase*realWeight - imagPhase*imagWeight;
                imagBuffer[t] += imagPhase*realWeight + realPhase*imagWeight;
            }
        }
    }
}

} // general_fio
} // bfio

#endif // BFIO_GENERAL_FIO_TARGET_WEIGHT_RECURSION_HPP

