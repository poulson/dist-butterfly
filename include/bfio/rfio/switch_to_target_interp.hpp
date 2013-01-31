/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of ButterflyFIO and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef BFIO_RFIO_SWITCH_TO_TARGET_INTERP_HPP
#define BFIO_RFIO_SWITCH_TO_TARGET_INTERP_HPP

#include <cstddef>
#include <complex>
#include <vector>

#include "bfio/structures/array.hpp"
#include "bfio/structures/box.hpp"
#include "bfio/structures/constrained_htree_walker.hpp"
#include "bfio/structures/plan.hpp"
#include "bfio/structures/weight_grid.hpp"
#include "bfio/structures/weight_grid_list.hpp"

#include "bfio/rfio/context.hpp"

namespace bfio {
namespace rfio {

template<typename R,std::size_t d,std::size_t q>
void
SwitchToTargetInterp
( const rfio::Context<R,d,q>& context,
  const Plan<d>& plan,
  const Amplitude<R,d>& amplitude,
  const Phase<R,d>& phase,
  const Box<R,d>& sourceBox,
  const Box<R,d>& targetBox,
  const Box<R,d>& mySourceBox,
  const Box<R,d>& myTargetBox,
  const std::size_t log2LocalSourceBoxes,
  const std::size_t log2LocalTargetBoxes,
  const Array<std::size_t,d>& log2LocalSourceBoxesPerDim,
  const Array<std::size_t,d>& log2LocalTargetBoxesPerDim,
        WeightGridList<R,d,q>& weightGridList )
{
    typedef std::complex<R> C;
    const std::size_t q_to_d = Pow<q,d>::val;

    // Compute the width of the nodes at level log2N/2
    const std::size_t N = plan.GetN();
    const std::size_t log2N = Log2( N );
    const std::size_t level = log2N/2;
    Array<R,d> wA, wB;
    for( std::size_t j=0; j<d; ++j )
    {
        wA[j] = targetBox.widths[j] / (1<<level);
        wB[j] = sourceBox.widths[j] / (1<<(log2N-level));
    }

    std::vector<R> oldRealWeights( q_to_d );
    std::vector<R> oldImagWeights( q_to_d );
    const bool unitAmplitude = amplitude.IsUnity();
    const std::vector< Array<R,d> >& chebyshevGrid = context.GetChebyshevGrid();
    ConstrainedHTreeWalker<d> AWalker( log2LocalTargetBoxesPerDim );
    for( std::size_t i=0; i<(1u<<log2LocalTargetBoxes); ++i, AWalker.Walk() )
    {
        const Array<std::size_t,d> A = AWalker.State();

        // Compute the coordinates and center of this target box
        Array<R,d> x0A;
        for( std::size_t j=0; j<d; ++j )
            x0A[j] = myTargetBox.offsets[j] + (A[j]+0.5)*wA[j];

        std::vector< Array<R,d> > xPoints( q_to_d );
        {
            R* RESTRICT xPointsBuffer = &xPoints[0][0];
            const R* RESTRICT x0ABuffer = &x0A[0];
            const R* RESTRICT wABuffer = &wA[0];
            const R* RESTRICT chebyshevBuffer = &chebyshevGrid[0][0];
            for( std::size_t t=0; t<q_to_d; ++t )
                for( std::size_t j=0; j<d; ++j )
                    xPointsBuffer[t*d+j] = 
                        x0ABuffer[j] + wABuffer[j]*chebyshevBuffer[t*d+j];
        }

        std::vector<C> ampResults;
        std::vector<R> phiResults;
        std::vector<R> sinResults;
        std::vector<R> cosResults;
        ConstrainedHTreeWalker<d> BWalker( log2LocalSourceBoxesPerDim );
        for( std::size_t k=0; 
             k<(1u<<log2LocalSourceBoxes); 
             ++k, BWalker.Walk() )
        {
            const Array<std::size_t,d> B = BWalker.State();

            // Compute the coordinates and center of this source box
            Array<R,d> p0B;
            for( std::size_t j=0; j<d; ++j )
                p0B[j] = mySourceBox.offsets[j] + (B[j]+0.5)*wB[j];

            std::vector< Array<R,d> > pPoints( q_to_d );
            {
                R* RESTRICT pPointsBuffer = &pPoints[0][0];
                const R* RESTRICT p0BBuffer = &p0B[0];
                const R* RESTRICT wBBuffer = &wB[0];
                const R* RESTRICT chebyshevBuffer = &chebyshevGrid[0][0];
                for( std::size_t t=0; t<q_to_d; ++t )
                    for( std::size_t j=0; j<d; ++j )
                        pPointsBuffer[t*d+j] = 
                            p0BBuffer[j] + wBBuffer[j]*chebyshevBuffer[t*d+j];
            }

            phase.BatchEvaluate( xPoints, pPoints, phiResults );
            SinCosBatch( phiResults, sinResults, cosResults );
            const std::size_t key = k+(i<<log2LocalSourceBoxes);

            std::memcpy
            ( &oldRealWeights[0], weightGridList[key].RealBuffer(), 
              q_to_d*sizeof(R) );
            std::memcpy
            ( &oldImagWeights[0], weightGridList[key].ImagBuffer(),
              q_to_d*sizeof(R) );
            std::memset( weightGridList[key].Buffer(), 0, 2*q_to_d*sizeof(R) );
            R* RESTRICT realBuffer = weightGridList[key].RealBuffer();
            R* RESTRICT imagBuffer = weightGridList[key].ImagBuffer();
            const R* RESTRICT oldRealBuffer = &oldRealWeights[0];
            const R* RESTRICT oldImagBuffer = &oldImagWeights[0];
            const R* RESTRICT cosBuffer = &cosResults[0];
            const R* RESTRICT sinBuffer = &sinResults[0];
            if( unitAmplitude )
            {
                for( std::size_t t=0; t<q_to_d; ++t )
                {
                    for( std::size_t tPrime=0; tPrime<q_to_d; ++tPrime )
                    {
                        const R realWeight = oldRealBuffer[tPrime];
                        const R imagWeight = oldImagBuffer[tPrime];
                        const R realPhase = cosBuffer[t*q_to_d+tPrime];
                        const R imagPhase = sinBuffer[t*q_to_d+tPrime];
                        const R realBeta = 
                            realPhase*realWeight - imagPhase*imagWeight;
                        const R imagBeta = 
                            imagPhase*realWeight + realPhase*imagWeight;
                        realBuffer[t] += realBeta;
                        imagBuffer[t] += imagBeta;
                    }
                }
            }
            else
            {
                amplitude.BatchEvaluate( xPoints, pPoints, ampResults );
                const C* RESTRICT ampBuffer = &ampResults[0];
                for( std::size_t t=0; t<q_to_d; ++t )
                {
                    for( std::size_t tPrime=0; tPrime<q_to_d; ++tPrime )
                    {
                        const R realWeight = oldRealBuffer[tPrime];
                        const R imagWeight = oldImagBuffer[tPrime];
                        const R realPhase = cosBuffer[t*q_to_d+tPrime];
                        const R imagPhase = sinBuffer[t*q_to_d+tPrime];
                        const R realBeta = 
                            realPhase*realWeight - imagPhase*imagWeight;
                        const R imagBeta = 
                            imagPhase*realWeight + realPhase*imagWeight;
                        const R realAmp = real(ampBuffer[t*q_to_d+tPrime]);
                        const R imagAmp = imag(ampBuffer[t*q_to_d+tPrime]);
                        realBuffer[t] += realAmp*realBeta - imagAmp*imagBeta;
                        imagBuffer[t] += imagAmp*realBeta + realAmp*imagBeta;
                    }
                }
            }
        }
    }
}

} // rfio
} // bfio

#endif // ifndef BFIO_RFIO_SWITCH_TO_TARGET_INTERP_HPP
