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
#ifndef BFIO_GENERAL_FIO_SWITCH_TO_TARGET_INTERP_HPP
#define BFIO_GENERAL_FIO_SWITCH_TO_TARGET_INTERP_HPP 1

#include <cstddef>
#include <complex>
#include <vector>

#include "bfio/structures/array.hpp"
#include "bfio/structures/box.hpp"
#include "bfio/structures/constrained_htree_walker.hpp"
#include "bfio/structures/plan.hpp"
#include "bfio/structures/weight_grid.hpp"
#include "bfio/structures/weight_grid_list.hpp"

#include "bfio/general_fio/context.hpp"

namespace bfio {
namespace general_fio {

template<typename R,std::size_t d,std::size_t q>
void
SwitchToTargetInterp
( const general_fio::Context<R,d,q>& context,
  const Plan<d>& plan,
  const AmplitudeFunctor<R,d>& Amp,
  const PhaseFunctor<R,d>& Phi,
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

    const bool unitAmplitude = Amp.IsUnity();
    const std::vector< Array<R,d> >& chebyshevGrid = context.GetChebyshevGrid();
    ConstrainedHTreeWalker<d> AWalker( log2LocalTargetBoxesPerDim );
    WeightGridList<R,d,q> oldWeightGridList( weightGridList );
    for( std::size_t i=0; i<(1u<<log2LocalTargetBoxes); ++i, AWalker.Walk() )
    {
        const Array<std::size_t,d> A = AWalker.State();

        // Compute the coordinates and center of this target box
        Array<R,d> x0A;
        for( std::size_t j=0; j<d; ++j )
            x0A[j] = myTargetBox.offsets[j] + (A[j]+0.5)*wA[j];

        std::vector< Array<R,d> > xPoints( q_to_d );
        {
            R* xPointsBuffer = &(xPoints[0][0]);
            const R* x0ABuffer = &x0A[0];
            const R* wABuffer = &wA[0];
            const R* chebyshevBuffer = &(chebyshevGrid[0][0]);
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
                R* pPointsBuffer = &(pPoints[0][0]);
                const R* p0BBuffer = &p0B[0];
                const R* wBBuffer = &wB[0];
                const R* chebyshevBuffer = &(chebyshevGrid[0][0]);
                for( std::size_t t=0; t<q_to_d; ++t )
                    for( std::size_t j=0; j<d; ++j )
                        pPointsBuffer[t*d+j] = 
                            p0BBuffer[j] + wBBuffer[j]*chebyshevBuffer[t*d+j];
            }

            Phi.BatchEvaluate( xPoints, pPoints, phiResults );
            {
                R* phiBuffer = &phiResults[0];
                for( std::size_t j=0; j<phiResults.size(); ++j )
                    phiBuffer[j] *= TwoPi;
            }
            SinCosBatch( phiResults, sinResults, cosResults );
            const std::size_t key = k+(i<<log2LocalSourceBoxes);

            std::memset( weightGridList[key].Buffer(), 0, 2*q_to_d*sizeof(R) );
            R* realBuffer = weightGridList[key].RealBuffer();
            R* imagBuffer = weightGridList[key].ImagBuffer();
            const WeightGrid<R,d,q>& oldGrid = oldWeightGridList[key];
            const R* oldRealBuffer = oldGrid.RealBuffer();
            const R* oldImagBuffer = oldGrid.ImagBuffer();
            const R* cosBuffer = &cosResults[0];
            const R* sinBuffer = &sinResults[0];
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
                Amp.BatchEvaluate( xPoints, pPoints, ampResults );
                const C* ampBuffer = &ampResults[0];
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

} // general_fio
} // bfio

#endif // BFIO_GENERAL_FIO_SWITCH_TO_TARGET_INTERP_HPP

