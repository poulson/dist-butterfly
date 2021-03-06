/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of DistButterfly and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef DBF_BFLY_M2L_HPP
#define DBF_BFLY_M2L_HPP

#include "dist-butterfly/structures/box.hpp"
#include "dist-butterfly/structures/constrained_htree_walker.hpp"
#include "dist-butterfly/structures/plan.hpp"
#include "dist-butterfly/structures/weight_grid.hpp"
#include "dist-butterfly/structures/weight_grid_list.hpp"

#include "dist-butterfly/butterfly/context.hpp"

namespace dbf {

using std::array;
using std::complex;
using std::memset;
using std::size_t;
using std::vector;

namespace bfly {

template<typename R,size_t d,size_t q>
inline void
M2L
( const Context<R,d,q>& context,
  const Plan<d>& plan,
  const Amplitude<R,d>& amplitude,
  const Phase<R,d>& phase,
  const Box<R,d>& sBox,
  const Box<R,d>& tBox,
  const Box<R,d>& mySBox,
  const Box<R,d>& myTBox,
  const size_t log2LocalSBoxes,
  const size_t log2LocalTBoxes,
  const array<size_t,d>& log2LocalSBoxesPerDim,
  const array<size_t,d>& log2LocalTBoxesPerDim,
        WeightGridList<R,d,q>& weightGridList )
{
    typedef complex<R> C;
    const size_t q_to_d = Pow<q,d>::val;

    // Compute the width of the nodes at level log2N/2
    const size_t N = plan.GetN();
    const size_t log2N = Log2( N );
    const size_t level = log2N/2;
    array<R,d> wA, wB;
    for( size_t j=0; j<d; ++j )
    {
        wA[j] = tBox.widths[j] / (1<<level);
        wB[j] = sBox.widths[j] / (1<<(log2N-level));
    }

    vector<R> oldRealWeights( q_to_d ), oldImagWeights( q_to_d );
    const bool unitAmplitude = amplitude.IsUnity();
    const vector<array<R,d>>& chebyshevGrid = context.GetChebyshevGrid();
    ConstrainedHTreeWalker<d> AWalker( log2LocalTBoxesPerDim );
    for( size_t i=0; i<(1u<<log2LocalTBoxes); ++i, AWalker.Walk() )
    {
        const array<size_t,d>& A = AWalker.State();

        // Compute the coordinates and center of this target box
        array<R,d> x0A;
        for( size_t j=0; j<d; ++j )
            x0A[j] = myTBox.offsets[j] + (A[j]+R(1)/R(2))*wA[j];

        vector<array<R,d>> xPoints( q_to_d );
        {
            R* RESTRICT xPointsBuffer = &xPoints[0][0];
            const R* RESTRICT x0ABuffer = &x0A[0];
            const R* RESTRICT wABuffer = &wA[0];
            const R* RESTRICT chebyshevBuffer = &chebyshevGrid[0][0];
            for( size_t t=0; t<q_to_d; ++t )
                for( size_t j=0; j<d; ++j )
                    xPointsBuffer[t*d+j] = 
                        x0ABuffer[j] + wABuffer[j]*chebyshevBuffer[t*d+j];
        }

        vector<C> ampResults;
        vector<R> phiResults, sinResults, cosResults;
        ConstrainedHTreeWalker<d> BWalker( log2LocalSBoxesPerDim );
        for( size_t k=0; k<(1u<<log2LocalSBoxes); ++k, BWalker.Walk() )
        {
            const array<size_t,d>& B = BWalker.State();

            // Compute the coordinates and center of this source box
            array<R,d> p0B;
            for( size_t j=0; j<d; ++j )
                p0B[j] = mySBox.offsets[j] + (B[j]+R(1)/R(2))*wB[j];

            vector<array<R,d>> pPoints( q_to_d );
            {
                R* RESTRICT pPointsBuffer = &pPoints[0][0];
                const R* RESTRICT p0BBuffer = &p0B[0];
                const R* RESTRICT wBBuffer = &wB[0];
                const R* RESTRICT chebyshevBuffer = &chebyshevGrid[0][0];
                for( size_t t=0; t<q_to_d; ++t )
                    for( size_t j=0; j<d; ++j )
                        pPointsBuffer[t*d+j] = 
                            p0BBuffer[j] + wBBuffer[j]*chebyshevBuffer[t*d+j];
            }

            phase.BatchEvaluate( xPoints, pPoints, phiResults );
            SinCosBatch( phiResults, sinResults, cosResults );
            const size_t key = k+(i<<log2LocalSBoxes);

            memcpy
            ( &oldRealWeights[0], weightGridList[key].RealBuffer(), 
              q_to_d*sizeof(R) );
            memcpy
            ( &oldImagWeights[0], weightGridList[key].ImagBuffer(),
              q_to_d*sizeof(R) );
            memset( weightGridList[key].Buffer(), 0, 2*q_to_d*sizeof(R) );
            R* RESTRICT reals = weightGridList[key].RealBuffer();
            R* RESTRICT imags = weightGridList[key].ImagBuffer();
            const R* RESTRICT oldReals = &oldRealWeights[0];
            const R* RESTRICT oldImags = &oldImagWeights[0];
            const R* RESTRICT cosBuffer = &cosResults[0];
            const R* RESTRICT sinBuffer = &sinResults[0];
            if( unitAmplitude )
            {
                for( size_t t=0; t<q_to_d; ++t )
                {
                    for( size_t tPrime=0; tPrime<q_to_d; ++tPrime )
                    {
                        const R realWeight = oldReals[tPrime];
                        const R imagWeight = oldImags[tPrime];
                        const R realPhase = cosBuffer[t*q_to_d+tPrime];
                        const R imagPhase = sinBuffer[t*q_to_d+tPrime];
                        reals[t] += realPhase*realWeight - imagPhase*imagWeight;
                        imags[t] += imagPhase*realWeight + realPhase*imagWeight;
                    }
                }
            }
            else
            {
                amplitude.BatchEvaluate( xPoints, pPoints, ampResults );
                const C* RESTRICT ampBuffer = &ampResults[0];
                for( size_t t=0; t<q_to_d; ++t )
                {
                    for( size_t tPrime=0; tPrime<q_to_d; ++tPrime )
                    {
                        const R realWeight = oldReals[tPrime];
                        const R imagWeight = oldImags[tPrime];
                        const R realPhase = cosBuffer[t*q_to_d+tPrime];
                        const R imagPhase = sinBuffer[t*q_to_d+tPrime];
                        const R realBeta = 
                            realPhase*realWeight - imagPhase*imagWeight;
                        const R imagBeta = 
                            imagPhase*realWeight + realPhase*imagWeight;
                        const R realAmp = ampBuffer[t*q_to_d+tPrime].real();
                        const R imagAmp = ampBuffer[t*q_to_d+tPrime].imag();
                        reals[t] += realAmp*realBeta - imagAmp*imagBeta;
                        imags[t] += imagAmp*realBeta + realAmp*imagBeta;
                    }
                }
            }
        }
    }
}

} // bfly
} // dbf

#endif // ifndef DBF_BFLY_M2L_HPP
