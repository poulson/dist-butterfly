/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of ButterflyFIO and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef BFIO_RFIO_SOURCE_WEIGHT_RECURSION_HPP
#define BFIO_RFIO_SOURCE_WEIGHT_RECURSION_HPP

#include <array>
#include <cstddef>
#include <cstring>
#include <vector>

#include "bfio/constants.hpp"

#include "bfio/structures/weight_grid.hpp"
#include "bfio/structures/weight_grid_list.hpp"

#include "bfio/functors/phase.hpp"

#include "bfio/tools/special_functions.hpp"

#include "bfio/rfio/context.hpp"

namespace bfio {

using std::array;
using std::memset;
using std::size_t;
using std::vector;

namespace rfio {

// 1d specialization
template<typename R,size_t q>
void
SourceWeightRecursion
( const rfio::Context<R,1,q>& context,
  const Plan<1>& plan,
  const Phase<R,1>& phase,
  const size_t level,
  const array<R,1>& x0A,
  const array<R,1>& p0B,
  const array<R,1>& wB,
  const size_t parentInteractionOffset,
  const WeightGridList<R,1,q>& oldWeightGridList,
        WeightGrid<R,1,q>& weightGrid )
{
    memset( weightGrid.Buffer(), 0, 2*q*sizeof(R) );

    const size_t log2NumMergingProcesses = 
        plan.GetLog2NumMergingProcesses( level );
    const vector<R>& leftMap = context.GetLeftChebyshevMap();
    const vector<R>& rightMap = context.GetRightChebyshevMap();

    vector<R> phiResults, sinResults, cosResults;
    vector<array<R,1>> xPoint( 1, x0A ), pPoints( q );
    const vector<array<R,1>>& sourceChildGrids = context.GetSourceChildGrids();
    for( size_t cLocal=0;
         cLocal<(1u<<(1-log2NumMergingProcesses));
         ++cLocal )
    {
        //--------------------------------------------------------------------//
        // Step 1                                                             //
        //--------------------------------------------------------------------//
        const size_t interactionIndex = parentInteractionOffset + cLocal;
        const size_t c = plan.LocalToClusterSourceIndex( level, cLocal );

        // Form the set of p points to evaluate
        {
            R* RESTRICT pPointsBuffer = &pPoints[0][0];
            const R* RESTRICT sourceChildBuffer = &sourceChildGrids[c*q][0];
            for( size_t tPrime=0; tPrime<q; ++tPrime )
                pPointsBuffer[tPrime] = 
                    p0B[0] + wB[0]*sourceChildBuffer[tPrime];
        }

        // Form the phase factors
        phase.BatchEvaluate( xPoint, pPoints, phiResults );
        SinCosBatch( phiResults, sinResults, cosResults );

        WeightGrid<R,1,q> scaledWeightGrid;
        {
            R* RESTRICT scaledRealBuffer = scaledWeightGrid.RealBuffer();
            R* RESTRICT scaledImagBuffer = scaledWeightGrid.ImagBuffer();
            const R* RESTRICT cosBuffer = &cosResults[0];
            const R* RESTRICT sinBuffer = &sinResults[0];
            const R* RESTRICT oldRealBuffer = 
                oldWeightGridList[interactionIndex].RealBuffer();
            const R* RESTRICT oldImagBuffer = 
                oldWeightGridList[interactionIndex].ImagBuffer();
            for( size_t tPrime=0; tPrime<q; ++tPrime )
            {
                const R realWeight = oldRealBuffer[tPrime];
                const R imagWeight = oldImagBuffer[tPrime];
                const R realPhase = cosBuffer[tPrime];
                const R imagPhase = sinBuffer[tPrime];
                scaledRealBuffer[tPrime] = 
                    realPhase*realWeight - imagPhase*imagWeight;
                scaledImagBuffer[tPrime] =
                    imagPhase*realWeight + realPhase*imagWeight;
            }
        }

        //--------------------------------------------------------------------//
        // Step 2                                                             //
        //--------------------------------------------------------------------//
        // TODO: Create preprocessor flag for switching to two Gemv's, as
        //       Gemm is probably not optimized for only 2 right-hand sides.
        {
            const R* mapBuffer = ( c&1 ? &rightMap[0] : &leftMap[0] );
            Gemm
            ( 'N', 'N', q, 2, q,
              (R)1, mapBuffer,                 q,
                    scaledWeightGrid.Buffer(), q,
              (R)1,       weightGrid.Buffer(), q );
        }
    }

    //------------------------------------------------------------------------//
    // Step 3                                                                 //
    //------------------------------------------------------------------------//
    const vector<array<R,1>>& chebyshevGrid = context.GetChebyshevGrid();
    {
        R* RESTRICT pPointsBuffer = &pPoints[0][0];
        const R* RESTRICT chebyshevBuffer = &chebyshevGrid[0][0];
        for( size_t t=0; t<q; ++t )
            pPointsBuffer[t] = p0B[0] + wB[0]*chebyshevBuffer[t];
    }
    phase.BatchEvaluate( xPoint, pPoints, phiResults );
    SinCosBatch( phiResults, sinResults, cosResults );
    {
        R* RESTRICT realBuffer = weightGrid.RealBuffer();
        R* RESTRICT imagBuffer = weightGrid.ImagBuffer();
        const R* RESTRICT cosBuffer = &cosResults[0];
        const R* RESTRICT sinBuffer = &sinResults[0];
        for( size_t t=0; t<q; ++t )
        {
            const R realPhase = cosBuffer[t];
            const R imagPhase = -sinBuffer[t];
            const R realWeight = realBuffer[t];
            const R imagWeight = imagBuffer[t];
            realBuffer[t] = realPhase*realWeight - imagPhase*imagWeight;
            imagBuffer[t] = imagPhase*realWeight + realPhase*imagWeight;
        }
    }
}

// 2d specialization
template<typename R,size_t q>
void
SourceWeightRecursion
( const rfio::Context<R,2,q>& context,
  const Plan<2>& plan,
  const Phase<R,2>& phase,
  const size_t level,
  const array<R,2>& x0A,
  const array<R,2>& p0B,
  const array<R,2>& wB,
  const size_t parentInteractionOffset,
  const WeightGridList<R,2,q>& oldWeightGridList,
        WeightGrid<R,2,q>& weightGrid )
{
    memset( weightGrid.Buffer(), 0, 2*q*q*sizeof(R) );

    const size_t log2NumMergingProcesses = 
        plan.GetLog2NumMergingProcesses( level );
    const vector<R>& leftMap = context.GetLeftChebyshevMap();
    const vector<R>& rightMap = context.GetRightChebyshevMap();

    vector<R> phiResults, sinResults, cosResults;
    vector<array<R,2>> xPoint( 1, x0A ), pPoints( q*q );
    const vector<array<R,2>>& sourceChildGrids = context.GetSourceChildGrids();
    for( size_t cLocal=0; cLocal<(1u<<(2-log2NumMergingProcesses)); ++cLocal )
    {
        //--------------------------------------------------------------------//
        // Step 1                                                             //
        //--------------------------------------------------------------------//
        const size_t interactionIndex = parentInteractionOffset + cLocal;
        const size_t c = plan.LocalToClusterSourceIndex( level, cLocal );

        // Form the set of p points to evaluate
        {
            R* RESTRICT pPointsBuffer = &pPoints[0][0];
            const R* RESTRICT wBBuffer = &wB[0];
            const R* RESTRICT p0Buffer = &p0B[0];
            const R* RESTRICT sourceChildBuffer = &sourceChildGrids[c*q*q][0];
            for( size_t tPrime=0; tPrime<q*q; ++tPrime )
                for( size_t j=0; j<2; ++j )
                    pPointsBuffer[tPrime*2+j] = 
                        p0Buffer[j] + 
                        wBBuffer[j]*sourceChildBuffer[tPrime*2+j];
        }

        // Form the phase factors
        phase.BatchEvaluate( xPoint, pPoints, phiResults );
        SinCosBatch( phiResults, sinResults, cosResults );

        WeightGrid<R,2,q> scaledWeightGrid;
        {
            R* RESTRICT scaledRealBuffer = scaledWeightGrid.RealBuffer();
            R* RESTRICT scaledImagBuffer = scaledWeightGrid.ImagBuffer();
            const R* RESTRICT cosBuffer = &cosResults[0];
            const R* RESTRICT sinBuffer = &sinResults[0];
            const R* RESTRICT oldRealBuffer = 
                oldWeightGridList[interactionIndex].RealBuffer();
            const R* RESTRICT oldImagBuffer =
                oldWeightGridList[interactionIndex].ImagBuffer();
            for( size_t tPrime=0; tPrime<q*q; ++tPrime )
            {
                const R realWeight = oldRealBuffer[tPrime];
                const R imagWeight = oldImagBuffer[tPrime];
                const R realPhase = cosBuffer[tPrime];
                const R imagPhase = sinBuffer[tPrime];
                scaledRealBuffer[tPrime] =
                    realPhase*realWeight - imagPhase*imagWeight;
                scaledImagBuffer[tPrime] = 
                    imagPhase*realWeight + realPhase*imagWeight;
            }
        }

        //--------------------------------------------------------------------//
        // Step 2                                                             //
        //--------------------------------------------------------------------//
        // Interpolate over the first dimension. We can take care of the real 
        // and imaginary weights at once.
        WeightGrid<R,2,q> tempWeightGrid;
        {
            const R* mapBuffer = ( c&1 ? &rightMap[0] : &leftMap[0] );
            Gemm
            ( 'N', 'N', q, 2*q, q,
              (R)1, mapBuffer,                 q,
                    scaledWeightGrid.Buffer(), q,
              (R)0,   tempWeightGrid.Buffer(), q );
        }

        // Interpolate over the second dimension. We can get away with applying
        // our maps with a gemm on the real and imag parts.
        {
            const R* mapBuffer = ( (c>>1)&1 ? &rightMap[0] : &leftMap[0] );
            Gemm
            ( 'N', 'T', q, q, q,
              (R)1, tempWeightGrid.RealBuffer(), q,
                    mapBuffer,                   q,
              (R)1, weightGrid.RealBuffer(),     q );
            Gemm
            ( 'N', 'T', q, q, q,
              (R)1, tempWeightGrid.ImagBuffer(), q,
                    mapBuffer,                   q,
              (R)1, weightGrid.ImagBuffer(),     q );
        }
    }

    //------------------------------------------------------------------------//
    // Step 3                                                                 //
    //------------------------------------------------------------------------//
    const vector<array<R,2>>& chebyshevGrid = context.GetChebyshevGrid();
    {
        R* RESTRICT pPointsBuffer = &pPoints[0][0];
        const R* RESTRICT wBBuffer = &wB[0];
        const R* RESTRICT p0Buffer = &p0B[0];
        const R* RESTRICT chebyshevBuffer = &chebyshevGrid[0][0];
        for( size_t t=0; t<q*q; ++t )
            for( size_t j=0; j<2; ++j )
                pPointsBuffer[t*2+j] = 
                    p0Buffer[j] + wBBuffer[j]*chebyshevBuffer[t*2+j];
    }
    phase.BatchEvaluate( xPoint, pPoints, phiResults );
    SinCosBatch( phiResults, sinResults, cosResults );
    {
        R* RESTRICT realBuffer = weightGrid.RealBuffer();
        R* RESTRICT imagBuffer = weightGrid.ImagBuffer();
        const R* RESTRICT cosBuffer = &cosResults[0];
        const R* RESTRICT sinBuffer = &sinResults[0];
        for( size_t t=0; t<q*q; ++t )
        {
            const R realPhase = cosBuffer[t];
            const R imagPhase = -sinBuffer[t];
            const R realWeight = realBuffer[t];
            const R imagWeight = imagBuffer[t];
            realBuffer[t] = realPhase*realWeight - imagPhase*imagWeight;
            imagBuffer[t] = imagPhase*realWeight + realPhase*imagWeight;
        }
    }
}

// Fallback for 3d and above
template<typename R,size_t d,size_t q>
void
SourceWeightRecursion
( const rfio::Context<R,d,q>& context,
  const Plan<d>& plan,
  const Phase<R,d>& phase,
  const size_t level,
  const array<R,d>& x0A,
  const array<R,d>& p0B,
  const array<R,d>& wB,
  const size_t parentInteractionOffset,
  const WeightGridList<R,d,q>& oldWeightGridList,
        WeightGrid<R,d,q>& weightGrid )
{
    const size_t q_to_d = Pow<q,d>::val;
    memset( weightGrid.Buffer(), 0, 2*q_to_d*sizeof(R) );

    const size_t log2NumMergingProcesses = 
        plan.GetLog2NumMergingProcesses( level );
    const vector<R>& leftMap = context.GetLeftChebyshevMap();
    const vector<R>& rightMap = context.GetRightChebyshevMap();

    vector<R> phiResults, sinResults, cosResults;
    vector<array<R,d>> xPoint( 1, x0A ), pPoints( q_to_d );
    const vector<array<R,d>>& sourceChildGrids = context.GetSourceChildGrids();
    WeightGrid<R,d,q> scaledWeightGrid;
    for( size_t cLocal=0; cLocal<(1u<<(d-log2NumMergingProcesses)); ++cLocal )
    {
        //--------------------------------------------------------------------//
        // Step 1                                                             //
        //--------------------------------------------------------------------//
        const size_t interactionIndex = parentInteractionOffset + cLocal;
        const size_t c = plan.LocalToClusterSourceIndex( level, cLocal );

        // Form the set of p points to evaluate
        {
            R* RESTRICT pPointsBuffer = &pPoints[0][0];
            const R* RESTRICT wBBuffer = &wB[0];
            const R* RESTRICT p0BBuffer = &p0B[0];
            const R* RESTRICT sourceChildBuffer = 
                &sourceChildGrids[c*q_to_d][0];
            for( size_t tPrime=0; tPrime<q_to_d; ++tPrime )
                for( size_t j=0; j<d; ++j )
                    pPointsBuffer[tPrime*d+j] = 
                        p0BBuffer[j] + 
                        wBBuffer[j]*sourceChildBuffer[tPrime*d+j];
        }

        // Form the phase factors
        phase.BatchEvaluate( xPoint, pPoints, phiResults );
        SinCosBatch( phiResults, sinResults, cosResults );

        {
            R* RESTRICT scaledRealBuffer = scaledWeightGrid.RealBuffer();
            R* RESTRICT scaledImagBuffer = scaledWeightGrid.ImagBuffer();
            const R* RESTRICT cosBuffer = &cosResults[0];
            const R* RESTRICT sinBuffer = &sinResults[0];
            const R* RESTRICT oldRealBuffer = 
                oldWeightGridList[interactionIndex].RealBuffer();
            const R* RESTRICT oldImagBuffer = 
                oldWeightGridList[interactionIndex].ImagBuffer();
            for( size_t tPrime=0; tPrime<q_to_d; ++tPrime )
            {
                const R realWeight = oldRealBuffer[tPrime];
                const R imagWeight = oldImagBuffer[tPrime];
                const R realPhase = cosBuffer[tPrime];
                const R imagPhase = sinBuffer[tPrime];
                scaledRealBuffer[tPrime] = 
                    realPhase*realWeight - imagPhase*imagWeight;
                scaledImagBuffer[tPrime] = 
                    imagPhase*realWeight + realPhase*imagWeight;
            }
        }

        //--------------------------------------------------------------------//
        // Step 2                                                             //
        //--------------------------------------------------------------------//

        // Interpolate over the first dimension. This can be performed with a 
        // single gemm that takes care of both the real and imaginary weights.
        WeightGrid<R,d,q> tempWeightGrid;
        {
            const R* mapBuffer = ( c&1 ? &rightMap[0] : &leftMap[0] );
            Gemm
            ( 'N', 'N', q, 2*Pow<q,d-1>::val, q,
              (R)1, mapBuffer,                 q,
                    scaledWeightGrid.Buffer(), q,
              (R)0,   tempWeightGrid.Buffer(), q );
        }

        // Interpolate over the second dimension. We can do so by using 
        // q^(d-2) gemms on right-hand sides of size q x q for both the real 
        // and imaginary buffers.
        {
            R* realWriteBuffer = 
                ( d==2 ? weightGrid.RealBuffer()
                       : scaledWeightGrid.RealBuffer() );
            R* imagWriteBuffer = 
                ( d==2 ? weightGrid.ImagBuffer()
                       : scaledWeightGrid.ImagBuffer() );
            const R* realReadBuffer = tempWeightGrid.RealBuffer();
            const R* imagReadBuffer = tempWeightGrid.ImagBuffer();
            const R* mapBuffer = ( (c>>1)&1 ? &rightMap[0] : &leftMap[0] );
            for( size_t w=0; w<Pow<q,d-2>::val; ++w )
            {
                Gemm
                ( 'N', 'T', q, q, q,
                  (R)1, &realReadBuffer[w*q*q],  q,
                        mapBuffer,               q,
                  (R)0, &realWriteBuffer[w*q*q], q );
            }
            for( size_t w=0; w<Pow<q,d-2>::val; ++w )
            {
                Gemm
                ( 'N', 'T', q, q, q,
                  (R)1, &imagReadBuffer[w*q*q],  q,
                        mapBuffer,               q,
                  (R)0, &imagWriteBuffer[w*q*q], q );
            }
        }

        // Interpolate over the remaining dimensions. These are necessarily 
        // more expensive because we cannot make use of gemm.
        //
        // TODO: Compile flag for making the left and right maps transposed 
        //       to get contiguous memory access in the summations.
        //
        // TODO: Check if loading discontinuous chunks is faster than repeatedly
        //       striding over them.
        size_t q_to_j = q*q;
        for( size_t j=2; j<d; ++j )
        {
            const size_t stride = q_to_j;

            R* realWriteBuffer = 
                ( j==d-1 ? weightGrid.RealBuffer()
                         : ( j&1 ? scaledWeightGrid.RealBuffer()
                                 : tempWeightGrid.RealBuffer() ) );
            R* imagWriteBuffer = 
                ( j==d-1 ? weightGrid.ImagBuffer()
                         : ( j&1 ? scaledWeightGrid.ImagBuffer()
                                 : tempWeightGrid.ImagBuffer() ) );
            const R* realReadBuffer = 
                ( j&1 ? tempWeightGrid.RealBuffer() 
                      : scaledWeightGrid.RealBuffer() );
            const R* imagReadBuffer = 
                ( j&1 ? tempWeightGrid.ImagBuffer()
                      : scaledWeightGrid.ImagBuffer() );
            const R* RESTRICT mapBuffer = 
                ( (c>>j)&1 ? &rightMap[0] : &leftMap[0] );

            if( j != d-1 )
            {
                memset( realWriteBuffer, 0, q_to_d*sizeof(R) );
                memset( imagWriteBuffer, 0, q_to_d*sizeof(R) );
            }
            for( size_t p=0; p<q_to_d/(q_to_j*q); ++p )
            {
                const size_t offset = p*(q_to_j*q);
                R* RESTRICT offsetRealWriteBuffer = &realWriteBuffer[offset];
                R* RESTRICT offsetImagWriteBuffer = &imagWriteBuffer[offset];
                const R* RESTRICT offsetRealReadBuffer = &realReadBuffer[offset];
                const R* RESTRICT offsetImagReadBuffer = &imagReadBuffer[offset];
                for( size_t w=0; w<q_to_j; ++w )
                {
                    for( size_t t=0; t<q; ++t )
                    {
                        for( size_t tPrime=0; tPrime<q; ++tPrime )
                        {
                            offsetRealWriteBuffer[w+t*stride] +=
                                mapBuffer[t+tPrime*q] * 
                                offsetRealReadBuffer[w+tPrime*stride];
                        }
                    }
                    for( size_t t=0; t<q; ++t )
                    {
                        for( size_t tPrime=0; tPrime<q; ++tPrime )
                        {
                            offsetImagWriteBuffer[w+t*stride] +=
                                mapBuffer[t+tPrime*q] *
                                offsetImagReadBuffer[w+tPrime*stride];
                        }
                    }
                }
            }
            q_to_j *= q;
        }
    }

    //------------------------------------------------------------------------//
    // Step 3                                                                 //
    //------------------------------------------------------------------------//
    const vector<array<R,d>>& chebyshevGrid = context.GetChebyshevGrid();
    {
        R* RESTRICT pPointsBuffer = &pPoints[0][0];
        const R* RESTRICT wBBuffer = &wB[0];
        const R* RESTRICT p0BBuffer = &p0B[0];
        const R* RESTRICT chebyshevBuffer = &chebyshevGrid[0][0];
        for( size_t t=0; t<q_to_d; ++t )
            for( size_t j=0; j<d; ++j )
                pPointsBuffer[t*d+j] =  
                    p0BBuffer[j] + wBBuffer[j]*chebyshevBuffer[t*d+j];
    }
    phase.BatchEvaluate( xPoint, pPoints, phiResults );
    SinCosBatch( phiResults, sinResults, cosResults );
    {
        R* RESTRICT realBuffer = weightGrid.RealBuffer();
        R* RESTRICT imagBuffer = weightGrid.ImagBuffer();
        const R* RESTRICT cosBuffer = &cosResults[0];
        const R* RESTRICT sinBuffer = &sinResults[0];
        for( size_t t=0; t<q_to_d; ++t )
        {
            const R realPhase = cosBuffer[t];
            const R imagPhase = -sinBuffer[t];
            const R realWeight = realBuffer[t];
            const R imagWeight = imagBuffer[t];
            realBuffer[t] = realPhase*realWeight - imagPhase*imagWeight;
            imagBuffer[t] = imagPhase*realWeight + realPhase*imagWeight;
        }
    }
}

} // rfio
} // bfio

#endif // ifndef BFIO_RFIO_SOURCE_WEIGHT_RECURSION_HPP
