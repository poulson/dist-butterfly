/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of DistButterfly and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef DBF_BFLY_SOURCE_WEIGHT_RECURSION_HPP
#define DBF_BFLY_SOURCE_WEIGHT_RECURSION_HPP

#include <cstring>

#include "dist-butterfly/constants.hpp"

#include "dist-butterfly/structures/weight_grid.hpp"
#include "dist-butterfly/structures/weight_grid_list.hpp"

#include "dist-butterfly/functors/phase.hpp"

#include "dist-butterfly/tools/special_functions.hpp"

#include "dist-butterfly/butterfly/context.hpp"

namespace dbf {

using std::array;
using std::memset;
using std::size_t;
using std::vector;

namespace bfly {

// 1d specialization
template<typename R,size_t q>
inline void
SourceWeightRecursion
( const Context<R,1,q>& context,
  const Plan<1>& plan,
  const Phase<R,1>& phase,
  const size_t level,
  const array<R,1>& x0A,
  const array<R,1>& p0B,
  const array<R,1>& wB,
  const size_t parentIOffset,
  const WeightGridList<R,1,q>& oldWeights,
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
    for( size_t cLocal=0; cLocal<(1u<<(1-log2NumMergingProcesses)); ++cLocal )
    {
        //--------------------------------------------------------------------//
        // Step 1                                                             //
        //--------------------------------------------------------------------//
#ifdef TIMING
        sWeightRecursionTimer1.Start();
#endif
        const size_t iIndex = parentIOffset + cLocal;
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
#ifdef TIMING
        sWeightRecursionTimer1Phase.Start();
#endif
        phase.BatchEvaluate( xPoint, pPoints, phiResults );
#ifdef TIMING
        sWeightRecursionTimer1Phase.Stop();
        sWeightRecursionTimer1SinCos.Start();
#endif
        SinCosBatch( phiResults, sinResults, cosResults );
#ifdef TIMING
        sWeightRecursionTimer1SinCos.Stop();
#endif

        WeightGrid<R,1,q> scaledWeightGrid;
        {
            R* RESTRICT scaledReals = scaledWeightGrid.RealBuffer();
            R* RESTRICT scaledImags = scaledWeightGrid.ImagBuffer();
            const R* RESTRICT cosBuffer = &cosResults[0];
            const R* RESTRICT sinBuffer = &sinResults[0];
            const R* RESTRICT oldReals = oldWeights[iIndex].RealBuffer();
            const R* RESTRICT oldImags = oldWeights[iIndex].ImagBuffer();
            for( size_t tPrime=0; tPrime<q; ++tPrime )
            {
                const R realWeight = oldReals[tPrime];
                const R imagWeight = oldImags[tPrime];
                const R realPhase = cosBuffer[tPrime];
                const R imagPhase = sinBuffer[tPrime];
                scaledReals[tPrime] = realPhase*realWeight-imagPhase*imagWeight;
                scaledImags[tPrime] = imagPhase*realWeight+realPhase*imagWeight;
            }
        }
#ifdef TIMING
        sWeightRecursionTimer1.Stop();
#endif

        //--------------------------------------------------------------------//
        // Step 2                                                             //
        //--------------------------------------------------------------------//
#ifdef TIMING
        sWeightRecursionTimer2.Start();
#endif
        // TODO: Create preprocessor flag for switching to two Gemv's, as
        //       Gemm is probably not optimized for only 2 right-hand sides.
        {
            const R* mapBuffer = ( c&1 ? &rightMap[0] : &leftMap[0] );
            Gemm
            ( 'N', 'N', q, 2, q,
              R(1), mapBuffer,                 q,
                    scaledWeightGrid.Buffer(), q,
              R(1), weightGrid.Buffer(),       q );
        }
#ifdef TIMING
        sWeightRecursionTimer2.Stop();
#endif
    }

    //------------------------------------------------------------------------//
    // Step 3                                                                 //
    //------------------------------------------------------------------------//
#ifdef TIMING
    sWeightRecursionTimer3.Start();
#endif
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
        R* RESTRICT reals = weightGrid.RealBuffer();
        R* RESTRICT imags = weightGrid.ImagBuffer();
        const R* RESTRICT cosBuffer = &cosResults[0];
        const R* RESTRICT sinBuffer = &sinResults[0];
        for( size_t t=0; t<q; ++t )
        {
            const R realPhase = cosBuffer[t];
            const R imagPhase = -sinBuffer[t];
            const R realWeight = reals[t];
            const R imagWeight = imags[t];
            reals[t] = realPhase*realWeight - imagPhase*imagWeight;
            imags[t] = imagPhase*realWeight + realPhase*imagWeight;
        }
    }
#ifdef TIMING
    sWeightRecursionTimer3.Stop();
#endif
}

// 2d specialization
template<typename R,size_t q>
inline void
SourceWeightRecursion
( const Context<R,2,q>& context,
  const Plan<2>& plan,
  const Phase<R,2>& phase,
  const size_t level,
  const array<R,2>& x0A,
  const array<R,2>& p0B,
  const array<R,2>& wB,
  const size_t parentIOffset,
  const WeightGridList<R,2,q>& oldWeights,
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
#ifdef TIMING
        sWeightRecursionTimer1.Start();
#endif

        const size_t iIndex = parentIOffset + cLocal;
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
#ifdef TIMING
        sWeightRecursionTimer1Phase.Start();
#endif
        phase.BatchEvaluate( xPoint, pPoints, phiResults );
#ifdef TIMING
        sWeightRecursionTimer1Phase.Stop();
        sWeightRecursionTimer1SinCos.Start();
#endif
        SinCosBatch( phiResults, sinResults, cosResults );
#ifdef TIMING
        sWeightRecursionTimer1SinCos.Stop();
#endif

        WeightGrid<R,2,q> scaledWeightGrid;
        {
            R* RESTRICT scaledReals = scaledWeightGrid.RealBuffer();
            R* RESTRICT scaledImags = scaledWeightGrid.ImagBuffer();
            const R* RESTRICT cosBuffer = &cosResults[0];
            const R* RESTRICT sinBuffer = &sinResults[0];
            const R* RESTRICT oldReals = oldWeights[iIndex].RealBuffer();
            const R* RESTRICT oldImags = oldWeights[iIndex].ImagBuffer();
            for( size_t tPrime=0; tPrime<q*q; ++tPrime )
            {
                const R realWeight = oldReals[tPrime];
                const R imagWeight = oldImags[tPrime];
                const R realPhase = cosBuffer[tPrime];
                const R imagPhase = sinBuffer[tPrime];
                scaledReals[tPrime] = realPhase*realWeight-imagPhase*imagWeight;
                scaledImags[tPrime] = imagPhase*realWeight+realPhase*imagWeight;
            }
        }
#ifdef TIMING
        sWeightRecursionTimer1.Stop();
#endif

        //--------------------------------------------------------------------//
        // Step 2                                                             //
        //--------------------------------------------------------------------//
#ifdef TIMING
        sWeightRecursionTimer2.Start();
#endif
        // Interpolate over the first dimension. We can take care of the real 
        // and imaginary weights at once.
        WeightGrid<R,2,q> tempWeightGrid;
        {
            const R* mapBuffer = ( c&1 ? &rightMap[0] : &leftMap[0] );
            Gemm
            ( 'N', 'N', q, 2*q, q,
              R(1), mapBuffer,                 q,
                    scaledWeightGrid.Buffer(), q,
              R(0), tempWeightGrid.Buffer(),   q );
        }

        // Interpolate over the second dimension. We can get away with applying
        // our maps with a gemm on the real and imag parts.
        {
            const R* mapBuffer = ( (c>>1)&1 ? &rightMap[0] : &leftMap[0] );
            Gemm
            ( 'N', 'T', q, q, q,
              R(1), tempWeightGrid.RealBuffer(), q,
                    mapBuffer,                   q,
              R(1), weightGrid.RealBuffer(),     q );
            Gemm
            ( 'N', 'T', q, q, q,
              R(1), tempWeightGrid.ImagBuffer(), q,
                    mapBuffer,                   q,
              R(1), weightGrid.ImagBuffer(),     q );
        }
#ifdef TIMING
        sWeightRecursionTimer2.Stop();
#endif
    }

    //------------------------------------------------------------------------//
    // Step 3                                                                 //
    //------------------------------------------------------------------------//
#ifdef TIMING
    sWeightRecursionTimer3.Start();
#endif
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
        R* RESTRICT reals = weightGrid.RealBuffer();
        R* RESTRICT imags = weightGrid.ImagBuffer();
        const R* RESTRICT cosBuffer = &cosResults[0];
        const R* RESTRICT sinBuffer = &sinResults[0];
        for( size_t t=0; t<q*q; ++t )
        {
            const R realPhase = cosBuffer[t];
            const R imagPhase = -sinBuffer[t];
            const R realWeight = reals[t];
            const R imagWeight = imags[t];
            reals[t] = realPhase*realWeight - imagPhase*imagWeight;
            imags[t] = imagPhase*realWeight + realPhase*imagWeight;
        }
    }
#ifdef TIMING
    sWeightRecursionTimer3.Stop();
#endif
}

// Fallback for 3d and above
template<typename R,size_t d,size_t q>
inline void
SourceWeightRecursion
( const Context<R,d,q>& context,
  const Plan<d>& plan,
  const Phase<R,d>& phase,
  const size_t level,
  const array<R,d>& x0A,
  const array<R,d>& p0B,
  const array<R,d>& wB,
  const size_t parentIOffset,
  const WeightGridList<R,d,q>& oldWeights,
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
#ifdef TIMING
        sWeightRecursionTimer1.Start();
#endif
        const size_t iIndex = parentIOffset + cLocal;
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
#ifdef TIMING
        sWeightRecursionTimer1Phase.Start();
#endif
        phase.BatchEvaluate( xPoint, pPoints, phiResults );
#ifdef TIMING
        sWeightRecursionTimer1Phase.Stop();
        sWeightRecursionTimer1SinCos.Start();
#endif
        SinCosBatch( phiResults, sinResults, cosResults );
#ifdef TIMING
        sWeightRecursionTimer1SinCos.Stop();
#endif

        {
            R* RESTRICT scaledReals = scaledWeightGrid.RealBuffer();
            R* RESTRICT scaledImags = scaledWeightGrid.ImagBuffer();
            const R* RESTRICT cosBuffer = &cosResults[0];
            const R* RESTRICT sinBuffer = &sinResults[0];
            const R* RESTRICT oldReals = oldWeights[iIndex].RealBuffer();
            const R* RESTRICT oldImags = oldWeights[iIndex].ImagBuffer();
            for( size_t tPrime=0; tPrime<q_to_d; ++tPrime )
            {
                const R realWeight = oldReals[tPrime];
                const R imagWeight = oldImags[tPrime];
                const R realPhase = cosBuffer[tPrime];
                const R imagPhase = sinBuffer[tPrime];
                scaledReals[tPrime] = realPhase*realWeight-imagPhase*imagWeight;
                scaledImags[tPrime] = imagPhase*realWeight+realPhase*imagWeight;
            }
        }
#ifdef TIMING
        sWeightRecursionTimer1.Stop();
#endif

        //--------------------------------------------------------------------//
        // Step 2                                                             //
        //--------------------------------------------------------------------//
#ifdef TIMING
        sWeightRecursionTimer2.Start();
#endif

        // Interpolate over the first dimension. This can be performed with a 
        // single gemm that takes care of both the real and imaginary weights.
        WeightGrid<R,d,q> tempWeightGrid;
        {
            const R* mapBuffer = ( c&1 ? &rightMap[0] : &leftMap[0] );
            Gemm
            ( 'N', 'N', q, 2*Pow<q,d-1>::val, q,
              R(1), mapBuffer,                 q,
                    scaledWeightGrid.Buffer(), q,
              R(0), tempWeightGrid.Buffer(),   q );
        }

        // Interpolate over the second dimension. We can do so by using 
        // q^(d-2) gemms on right-hand sides of size q x q for both the real 
        // and imaginary buffers.
        {
            R* realWriteBuffer = scaledWeightGrid.RealBuffer();
            R* imagWriteBuffer = scaledWeightGrid.ImagBuffer();
            const R* realReadBuffer = tempWeightGrid.RealBuffer();
            const R* imagReadBuffer = tempWeightGrid.ImagBuffer();
            const R* mapBuffer = ( (c>>1)&1 ? &rightMap[0] : &leftMap[0] );
            for( size_t w=0; w<Pow<q,d-2>::val; ++w )
            {
                Gemm
                ( 'N', 'T', q, q, q,
                  R(1), &realReadBuffer[w*q*q],  q,
                        mapBuffer,               q,
                  R(0), &realWriteBuffer[w*q*q], q );
            }
            for( size_t w=0; w<Pow<q,d-2>::val; ++w )
            {
                Gemm
                ( 'N', 'T', q, q, q,
                  R(1), &imagReadBuffer[w*q*q],  q,
                        mapBuffer,               q,
                  R(0), &imagWriteBuffer[w*q*q], q );
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

            R* realWrites = 
                ( j==d-1 ? weightGrid.RealBuffer()
                         : ( j&1 ? scaledWeightGrid.RealBuffer()
                                 : tempWeightGrid.RealBuffer() ) );
            R* imagWrites = 
                ( j==d-1 ? weightGrid.ImagBuffer()
                         : ( j&1 ? scaledWeightGrid.ImagBuffer()
                                 : tempWeightGrid.ImagBuffer() ) );
            const R* realReads = 
                ( j&1 ? tempWeightGrid.RealBuffer() 
                      : scaledWeightGrid.RealBuffer() );
            const R* imagReads = 
                ( j&1 ? tempWeightGrid.ImagBuffer()
                      : scaledWeightGrid.ImagBuffer() );
            const R* RESTRICT mapBuffer = 
                ( (c>>j)&1 ? &rightMap[0] : &leftMap[0] );

            if( j != d-1 )
            {
                memset( realWrites, 0, q_to_d*sizeof(R) );
                memset( imagWrites, 0, q_to_d*sizeof(R) );
            }
            for( size_t p=0; p<q_to_d/(q_to_j*q); ++p )
            {
                const size_t offset = p*(q_to_j*q);
                R* RESTRICT offsetRealWrites = &realWrites[offset];
                R* RESTRICT offsetImagWrites = &imagWrites[offset];
                const R* RESTRICT offsetRealReads = &realReads[offset];
                const R* RESTRICT offsetImagReads = &imagReads[offset];
                for( size_t w=0; w<q_to_j; ++w )
                {
                    for( size_t t=0; t<q; ++t )
                    {
                        for( size_t tPrime=0; tPrime<q; ++tPrime )
                        {
                            offsetRealWrites[w+t*stride] +=
                                mapBuffer[t+tPrime*q] * 
                                offsetRealReads[w+tPrime*stride];
                        }
                    }
                    for( size_t t=0; t<q; ++t )
                    {
                        for( size_t tPrime=0; tPrime<q; ++tPrime )
                        {
                            offsetImagWrites[w+t*stride] +=
                                mapBuffer[t+tPrime*q] *
                                offsetImagReads[w+tPrime*stride];
                        }
                    }
                }
            }
            q_to_j *= q;
        }
#ifdef TIMING
        sWeightRecursionTimer2.Stop();
#endif
    }

    //------------------------------------------------------------------------//
    // Step 3                                                                 //
    //------------------------------------------------------------------------//
#ifdef TIMING
    sWeightRecursionTimer3.Start();
#endif

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
        R* RESTRICT reals = weightGrid.RealBuffer();
        R* RESTRICT imags = weightGrid.ImagBuffer();
        const R* RESTRICT cosBuffer = &cosResults[0];
        const R* RESTRICT sinBuffer = &sinResults[0];
        for( size_t t=0; t<q_to_d; ++t )
        {
            const R realPhase = cosBuffer[t];
            const R imagPhase = -sinBuffer[t];
            const R realWeight = reals[t];
            const R imagWeight = imags[t];
            reals[t] = realPhase*realWeight - imagPhase*imagWeight;
            imags[t] = imagPhase*realWeight + realPhase*imagWeight;
        }
    }
#ifdef TIMING
    sWeightRecursionTimer3.Stop();
#endif
}

} // bfly
} // dbf

#endif // ifndef DBF_BFLY_SOURCE_WEIGHT_RECURSION_HPP
