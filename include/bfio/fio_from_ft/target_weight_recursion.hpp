/*
   ButterflyFIO: a distributed-memory fast algorithm for applying FIOs.
   Copyright (C) 2010-2011 Jack Poulson <jack.poulson@gmail.com>
 
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
#ifndef BFIO_FIO_FROM_FT_TARGET_WEIGHT_RECURSION_HPP
#define BFIO_FIO_FROM_FT_TARGET_WEIGHT_RECURSION_HPP 1

#include <cstddef>
#include <cstring>
#include <vector>

#include "bfio/structures/array.hpp"
#include "bfio/structures/plan.hpp"
#include "bfio/structures/weight_grid.hpp"
#include "bfio/structures/weight_grid_list.hpp"

#include "bfio/functors/phase.hpp"

#include "bfio/tools/blas.hpp"
#include "bfio/tools/special_functions.hpp"

#include "bfio/fio_from_ft/context.hpp"

namespace bfio {
namespace fio_from_ft {

// 1d specialization
template<typename R,std::size_t q>
void
TargetWeightRecursion
( const fio_from_ft::Context<R,1,q>& context,
  const Plan<1>& plan,
  const Phase<R,1>& phase,
  const std::size_t level,
  const std::size_t ARelativeToAp,
  const Array<R,1>& x0A,
  const Array<R,1>& x0Ap,
  const Array<R,1>& p0B,
  const Array<R,1>& wA,
  const Array<R,1>& wB,
  const std::size_t parentInteractionOffset,
  const WeightGridList<R,1,q>& oldWeightGridList,
        WeightGrid<R,1,q>& weightGrid )
{
    std::memset( weightGrid.Buffer(), 0, 2*q*sizeof(R) );

    const std::size_t log2NumMergingProcesses = 
        plan.GetLog2NumMergingProcesses( level );
    const std::vector<R>& leftMap = context.GetLeftChebyshevMap();
    const std::vector<R>& rightMap = context.GetRightChebyshevMap();

    std::vector<R> phiResults;
    std::vector<R> sinResults;
    std::vector<R> cosResults;
    std::vector< Array<R,1> > pPoint( 1 );
    std::vector< Array<R,1> > xPoints( q );
    const std::vector< Array<R,1> >& chebyshevGrid = context.GetChebyshevGrid();
    for( std::size_t cLocal=0; 
         cLocal<(1u<<(1-log2NumMergingProcesses)); 
         ++cLocal )
    {
        //--------------------------------------------------------------------//
        // Step 1                                                             //
        //--------------------------------------------------------------------//
        const std::size_t interactionIndex = parentInteractionOffset + cLocal;
        const std::size_t c = plan.LocalToClusterSourceIndex( level, cLocal );

        pPoint[0][0] = p0B[0] + ( c&1 ? wB[0]/4 : -wB[0]/4 );
        {
            R* RESTRICT xPointsBuffer = &xPoints[0][0];
            const R* RESTRICT wABuffer = &wA[0];
            const R* RESTRICT x0ApBuffer = &x0Ap[0];
            const R* RESTRICT chebyshevBuffer = &chebyshevGrid[0][0];
            for( std::size_t tPrime=0; tPrime<q; ++tPrime )
                xPointsBuffer[tPrime] = 
                    x0ApBuffer[0] + 2*wABuffer[0]*chebyshevBuffer[tPrime];
        }
        phase.BatchEvaluate( xPoints, pPoint, phiResults );
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
            for( std::size_t tPrime=0; tPrime<q; ++tPrime )
            {
                const R realPhase = cosBuffer[tPrime];
                const R imagPhase = -sinBuffer[tPrime];
                const R realWeight = oldRealBuffer[tPrime];
                const R imagWeight = oldImagBuffer[tPrime];
                scaledRealBuffer[tPrime] = 
                    realPhase*realWeight - imagPhase*imagWeight;
                scaledImagBuffer[tPrime] = 
                    imagPhase*realWeight + realPhase*imagWeight;
            }
        }

        //--------------------------------------------------------------------//
        // Step 2                                                             //
        //--------------------------------------------------------------------//
        // TODO: Create a preprocessor flag for switching to two Gemv's, as 
        //       Gemm is probably not optimized for only 2 right-hand sides.
        WeightGrid<R,1,q> expandedWeightGrid;
        {
            const R* mapBuffer = 
                ( ARelativeToAp&1 ? &rightMap[0] : &leftMap[0] );
            Gemm
            ( 'T', 'N', q, 2, q,
              (R)1, mapBuffer,                   q,
                    scaledWeightGrid.Buffer(),   q,
              (R)0, expandedWeightGrid.Buffer(), q );
        }

        //--------------------------------------------------------------------//
        // Step 3                                                             //
        //--------------------------------------------------------------------//
        {
            R* RESTRICT xPointsBuffer = &xPoints[0][0];
            const R* RESTRICT wABuffer = &wA[0];
            const R* RESTRICT x0ABuffer = &x0A[0];
            const R* RESTRICT chebyshevBuffer = &chebyshevGrid[0][0];
            for( std::size_t t=0; t<q; ++t )
                xPointsBuffer[t] = 
                    x0ABuffer[0] + wABuffer[0]*chebyshevBuffer[t];
        }
        phase.BatchEvaluate( xPoints, pPoint, phiResults );
        SinCosBatch( phiResults, sinResults, cosResults );
        {
            R* RESTRICT realBuffer = weightGrid.RealBuffer();
            R* RESTRICT imagBuffer = weightGrid.ImagBuffer();
            const R* RESTRICT cosBuffer = &cosResults[0];
            const R* RESTRICT sinBuffer = &sinResults[0];
            const R* RESTRICT expandedRealBuffer = 
                expandedWeightGrid.RealBuffer();
            const R* RESTRICT expandedImagBuffer = 
                expandedWeightGrid.ImagBuffer();
            for( std::size_t t=0; t<q; ++t )
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

// 2d specialization
template<typename R,std::size_t q>
void
TargetWeightRecursion
( const fio_from_ft::Context<R,2,q>& context,
  const Plan<2>& plan,
  const Phase<R,2>& phase,
  const std::size_t level,
  const std::size_t ARelativeToAp,
  const Array<R,2>& x0A,
  const Array<R,2>& x0Ap,
  const Array<R,2>& p0B,
  const Array<R,2>& wA,
  const Array<R,2>& wB,
  const std::size_t parentInteractionOffset,
  const WeightGridList<R,2,q>& oldWeightGridList,
        WeightGrid<R,2,q>& weightGrid )
{
    std::memset( weightGrid.Buffer(), 0, 2*q*q*sizeof(R) );

    const std::size_t log2NumMergingProcesses = 
        plan.GetLog2NumMergingProcesses( level );
    const std::vector<R>& leftMap = context.GetLeftChebyshevMap();
    const std::vector<R>& rightMap = context.GetRightChebyshevMap();

    std::vector<R> phiResults;
    std::vector<R> sinResults;
    std::vector<R> cosResults;
    std::vector< Array<R,2> > pPoint( 1 );
    std::vector< Array<R,2> > xPoints( q*q );
    const std::vector< Array<R,2> >& chebyshevGrid = context.GetChebyshevGrid();
    for( std::size_t cLocal=0; 
         cLocal<(1u<<(2-log2NumMergingProcesses)); 
         ++cLocal )
    {
        //--------------------------------------------------------------------//
        // Step 1                                                             //
        //--------------------------------------------------------------------//
        const std::size_t interactionIndex = parentInteractionOffset + cLocal;
        const std::size_t c = plan.LocalToClusterSourceIndex( level, cLocal );

        for( std::size_t j=0; j<2; ++j )
            pPoint[0][j] = p0B[j] + ( (c>>j)&1 ? wB[j]/4 : -wB[j]/4 );
        {
            R* RESTRICT xPointsBuffer = &xPoints[0][0];
            const R* RESTRICT wABuffer = &wA[0];
            const R* RESTRICT x0ApBuffer = &x0Ap[0];
            const R* RESTRICT chebyshevBuffer = &chebyshevGrid[0][0];
            for( std::size_t tPrime=0; tPrime<q*q; ++tPrime )
                for( std::size_t j=0; j<2; ++j )
                    xPointsBuffer[tPrime*2+j] = 
                        x0ApBuffer[j] + 
                        2*wABuffer[j]*chebyshevBuffer[tPrime*2+j];
        }
        phase.BatchEvaluate( xPoints, pPoint, phiResults );
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
            for( std::size_t tPrime=0; tPrime<q*q; ++tPrime )
            {
                const R realPhase = cosBuffer[tPrime];
                const R imagPhase = -sinBuffer[tPrime];
                const R realWeight = oldRealBuffer[tPrime];
                const R imagWeight = oldImagBuffer[tPrime];
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
        // and imaginary weights at once
        WeightGrid<R,2,q> tempWeightGrid;
        {
            const R* mapBuffer = 
                ( ARelativeToAp&1 ? &rightMap[0] : &leftMap[0] );
            Gemm
            ( 'T', 'N', q, 2*q, q,
              (R)1, mapBuffer,                 q,
                    scaledWeightGrid.Buffer(), q,
              (R)0, tempWeightGrid.Buffer(), q );
        }

        // Interpolate over the second dimension. We can get away with applying
        // our maps with a gemm on the real and imag parts.
        WeightGrid<R,2,q> expandedWeightGrid;
        {
            const R* mapBuffer = 
                ( (ARelativeToAp>>1)&1 ? &rightMap[0] : &leftMap[0] );
            Gemm
            ( 'N', 'N', q, q, q,
              (R)1, tempWeightGrid.RealBuffer(),     q,
                    mapBuffer,                       q,
              (R)0, expandedWeightGrid.RealBuffer(), q );
            Gemm
            ( 'N', 'N', q, q, q,
              (R)1, tempWeightGrid.ImagBuffer(),     q,
                    mapBuffer,                       q,
              (R)0, expandedWeightGrid.ImagBuffer(), q );
        }

        //--------------------------------------------------------------------//
        // Step 3                                                             //
        //--------------------------------------------------------------------//
        {
            R* RESTRICT xPointsBuffer = &xPoints[0][0];
            const R* RESTRICT wABuffer = &wA[0];
            const R* RESTRICT x0ABuffer = &x0A[0];
            const R* RESTRICT chebyshevBuffer = &chebyshevGrid[0][0];
            for( std::size_t t=0; t<q*q; ++t )
                for( std::size_t j=0; j<2; ++j )
                    xPointsBuffer[t*2+j] = 
                        x0ABuffer[j] + wABuffer[j]*chebyshevBuffer[t*2+j];
        }
        phase.BatchEvaluate( xPoints, pPoint, phiResults );
        SinCosBatch( phiResults, sinResults, cosResults );
        {
            R* RESTRICT realBuffer = weightGrid.RealBuffer();
            R* RESTRICT imagBuffer = weightGrid.ImagBuffer();
            const R* RESTRICT cosBuffer = &cosResults[0];
            const R* RESTRICT sinBuffer = &sinResults[0];
            const R* RESTRICT expandedRealBuffer = 
                expandedWeightGrid.RealBuffer();
            const R* RESTRICT expandedImagBuffer = 
                expandedWeightGrid.ImagBuffer();
            for( std::size_t t=0; t<q*q; ++t )
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

// Fallback for 3d and above
template<typename R,std::size_t d,std::size_t q>
void
TargetWeightRecursion
( const fio_from_ft::Context<R,d,q>& context,
  const Plan<d>& plan,
  const Phase<R,d>& phase,
  const std::size_t level,
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
    std::memset( weightGrid.Buffer(), 0, 2*q_to_d*sizeof(R) );

    const std::size_t log2NumMergingProcesses = 
        plan.GetLog2NumMergingProcesses( level );
    const std::vector<R>& leftMap = context.GetLeftChebyshevMap();
    const std::vector<R>& rightMap = context.GetRightChebyshevMap();

    std::vector<R> phiResults;
    std::vector<R> sinResults;
    std::vector<R> cosResults;
    std::vector< Array<R,d> > pPoint( 1 );
    std::vector< Array<R,d> > xPoints( q_to_d );
    const std::vector< Array<R,d> >& chebyshevGrid = context.GetChebyshevGrid();
    for( std::size_t cLocal=0; 
         cLocal<(1u<<(d-log2NumMergingProcesses)); 
         ++cLocal )
    {
        //--------------------------------------------------------------------//
        // Step 1                                                             //
        //--------------------------------------------------------------------//
        const std::size_t interactionIndex = parentInteractionOffset + cLocal;
        const std::size_t c = plan.LocalToClusterSourceIndex( level, cLocal );

        for( std::size_t j=0; j<d; ++j )
            pPoint[0][j] = p0B[j] + ( (c>>j)&1 ? wB[j]/4 : -wB[j]/4 );
        {
            R* RESTRICT xPointsBuffer = &xPoints[0][0];
            const R* RESTRICT wABuffer = &wA[0];
            const R* RESTRICT x0ApBuffer = &x0Ap[0];
            const R* RESTRICT chebyshevBuffer = &chebyshevGrid[0][0];
            for( std::size_t tPrime=0; tPrime<q_to_d; ++tPrime )
                for( std::size_t j=0; j<d; ++j )
                    xPointsBuffer[tPrime*d+j] = 
                        x0ApBuffer[j] + 
                        2*wABuffer[j]*chebyshevBuffer[tPrime*d+j];
        }
        phase.BatchEvaluate( xPoints, pPoint, phiResults );
        SinCosBatch( phiResults, sinResults, cosResults );

        WeightGrid<R,d,q> scaledWeightGrid;
        {
            R* RESTRICT scaledRealBuffer = scaledWeightGrid.RealBuffer();
            R* RESTRICT scaledImagBuffer = scaledWeightGrid.ImagBuffer();
            const R* RESTRICT cosBuffer = &cosResults[0];
            const R* RESTRICT sinBuffer = &sinResults[0];
            const R* RESTRICT oldRealBuffer = 
                oldWeightGridList[interactionIndex].RealBuffer();
            const R* RESTRICT oldImagBuffer = 
                oldWeightGridList[interactionIndex].ImagBuffer();
            for( std::size_t tPrime=0; tPrime<q_to_d; ++tPrime )
            {
                const R realPhase = cosBuffer[tPrime];
                const R imagPhase = -sinBuffer[tPrime];
                const R realWeight = oldRealBuffer[tPrime];
                const R imagWeight = oldImagBuffer[tPrime];
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
        // and imaginary weights at once
        WeightGrid<R,d,q> tempWeightGrid;
        {
            const R* mapBuffer = 
                ( ARelativeToAp&1 ? &rightMap[0] : &leftMap[0] );
            Gemm
            ( 'T', 'N', q, 2*Pow<q,d-1>::val, q,
              (R)1, mapBuffer,                 q,
                    scaledWeightGrid.Buffer(), q,
              (R)0,   tempWeightGrid.Buffer(), q );
        }

        // Interpolate over the second dimension. We can do so by using
        // q^(d-2) gemms on right-hand sides of size q x q for both the real
        // and imaginary buffers.
        {
            R* realWriteBuffer = scaledWeightGrid.RealBuffer();
            R* imagWriteBuffer = scaledWeightGrid.ImagBuffer();
            const R* realReadBuffer = tempWeightGrid.RealBuffer();
            const R* imagReadBuffer = tempWeightGrid.ImagBuffer();
            const R* mapBuffer = 
                ( (ARelativeToAp>>1)&1 ? &rightMap[0] : &leftMap[0] );
            for( std::size_t w=0; w<Pow<q,d-2>::val; ++w )
            {
                Gemm
                ( 'N', 'N', q, q, q,
                  (R)1, &realReadBuffer[w*q*q],  q,
                        mapBuffer,               q,
                  (R)0, &realWriteBuffer[w*q*q], q );
            }
            for( std::size_t w=0; w<Pow<q,d-2>::val; ++w )
            {
                Gemm
                ( 'N', 'N', q, q, q,
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
        std::size_t q_to_j = q*q;
        WeightGrid<R,d,q> expandedWeightGrid;
        for( std::size_t j=2; j<d; ++j )
        {
            const std::size_t stride = q_to_j;

            R* realWriteBuffer = 
                ( j==d-1 ? expandedWeightGrid.RealBuffer()
                         : ( j&1 ? scaledWeightGrid.RealBuffer()
                                 : tempWeightGrid.RealBuffer() ) );
            R* imagWriteBuffer = 
                ( j==d-1 ? expandedWeightGrid.ImagBuffer()
                         : ( j&1 ? scaledWeightGrid.RealBuffer()
                                 : tempWeightGrid.RealBuffer() ) );
            const R* realReadBuffer = 
                ( j&1 ? tempWeightGrid.RealBuffer()
                      : scaledWeightGrid.RealBuffer() );
            const R* imagReadBuffer = 
                ( j&1 ? tempWeightGrid.ImagBuffer()
                      : scaledWeightGrid.ImagBuffer() );
            const R* RESTRICT mapBuffer = 
                ( (ARelativeToAp>>j)&1 ? &rightMap[0] : &leftMap[0] );

            std::memset( realWriteBuffer, 0, q_to_d*sizeof(R) );
            std::memset( imagWriteBuffer, 0, q_to_d*sizeof(R) );
            for( std::size_t p=0; p<q_to_d/(q_to_j*q); ++p )
            {
                const std::size_t offset = p*(q_to_j*q);
                R* RESTRICT offsetRealWriteBuffer = &realWriteBuffer[offset];
                R* RESTRICT offsetImagWriteBuffer = &imagWriteBuffer[offset];
                const R* RESTRICT offsetRealReadBuffer = &realReadBuffer[offset];
                const R* RESTRICT offsetImagReadBuffer = &imagReadBuffer[offset];
                for( std::size_t w=0; w<q_to_j; ++w )
                {
                    for( std::size_t t=0; t<q; ++t )
                    {
                        for( std::size_t tPrime=0; tPrime<q; ++tPrime )
                        {
                            offsetRealWriteBuffer[w+t*stride] +=
                                mapBuffer[tPrime+t*q] *
                                offsetRealReadBuffer[w+tPrime*stride];
                        }
                    }
                    for( std::size_t t=0; t<q; ++t )
                    {
                        for( std::size_t tPrime=0; tPrime<q; ++tPrime )
                        {
                            offsetImagWriteBuffer[w+t*stride] +=
                                mapBuffer[tPrime+t*q] *
                                offsetImagReadBuffer[w+tPrime*stride];
                        }
                    }
                }
            }
            q_to_j *= q;
        }

        //--------------------------------------------------------------------//
        // Step 3                                                             //
        //--------------------------------------------------------------------//
        {
            R* RESTRICT xPointsBuffer = &xPoints[0][0];
            const R* RESTRICT wABuffer = &wA[0];
            const R* RESTRICT x0ABuffer = &x0A[0];
            const R* RESTRICT chebyshevBuffer = &chebyshevGrid[0][0];
            for( std::size_t t=0; t<q_to_d; ++t )
                for( std::size_t j=0; j<d; ++j )
                    xPointsBuffer[t*d+j] = 
                        x0ABuffer[j] + wABuffer[j]*chebyshevBuffer[t*d+j];
        }
        phase.BatchEvaluate( xPoints, pPoint, phiResults );
        SinCosBatch( phiResults, sinResults, cosResults );
        {
            R* RESTRICT realBuffer = weightGrid.RealBuffer();
            R* RESTRICT imagBuffer = weightGrid.ImagBuffer();
            const R* RESTRICT cosBuffer = &cosResults[0];
            const R* RESTRICT sinBuffer = &sinResults[0];
            const R* RESTRICT expandedRealBuffer = 
                expandedWeightGrid.RealBuffer();
            const R* RESTRICT expandedImagBuffer = 
                expandedWeightGrid.ImagBuffer();
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

} // fio_from_ft
} // bfio

#endif // BFIO_FIO_FROM_FT_TARGET_WEIGHT_RECURSION_HPP

