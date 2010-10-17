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
#ifndef BFIO_INTERPOLATIVE_NUFT_FORM_CHECK_POTENTIALS_HPP
#define BFIO_INTERPOLATIVE_NUFT_FORM_CHECK_POTENTIALS_HPP 1

#include <cstddef>
#include <cstring>
#include <vector>

#include "bfio/constants.hpp"

#include "bfio/structures/array.hpp"
#include "bfio/structures/weight_grid.hpp"
#include "bfio/structures/weight_grid_list.hpp"

#include "bfio/tools/special_functions.hpp"

#include "bfio/interpolative_nuft/context.hpp"

namespace bfio {
namespace interpolative_nuft {

// 1d specialization
template<typename R,std::size_t q>
void
FormCheckPotentials
( const interpolative_nuft::Context<R,1,q>& context,
  const Plan<1>& plan,
  const std::size_t level,
  const Array<R,1>& x0A,
  const Array<R,1>& p0B,
  const Array<R,1>& wA,
  const Array<R,1>& wB,
  const std::size_t parentInteractionOffset,
  const WeightGridList<R,1,q>& oldWeightGridList,
        WeightGrid<R,1,q>& weightGrid )
{
    const std::size_t d = 1;
    std::memset( weightGrid.Buffer(), 0, 2*q*sizeof(R) );

    const std::size_t log2NumMergingProcesses = 
        plan.GetLog2NumMergingProcesses( level );
    const std::vector<R>& chebyshevNodes = context.GetChebyshevNodes();

    std::vector<R> realTempWeights0( q );
    std::vector<R> imagTempWeights0( q );
    std::vector<R> realTempWeights1( q );
    std::vector<R> imagTempWeights1( q );
    std::vector<R> scalingArguments( q );
    std::vector<R> realScalings( q );
    std::vector<R> imagScalings( q );
    for( std::size_t cLocal=0;
         cLocal<(1u<<(1-log2NumMergingProcesses));
         ++cLocal )
    {
        const std::size_t interactionIndex = parentInteractionOffset + cLocal;
        const std::size_t c = plan.LocalToClusterSourceIndex( level, cLocal );
        const WeightGrid<R,d,q>& oldWeightGrid = 
            oldWeightGridList[interactionIndex];

        // Find the center of child c
        Array<R,d> p0Bc;
        p0Bc[0] = p0B[0] + ( c&1 ? wB[0]/4 : -wB[0]/4 );

        // Prescaling
        for( std::size_t t=0; t<q; ++t )
            scalingArguments[t] = TwoPi*x0A[0]*chebyshevNodes[t]*wB[0]/2;
        SinCosBatch( scalingArguments, imagScalings, realScalings );
        {
            R* realWriteBuffer = &realTempWeights0[0];
            R* imagWriteBuffer = &imagTempWeights0[0];
            const R* realReadBuffer = oldWeightGrid.RealBuffer();
            const R* imagReadBuffer = oldWeightGrid.ImagBuffer();
            const R* realScalingBuffer = &realScalings[0];
            const R* imagScalingBuffer = &imagScalings[0];
            for( std::size_t t=0; t<q; ++t )
            {
                const R realWeight = realReadBuffer[t];
                const R imagWeight = imagReadBuffer[t];
                const R realScaling = realScalingBuffer[t];
                const R imagScaling = imagScalingBuffer[t];
                realWriteBuffer[t] =
                    realWeight*realScaling - imagWeight*imagScaling;
                imagWriteBuffer[t] = 
                    imagWeight*realScaling + realWeight*imagScaling;
            }
        }
        // Apply forward map
        {
            const std::vector<R>& realForwardMap = 
                context.GetRealForwardMap( 0 );
            const std::vector<R>& imagForwardMap = 
                context.GetImagForwardMap( 0 );
            // Form the real part
            Gemv
            ( 'N', q, q,
              (R)1, &realForwardMap[0], q,
                    &realTempWeights0[0], 1,
              (R)0, &realTempWeights1[0], 1 );
            Gemv
            ( 'N', q, q,
              (R)-1, &imagForwardMap[0], q,
                     &imagTempWeights0[0], 1,
              (R)+1, &realTempWeights1[0], 1 );
            // Form the imaginary part
            Gemv
            ( 'N', q, q,
              (R)1, &realForwardMap[0], q,
                    &imagTempWeights0[0], 1,
              (R)0, &imagTempWeights1[0], 1 );
            Gemv
            ( 'N', q, q,
              (R)1, &imagForwardMap[0], q,
                    &realTempWeights0[0], 1,
              (R)1, &imagTempWeights1[0], 1 );
        }
        // Postscaling
        for( std::size_t t=0; t<q; ++t )
            scalingArguments[t] = 
                -TwoPi*(x0A[0]+chebyshevNodes[t]*wA[0])*p0Bc[0];
        SinCosBatch( scalingArguments, imagScalings, realScalings );
        {
            R* realWriteBuffer = weightGrid.RealBuffer();
            R* imagWriteBuffer = weightGrid.ImagBuffer();
            const R* realReadBuffer = &realTempWeights1[0];
            const R* imagReadBuffer = &imagTempWeights1[0];
            const R* realScalingBuffer = &realScalings[0];
            const R* imagScalingBuffer = &imagScalings[0];
            for( std::size_t t=0; t<q; ++t )
            {
                const R realWeight = realReadBuffer[t];
                const R imagWeight = imagReadBuffer[t];
                const R realScaling = realScalingBuffer[t];
                const R imagScaling = imagScalingBuffer[t];
                realWriteBuffer[t] += 
                    realWeight*realScaling - imagWeight*imagScaling;
                imagWriteBuffer[t] += 
                    imagWeight*realScaling + realWeight*imagScaling;
            }
        }
    }
}

// 2d specialization
template<typename R,std::size_t q>
void
FormCheckPotentials
( const interpolative_nuft::Context<R,2,q>& context,
  const Plan<2>& plan,
  const std::size_t level,
  const Array<R,2>& x0A,
  const Array<R,2>& p0B,
  const Array<R,2>& wA,
  const Array<R,2>& wB,
  const std::size_t parentInteractionOffset,
  const WeightGridList<R,2,q>& oldWeightGridList,
        WeightGrid<R,2,q>& weightGrid )
{
    const std::size_t d = 2;
    const std::size_t q_to_d = q*q;
    std::memset( weightGrid.Buffer(), 0, 2*q_to_d*sizeof(R) );

    const std::size_t log2NumMergingProcesses = 
        plan.GetLog2NumMergingProcesses( level );
    const std::vector<R>& chebyshevNodes = context.GetChebyshevNodes();

    std::vector<R> realTempWeights0( q_to_d );
    std::vector<R> imagTempWeights0( q_to_d );
    std::vector<R> realTempWeights1( q_to_d );
    std::vector<R> imagTempWeights1( q_to_d );
    std::vector<R> scalingArguments( q );
    std::vector<R> realScalings( q );
    std::vector<R> imagScalings( q );
    for( std::size_t cLocal=0; 
         cLocal<(1u<<(2-log2NumMergingProcesses));
         ++cLocal )
    {
        const std::size_t interactionIndex = parentInteractionOffset + cLocal;
        const std::size_t c = plan.LocalToClusterSourceIndex( level, cLocal );
        const WeightGrid<R,d,q>& oldWeightGrid = 
            oldWeightGridList[interactionIndex];

        // Find the center of child c
        Array<R,d> p0Bc;
        for( std::size_t j=0; j<d; ++j )
            p0Bc[j] = p0B[j] + ( (c>>j)&1 ? wB[j]/4 : -wB[j]/4 );

        //--------------------------------------------------------------------//
        // Transform the first dimension                                      //
        //--------------------------------------------------------------------//
        // Prescale
        for( std::size_t t=0; t<q; ++t )
            scalingArguments[t] = TwoPi*x0A[0]*chebyshevNodes[t]*wB[0]/2;
        SinCosBatch( scalingArguments, imagScalings, realScalings );
        {
            R* realWriteBuffer = &realTempWeights0[0];
            R* imagWriteBuffer = &imagTempWeights0[0];
            const R* realReadBuffer = oldWeightGrid.RealBuffer();
            const R* imagReadBuffer = oldWeightGrid.ImagBuffer();
            const R* realScalingBuffer = &realScalings[0];
            const R* imagScalingBuffer = &imagScalings[0];
            for( std::size_t t=0; t<q; ++t )
            {
                for( std::size_t tPrime=0; tPrime<q; ++tPrime )
                {
                    const R realWeight = realReadBuffer[t*q+tPrime];
                    const R imagWeight = imagReadBuffer[t*q+tPrime];
                    const R realScaling = realScalingBuffer[tPrime];
                    const R imagScaling = imagScalingBuffer[tPrime];
                    realWriteBuffer[t*q+tPrime] = 
                        realWeight*realScaling - imagWeight*imagScaling;
                    imagWriteBuffer[t*q+tPrime] = 
                        imagWeight*realScaling + realWeight*imagScaling;
                }
            }
        }
        // Apply forward map
        {
            const std::vector<R>& realForwardMap = 
                context.GetRealForwardMap( 0 );
            const std::vector<R>& imagForwardMap = 
                context.GetImagForwardMap( 0 );
            // Form the real part
            Gemm
            ( 'N', 'N', q, q, q,
              (R)1, &realForwardMap[0], q,
                    &realTempWeights0[0], q,
              (R)0, &realTempWeights1[0], q );
            Gemm
            ( 'N', 'N', q, q, q,
              (R)-1, &imagForwardMap[0], q,
                     &imagTempWeights0[0], q,
              (R)+1, &realTempWeights1[0], q );
            // Form the imaginary part
            Gemm
            ( 'N', 'N', q, q, q,
              (R)1, &realForwardMap[0], q,
                    &imagTempWeights0[0], q,
              (R)0, &imagTempWeights1[0], q );
            Gemm
            ( 'N', 'N', q, q, q,
              (R)1, &imagForwardMap[0], q,
                    &realTempWeights0[0], q,
              (R)1, &imagTempWeights1[0], q );
        }
        // Postscale
        for( std::size_t t=0; t<q; ++t )
            scalingArguments[t] = 
                TwoPi*(x0A[0]+chebyshevNodes[t]*wA[0])*p0Bc[0];
        SinCosBatch( scalingArguments, imagScalings, realScalings );
        {
            R* realBuffer = &realTempWeights1[0];
            R* imagBuffer = &imagTempWeights1[0];
            const R* realScalingBuffer = &realScalings[0];
            const R* imagScalingBuffer = &imagScalings[0];
            for( std::size_t t=0; t<q; ++t )
            {
                for( std::size_t tPrime=0; tPrime<q; ++tPrime )
                {
                    const R realWeight = realBuffer[t*q+tPrime];
                    const R imagWeight = imagBuffer[t*q+tPrime];
                    const R realScaling = realScalingBuffer[tPrime];
                    const R imagScaling = imagScalingBuffer[tPrime];
                    realBuffer[t*q+tPrime] = 
                        realWeight*realScaling - imagWeight*imagScaling;
                    imagBuffer[t*q+tPrime] = 
                        imagWeight*realScaling + realWeight*imagScaling;
                }
            }
        }

        //--------------------------------------------------------------------//
        // Transform the second dimension                                     //
        //--------------------------------------------------------------------//
        // Prescale
        for( std::size_t t=0; t<q; ++t )
            scalingArguments[t] = TwoPi*x0A[1]*chebyshevNodes[t]*wB[1]/2;
        SinCosBatch( scalingArguments, imagScalings, realScalings );
        {
            R* realBuffer = &realTempWeights1[0];
            R* imagBuffer = &imagTempWeights1[0];
            const R* realScalingBuffer = &realScalings[0];
            const R* imagScalingBuffer = &imagScalings[0];
            for( std::size_t w=0; w<q; ++w )
            {
                for( std::size_t t=0; t<q; ++t )
                {
                    const R realWeight = realBuffer[w+t*q];
                    const R imagWeight = imagBuffer[w+t*q];
                    const R realScaling = realScalingBuffer[t];
                    const R imagScaling = imagScalingBuffer[t];
                    realBuffer[w+t*q] = 
                        realWeight*realScaling - imagWeight*imagScaling;
                    imagBuffer[w+t*q] = 
                        imagWeight*realScaling + realWeight*imagScaling;
                }
            }
        }
        // Apply forward map
        {
            const std::vector<R>& realForwardMap = 
                context.GetRealForwardMap( 1 );
            const std::vector<R>& imagForwardMap = 
                context.GetImagForwardMap( 1 );
            // Form the real part
            Gemm
            ( 'N', 'T', q, q, q,
              (R)1, &realTempWeights1[0], q,
                    &realForwardMap[0], q,
              (R)0, &realTempWeights0[0], q );
            Gemm
            ( 'N', 'T', q, q, q,
              (R)-1, &imagTempWeights1[0], q,
                     &imagForwardMap[0], q,
              (R)+1, &realTempWeights0[0], q );
            // Form the imaginary part
            Gemm
            ( 'N', 'T', q, q, q,
              (R)1, &imagTempWeights1[0], q,
                    &realForwardMap[0], q,
              (R)0, &imagTempWeights0[0], q );
            Gemm
            ( 'N', 'T', q, q, q,
              (R)1, &realTempWeights1[0], q,
                    &imagForwardMap[0], q,
              (R)1, &imagTempWeights0[0], q );
        }
        // Postscale
        for( std::size_t t=0; t<q; ++t )
            scalingArguments[t] = 
                TwoPi*(x0A[1]+chebyshevNodes[t]*wA[1])*p0Bc[1];
        SinCosBatch( scalingArguments, imagScalings, realScalings );
        {
            R* realWriteBuffer = weightGrid.RealBuffer();
            R* imagWriteBuffer = weightGrid.ImagBuffer();
            const R* realReadBuffer = &realTempWeights0[0];
            const R* imagReadBuffer = &imagTempWeights0[0];
            const R* realScalingBuffer = &realScalings[0];
            const R* imagScalingBuffer = &imagScalings[0];
            for( std::size_t w=0; w<q; ++w )
            {
                for( std::size_t t=0; t<q; ++t )
                {
                    const R realWeight = realReadBuffer[w+t*q];
                    const R imagWeight = imagReadBuffer[w+t*q];
                    const R realScaling = realScalingBuffer[t];
                    const R imagScaling = imagScalingBuffer[t];
                    realWriteBuffer[w+t*q] += 
                        realWeight*realScaling - imagWeight*imagScaling;
                    imagWriteBuffer[w+t*q] += 
                        imagWeight*realScaling + realWeight*imagScaling;
                }
            }
        }
    }
}

// Fallback for 3d and above
template<typename R,std::size_t d,std::size_t q>
void
FormCheckPotentials
( const interpolative_nuft::Context<R,d,q>& context,
  const Plan<d>& plan,
  const std::size_t level,
  const Array<R,d>& x0A,
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
    const std::vector<R>& chebyshevNodes = context.GetChebyshevNodes();

    std::vector<R> realTempWeights0( q_to_d );
    std::vector<R> imagTempWeights0( q_to_d );
    std::vector<R> realTempWeights1( q_to_d );
    std::vector<R> imagTempWeights1( q_to_d );
    std::vector<R> scalingArguments( q );
    std::vector<R> realScalings( q );
    std::vector<R> imagScalings( q );
    for( std::size_t cLocal=0;
         cLocal<(1u<<(d-log2NumMergingProcesses));
         ++cLocal )
    {
        const std::size_t interactionIndex = parentInteractionOffset + cLocal;
        const std::size_t c = plan.LocalToClusterSourceIndex( level, cLocal );
        const WeightGrid<R,d,q>& oldWeightGrid = 
            oldWeightGridList[interactionIndex];

        // Find the center of child c
        Array<R,d> p0Bc;
        for( std::size_t j=0; j<d; ++j )
            p0Bc[j] = p0B[j] + ( (c>>j)&1 ? wB[j]/4 : -wB[j]/4 );

        //--------------------------------------------------------------------//
        // Transform the first dimension                                      //
        //--------------------------------------------------------------------//
        // Prescale
        for( std::size_t t=0; t<q; ++t )
            scalingArguments[t] = TwoPi*x0A[0]*chebyshevNodes[t]*wB[0]/2;
        SinCosBatch( scalingArguments, imagScalings, realScalings );
        {
            R* realWriteBuffer = &realTempWeights0[0];
            R* imagWriteBuffer = &imagTempWeights0[0];
            const R* realReadBuffer = oldWeightGrid.RealBuffer();
            const R* imagReadBuffer = oldWeightGrid.ImagBuffer();
            const R* realScalingBuffer = &realScalings[0];
            const R* imagScalingBuffer = &imagScalings[0];
            for( std::size_t t=0; t<Pow<q,d-1>::val; ++t )
            {
                for( std::size_t tPrime=0; tPrime<q; ++tPrime )
                {
                    const R realWeight = realReadBuffer[t*q+tPrime];
                    const R imagWeight = imagReadBuffer[t*q+tPrime];
                    const R realScaling = realScalingBuffer[tPrime];
                    const R imagScaling = imagScalingBuffer[tPrime];
                    realWriteBuffer[t*q+tPrime] = 
                        realWeight*realScaling - imagWeight*imagScaling;
                    imagWriteBuffer[t*q+tPrime] = 
                        imagWeight*realScaling + realWeight*imagScaling;
                }
            }
        }
        // Apply forward map
        {
            const std::vector<R>& realForwardMap = 
                context.GetRealForwardMap( 0 );
            const std::vector<R>& imagForwardMap = 
                context.GetImagForwardMap( 0 );
            // Form the real part
            Gemm
            ( 'N', 'N', q, Pow<q,d-1>::val, q,
              (R)1, &realForwardMap[0], q,
                    &realTempWeights0[0], q,
              (R)0, &realTempWeights1[0], q );
            Gemm
            ( 'N', 'N', q, Pow<q,d-1>::val, q,
              (R)-1, &imagForwardMap[0], q,
                     &imagTempWeights0[0], q,
              (R)+1, &realTempWeights1[0], q );
            // Form the imaginary part
            Gemm
            ( 'N', 'N', q, Pow<q,d-1>::val, q,
              (R)1, &realForwardMap[0], q,
                    &imagTempWeights0[0], q,
              (R)0, &imagTempWeights1[0], q );
            Gemm
            ( 'N', 'N', q, Pow<q,d-1>::val, q,
              (R)1, &imagForwardMap[0], q,
                    &realTempWeights0[0], q,
              (R)1, &imagTempWeights1[0], q );
        }
        // Postscale
        for( std::size_t t=0; t<q; ++t )
            scalingArguments[t] = 
                TwoPi*(x0A[0]+chebyshevNodes[t]*wA[0])*p0Bc[0];
        SinCosBatch( scalingArguments, imagScalings, realScalings );
        {
            R* realBuffer = &realTempWeights1[0];
            R* imagBuffer = &imagTempWeights1[0];
            const R* realScalingBuffer = &realScalings[0];
            const R* imagScalingBuffer = &imagScalings[0];
            for( std::size_t t=0; t<Pow<q,d-1>::val; ++t )
            {
                for( std::size_t tPrime=0; tPrime<q; ++tPrime )
                {
                    const R realWeight = realBuffer[t*q+tPrime];
                    const R imagWeight = imagBuffer[t*q+tPrime];
                    const R realScaling = realScalingBuffer[tPrime];
                    const R imagScaling = imagScalingBuffer[tPrime];
                    realBuffer[t*q+tPrime] = 
                        realWeight*realScaling - imagWeight*imagScaling;
                    imagBuffer[t*q+tPrime] = 
                        imagWeight*realScaling + realWeight*imagScaling;
                }
            }
        }

        //--------------------------------------------------------------------//
        // Transform the second dimension                                     //
        //--------------------------------------------------------------------//
        // Prescale
        for( std::size_t t=0; t<q; ++t )
            scalingArguments[t] = TwoPi*x0A[1]*chebyshevNodes[t]*wB[1]/2;
        SinCosBatch( scalingArguments, imagScalings, realScalings );
        for( std::size_t p=0; p<Pow<q,d-2>::val; ++p )
        {
            const std::size_t offset = p*q*q; 
            R* offsetRealBuffer = &realTempWeights1[offset];
            R* offsetImagBuffer = &imagTempWeights1[offset];
            const R* realScalingBuffer = &realScalings[0];
            const R* imagScalingBuffer = &imagScalings[0];
            for( std::size_t w=0; w<q; ++w )
            {
                for( std::size_t t=0; t<q; ++t )
                {
                    const R realWeight = offsetRealBuffer[w+t*q];
                    const R imagWeight = offsetImagBuffer[w+t*q];
                    const R realScaling = realScalingBuffer[t];
                    const R imagScaling = imagScalingBuffer[t];
                    offsetRealBuffer[w+t*q] = 
                        realWeight*realScaling - imagWeight*imagScaling;
                    offsetImagBuffer[w+t*q] = 
                        imagWeight*realScaling + realWeight*imagScaling;
                }
            }
        }
        // Apply forward map
        {
            const std::vector<R>& realForwardMap = 
                context.GetRealForwardMap( 1 );
            const std::vector<R>& imagForwardMap = 
                context.GetImagForwardMap( 1 );
            for( std::size_t w=0; w<Pow<q,d-2>::val; ++w )
            {
                // Form the real part
                Gemm
                ( 'N', 'T', q, q, q,
                  (R)1, &realTempWeights1[w*q*q], q,
                        &realForwardMap[0], q,
                  (R)0, &realTempWeights0[w*q*q], q );
                Gemm
                ( 'N', 'T', q, q, q,
                  (R)-1, &imagTempWeights1[w*q*q], q,
                         &imagForwardMap[0], q,
                  (R)+1, &realTempWeights0[w*q*q], q );
                // Form the imaginary part
                Gemm
                ( 'N', 'T', q, q, q,
                  (R)1, &imagTempWeights1[w*q*q], q,
                        &realForwardMap[0], q,
                  (R)0, &imagTempWeights0[w*q*q], q );
                Gemm
                ( 'N', 'T', q, q, q,
                  (R)1, &realTempWeights1[w*q*q], q,
                        &imagForwardMap[0], q,
                  (R)1, &imagTempWeights0[w*q*q], q );
            }
        }
        // Postscale
        for( std::size_t t=0; t<q; ++t )
            scalingArguments[t] =
                TwoPi*(x0A[1]+chebyshevNodes[t]*wA[1])*p0Bc[1];
        SinCosBatch( scalingArguments, imagScalings, realScalings );
        for( std::size_t p=0; p<Pow<q,d-2>::val; ++p )
        {
            const std::size_t offset = p*q*q;
            R* offsetRealBuffer = &realTempWeights0[offset];
            R* offsetImagBuffer = &imagTempWeights0[offset];
            const R* realScalingBuffer = &realScalings[0];
            const R* imagScalingBuffer = &imagScalings[0];
            for( std::size_t w=0; w<q; ++w )
            {
                for( std::size_t t=0; t<q; ++t )
                {
                    const R realWeight = offsetRealBuffer[w+t*q];
                    const R imagWeight = offsetImagBuffer[w+t*q];
                    const R realScaling = realScalingBuffer[t];
                    const R imagScaling = imagScalingBuffer[t];
                    offsetRealBuffer[w+t*q] =
                        realWeight*realScaling - imagWeight*imagScaling;
                    offsetImagBuffer[w+t*q] =
                        imagWeight*realScaling + realWeight*imagScaling;
                }
            }
        }
        
        //--------------------------------------------------------------------//
        // Transform the remaining dimension                                  //
        //--------------------------------------------------------------------//
        std::size_t q_to_j = q*q;
        for( std::size_t j=2; j<d; ++j )
        {
            const std::size_t stride = q_to_j;

            R* realWriteBuffer = 
                ( j==d-1 ? weightGrid.RealBuffer()
                         : ( j&1 ? &realTempWeights0[0] 
                                 : &realTempWeights1[0] ) );
            R* imagWriteBuffer = 
                ( j==d-1 ? weightGrid.ImagBuffer()
                         : ( j&1 ? &imagTempWeights0[0]
                                 : &imagTempWeights1[0] ) );
            R* realReadBuffer = 
                ( j&1 ? &realTempWeights1[0] : &realTempWeights0[0] );
            R* imagReadBuffer = 
                ( j&1 ? &imagTempWeights1[0] : &imagTempWeights0[0] );

            const std::vector<R>& realForwardMap = 
                context.GetRealForwardMap( j );
            const std::vector<R>& imagForwardMap = 
                context.GetImagForwardMap( j );
            const R* realForwardBuffer = &realForwardMap[0];
            const R* imagForwardBuffer = &imagForwardMap[0];

            if( j != d-1 )
            {
                std::memset( realWriteBuffer, 0, q_to_d*sizeof(R) );
                std::memset( imagWriteBuffer, 0, q_to_d*sizeof(R) );
            }

            // Prescale, apply the forward map, and then postscale
            std::vector<R> realTempWeightStrip( q );
            std::vector<R> imagTempWeightStrip( q );
            std::vector<R> realPrescalings( q );
            std::vector<R> imagPrescalings( q );
            std::vector<R> realPostscalings( q );
            std::vector<R> imagPostscalings( q );
            for( std::size_t t=0; t<q; ++t )
                scalingArguments[t] = TwoPi*x0A[j]*chebyshevNodes[t]*wB[j]/2;
            SinCosBatch
            ( scalingArguments, imagPrescalings, realPrescalings );
            for( std::size_t t=0; t<q; ++t )
                scalingArguments[t] = 
                    TwoPi*(x0A[j]+chebyshevNodes[t]*wA[j])*p0Bc[j];
            SinCosBatch
            ( scalingArguments, imagPostscalings, realPostscalings );
            for( std::size_t p=0; p<q_to_d/(q_to_j*q); ++p )        
            {
                const std::size_t offset = p*(q_to_j*q);
                R* offsetRealReadBuffer = &realReadBuffer[offset];
                R* offsetImagReadBuffer = &imagReadBuffer[offset];
                R* offsetRealWriteBuffer = &realWriteBuffer[offset];
                R* offsetImagWriteBuffer = &imagWriteBuffer[offset];
                const R* realPrescalingBuffer = &realPrescalings[0];
                const R* imagPrescalingBuffer = &imagPrescalings[0];
                const R* realPostscalingBuffer = &realPostscalings[0];
                const R* imagPostscalingBuffer = &imagPostscalings[0];
                for( std::size_t w=0; w<q_to_j; ++w )
                {
                    // Prescale
                    for( std::size_t t=0; t<q; ++t )
                    {
                        const R realWeight = offsetRealReadBuffer[w+t*stride];
                        const R imagWeight = offsetImagReadBuffer[w+t*stride];
                        const R realScaling = realPrescalingBuffer[t];
                        const R imagScaling = imagPrescalingBuffer[t];
                        realTempWeightStrip[t] = 
                            realWeight*realScaling - imagWeight*imagScaling;
                        imagTempWeightStrip[t] = 
                            imagWeight*realScaling + realWeight*imagScaling;
                    }
                    // Transform
                    for( std::size_t t=0; t<q; ++t )
                    {
                        offsetRealReadBuffer[w+t*stride] = 0;
                        offsetImagReadBuffer[w+t*stride] = 0;
                        for( std::size_t tPrime=0; tPrime<q; ++tPrime )
                        {
                            offsetRealReadBuffer[w+t*stride] +=
                                realForwardBuffer[t+tPrime*q] * 
                                realTempWeightStrip[tPrime];
                            offsetRealReadBuffer[w+t*stride] -=
                                imagForwardBuffer[t+tPrime*q] *
                                imagTempWeightStrip[tPrime];

                            offsetImagReadBuffer[w+t*stride] +=
                                imagForwardBuffer[t+tPrime*q] *
                                realTempWeightStrip[tPrime];
                            offsetImagReadBuffer[w+t*stride] +=
                                realForwardBuffer[t+tPrime*q] *
                                imagTempWeightStrip[tPrime];
                        }
                    }
                    // Postscale
                    for( std::size_t t=0; t<q; ++t )
                    {
                        const R realWeight = 
                            offsetRealReadBuffer[w+t*stride];
                        const R imagWeight = 
                            offsetImagReadBuffer[w+t*stride];
                        const R realScaling = realPostscalingBuffer[t];
                        const R imagScaling = imagPostscalingBuffer[t];
                        offsetRealWriteBuffer[w+t*stride] +=
                            realWeight*realScaling - imagWeight*imagScaling;
                        offsetImagWriteBuffer[w+t*stride] +=
                            imagWeight*realScaling + realWeight*imagScaling;
                    }
                }
            }
            q_to_j *= q;
        }
    }
}

} // interpolative_nuft
} // bfio

#endif // BFIO_INTERPOLATIVE_NUFT_FORM_CHECK_POTENTIALS_HPP

