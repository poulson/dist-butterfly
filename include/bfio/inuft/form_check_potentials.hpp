/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of ButterflyFIO and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef BFIO_INUFT_FORM_CHECK_POTENTIALS_HPP
#define BFIO_INUFT_FORM_CHECK_POTENTIALS_HPP

#include <array>
#include <cstddef>
#include <cstring>
#include <vector>

#include "bfio/constants.hpp"

#include "bfio/structures/weight_grid.hpp"
#include "bfio/structures/weight_grid_list.hpp"

#include "bfio/tools/special_functions.hpp"

#include "bfio/inuft/context.hpp"

namespace bfio {

using std::array;
using std::memset;
using std::size_t;
using std::vector;

namespace inuft {

// 1d specialization
template<typename R,size_t q>
inline void
FormCheckPotentials
( const Context<R,1,q>& context,
  const Plan<1>& plan,
  const size_t level,
  const array<vector<R>,1>& realPrescalings,
  const array<vector<R>,1>& imagPrescalings,
  const array<R,1>& x0A,
  const array<R,1>& p0B,
  const array<R,1>& wA,
  const array<R,1>& wB,
  const size_t parentInteractionOffset,
  const WeightGridList<R,1,q>& oldWeightGridList,
        WeightGrid<R,1,q>& weightGrid )
{
    const size_t d = 1;
    memset( weightGrid.Buffer(), 0, 2*q*sizeof(R) );

    const Direction direction = context.GetDirection();
    const R SignedTwoPi = ( direction==FORWARD ? -TwoPi<R>() : TwoPi<R>() );

    const size_t log2NumMergingProcesses = 
        plan.GetLog2NumMergingProcesses( level );
    const vector<R>& chebyshevNodes = context.GetChebyshevNodes();

    vector<R> realTempWeights0( q ), imagTempWeights0( q ),
              realTempWeights1( q ), imagTempWeights1( q );
    vector<R> postscalingArguments( q );
    vector<R> realPostscalings( q ), imagPostscalings( q );
    for( size_t cLocal=0;
         cLocal<(1u<<(1-log2NumMergingProcesses));
         ++cLocal )
    {
        const size_t interactionIndex = parentInteractionOffset + cLocal;
        const size_t c = plan.LocalToClusterSourceIndex( level, cLocal );
        const WeightGrid<R,d,q>& oldWeightGrid = 
            oldWeightGridList[interactionIndex];

        // Find the center of child c
        array<R,d> p0Bc;
        p0Bc[0] = p0B[0] + ( c&1 ? wB[0]/4 : -wB[0]/4 );

        // Prescaling
        {
            R* realWriteBuffer = &realTempWeights0[0];
            R* imagWriteBuffer = &imagTempWeights0[0];
            const R* realReadBuffer = oldWeightGrid.RealBuffer();
            const R* imagReadBuffer = oldWeightGrid.ImagBuffer();
            const R* realScalingBuffer = &realPrescalings[0][0];
            const R* imagScalingBuffer = &imagPrescalings[0][0];
            for( size_t t=0; t<q; ++t )
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
            const vector<R>& realForwardMap = 
                context.GetRealForwardMap( 0 );
            const vector<R>& imagForwardMap = 
                context.GetImagForwardMap( 0 );
            // Form the real part
            Gemv
            ( 'N', q, q,
              R(1), &realForwardMap[0], q,
                    &realTempWeights0[0], 1,
              R(0), &realTempWeights1[0], 1 );
            Gemv
            ( 'N', q, q,
              R(-1), &imagForwardMap[0], q,
                     &imagTempWeights0[0], 1,
              R(+1), &realTempWeights1[0], 1 );
            // Form the imaginary part
            Gemv
            ( 'N', q, q,
              R(1), &realForwardMap[0], q,
                    &imagTempWeights0[0], 1,
              R(0), &imagTempWeights1[0], 1 );
            Gemv
            ( 'N', q, q,
              R(1), &imagForwardMap[0], q,
                    &realTempWeights0[0], 1,
              R(1), &imagTempWeights1[0], 1 );
        }
        // Postscaling
        for( size_t t=0; t<q; ++t )
            postscalingArguments[t] = 
                -SignedTwoPi*(x0A[0]+chebyshevNodes[t]*wA[0])*p0Bc[0];
        SinCosBatch( postscalingArguments, imagPostscalings, realPostscalings );
        {
            R* realWriteBuffer = weightGrid.RealBuffer();
            R* imagWriteBuffer = weightGrid.ImagBuffer();
            const R* realReadBuffer = &realTempWeights1[0];
            const R* imagReadBuffer = &imagTempWeights1[0];
            const R* realScalingBuffer = &realPostscalings[0];
            const R* imagScalingBuffer = &imagPostscalings[0];
            for( size_t t=0; t<q; ++t )
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
template<typename R,size_t q>
inline void
FormCheckPotentials
( const Context<R,2,q>& context,
  const Plan<2>& plan,
  const size_t level,
  const array<vector<R>,2>& realPrescalings,
  const array<vector<R>,2>& imagPrescalings,
  const array<R,2>& x0A,
  const array<R,2>& p0B,
  const array<R,2>& wA,
  const array<R,2>& wB,
  const size_t parentInteractionOffset,
  const WeightGridList<R,2,q>& oldWeightGridList,
        WeightGrid<R,2,q>& weightGrid )
{
    const size_t d = 2;
    const size_t q_to_d = q*q;
    memset( weightGrid.Buffer(), 0, 2*q_to_d*sizeof(R) );

    const Direction direction = context.GetDirection();
    const R SignedTwoPi = ( direction==FORWARD ? -TwoPi<R>() : TwoPi<R>() );

    const size_t log2NumMergingProcesses = 
        plan.GetLog2NumMergingProcesses( level );
    const vector<R>& chebyshevNodes = context.GetChebyshevNodes();

    vector<R> realTempWeights0( q_to_d ), imagTempWeights0( q_to_d ),
              realTempWeights1( q_to_d ), imagTempWeights1( q_to_d );
    vector<R> postscalingArguments( q );
    vector<R> realPostscalings( q ), imagPostscalings( q );
    for( size_t cLocal=0; 
         cLocal<(1u<<(2-log2NumMergingProcesses));
         ++cLocal )
    {
        const size_t interactionIndex = parentInteractionOffset + cLocal;
        const size_t c = plan.LocalToClusterSourceIndex( level, cLocal );
        const WeightGrid<R,d,q>& oldWeightGrid = 
            oldWeightGridList[interactionIndex];

        // Find the center of child c
        array<R,d> p0Bc;
        for( size_t j=0; j<d; ++j )
            p0Bc[j] = p0B[j] + ( (c>>j)&1 ? wB[j]/4 : -wB[j]/4 );

        //--------------------------------------------------------------------//
        // Transform the first dimension                                      //
        //--------------------------------------------------------------------//
        // Prescale
        {
            R* realWriteBuffer = &realTempWeights0[0];
            R* imagWriteBuffer = &imagTempWeights0[0];
            const R* realReadBuffer = oldWeightGrid.RealBuffer();
            const R* imagReadBuffer = oldWeightGrid.ImagBuffer();
            const R* realScalingBuffer = &realPrescalings[0][0];
            const R* imagScalingBuffer = &imagPrescalings[0][0];
            for( size_t t=0; t<q; ++t )
            {
                for( size_t tPrime=0; tPrime<q; ++tPrime )
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
            const vector<R>& realForwardMap = 
                context.GetRealForwardMap( 0 );
            const vector<R>& imagForwardMap = 
                context.GetImagForwardMap( 0 );
            // Form the real part
            Gemm
            ( 'N', 'N', q, q, q,
              R(1), &realForwardMap[0], q,
                    &realTempWeights0[0], q,
              R(0), &realTempWeights1[0], q );
            Gemm
            ( 'N', 'N', q, q, q,
              R(-1), &imagForwardMap[0], q,
                     &imagTempWeights0[0], q,
              R(+1), &realTempWeights1[0], q );
            // Form the imaginary part
            Gemm
            ( 'N', 'N', q, q, q,
              R(1), &realForwardMap[0], q,
                    &imagTempWeights0[0], q,
              R(0), &imagTempWeights1[0], q );
            Gemm
            ( 'N', 'N', q, q, q,
              R(1), &imagForwardMap[0], q,
                    &realTempWeights0[0], q,
              R(1), &imagTempWeights1[0], q );
        }
        // Postscale
        for( size_t t=0; t<q; ++t )
            postscalingArguments[t] = 
                SignedTwoPi*(x0A[0]+chebyshevNodes[t]*wA[0])*p0Bc[0];
        SinCosBatch( postscalingArguments, imagPostscalings, realPostscalings );
        {
            R* realBuffer = &realTempWeights1[0];
            R* imagBuffer = &imagTempWeights1[0];
            const R* realScalingBuffer = &realPostscalings[0];
            const R* imagScalingBuffer = &imagPostscalings[0];
            for( size_t t=0; t<q; ++t )
            {
                for( size_t tPrime=0; tPrime<q; ++tPrime )
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
        {
            R* realBuffer = &realTempWeights1[0];
            R* imagBuffer = &imagTempWeights1[0];
            const R* realScalingBuffer = &realPrescalings[1][0];
            const R* imagScalingBuffer = &imagPrescalings[1][0];
            for( size_t w=0; w<q; ++w )
            {
                for( size_t t=0; t<q; ++t )
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
            const vector<R>& realForwardMap = 
                context.GetRealForwardMap( 1 );
            const vector<R>& imagForwardMap = 
                context.GetImagForwardMap( 1 );
            // Form the real part
            Gemm
            ( 'N', 'T', q, q, q,
              R(1), &realTempWeights1[0], q,
                    &realForwardMap[0], q,
              R(0), &realTempWeights0[0], q );
            Gemm
            ( 'N', 'T', q, q, q,
              R(-1), &imagTempWeights1[0], q,
                     &imagForwardMap[0], q,
              R(+1), &realTempWeights0[0], q );
            // Form the imaginary part
            Gemm
            ( 'N', 'T', q, q, q,
              R(1), &imagTempWeights1[0], q,
                    &realForwardMap[0], q,
              R(0), &imagTempWeights0[0], q );
            Gemm
            ( 'N', 'T', q, q, q,
              R(1), &realTempWeights1[0], q,
                    &imagForwardMap[0], q,
              R(1), &imagTempWeights0[0], q );
        }
        // Postscale
        for( size_t t=0; t<q; ++t )
            postscalingArguments[t] = 
                SignedTwoPi*(x0A[1]+chebyshevNodes[t]*wA[1])*p0Bc[1];
        SinCosBatch( postscalingArguments, imagPostscalings, realPostscalings );
        {
            R* realWriteBuffer = weightGrid.RealBuffer();
            R* imagWriteBuffer = weightGrid.ImagBuffer();
            const R* realReadBuffer = &realTempWeights0[0];
            const R* imagReadBuffer = &imagTempWeights0[0];
            const R* realScalingBuffer = &realPostscalings[0];
            const R* imagScalingBuffer = &imagPostscalings[0];
            for( size_t w=0; w<q; ++w )
            {
                for( size_t t=0; t<q; ++t )
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
template<typename R,size_t d,size_t q>
inline void
FormCheckPotentials
( const Context<R,d,q>& context,
  const Plan<d>& plan,
  const size_t level,
  const array<vector<R>,d>& realPrescalings,
  const array<vector<R>,d>& imagPrescalings,
  const array<R,d>& x0A,
  const array<R,d>& p0B,
  const array<R,d>& wA,
  const array<R,d>& wB,
  const size_t parentInteractionOffset,
  const WeightGridList<R,d,q>& oldWeightGridList,
        WeightGrid<R,d,q>& weightGrid )
{
    const size_t q_to_d = Pow<q,d>::val;
    memset( weightGrid.Buffer(), 0, 2*q_to_d*sizeof(R) );

    const Direction direction = context.GetDirection();
    const R SignedTwoPi = ( direction==FORWARD ? -TwoPi<R>() : TwoPi<R>() );

    const size_t log2NumMergingProcesses = 
        plan.GetLog2NumMergingProcesses( level );
    const vector<R>& chebyshevNodes = context.GetChebyshevNodes();

    vector<R> realTempWeights0( q_to_d ), imagTempWeights0( q_to_d ),
              realTempWeights1( q_to_d ), imagTempWeights1( q_to_d );
    vector<R> postscalingArguments( q );
    vector<R> realPostscalings( q ), imagPostscalings( q );
    for( size_t cLocal=0;
         cLocal<(1u<<(d-log2NumMergingProcesses));
         ++cLocal )
    {
        const size_t interactionIndex = parentInteractionOffset + cLocal;
        const size_t c = plan.LocalToClusterSourceIndex( level, cLocal );
        const WeightGrid<R,d,q>& oldWeightGrid = 
            oldWeightGridList[interactionIndex];

        // Find the center of child c
        array<R,d> p0Bc;
        for( size_t j=0; j<d; ++j )
            p0Bc[j] = p0B[j] + ( (c>>j)&1 ? wB[j]/4 : -wB[j]/4 );

        //--------------------------------------------------------------------//
        // Transform the first dimension                                      //
        //--------------------------------------------------------------------//
        // Prescale
        {
            R* realWriteBuffer = &realTempWeights0[0];
            R* imagWriteBuffer = &imagTempWeights0[0];
            const R* realReadBuffer = oldWeightGrid.RealBuffer();
            const R* imagReadBuffer = oldWeightGrid.ImagBuffer();
            const R* realScalingBuffer = &realPrescalings[0][0];
            const R* imagScalingBuffer = &imagPrescalings[0][0];
            for( size_t t=0; t<Pow<q,d-1>::val; ++t )
            {
                for( size_t tPrime=0; tPrime<q; ++tPrime )
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
            const vector<R>& realForwardMap = 
                context.GetRealForwardMap( 0 );
            const vector<R>& imagForwardMap = 
                context.GetImagForwardMap( 0 );
            // Form the real part
            Gemm
            ( 'N', 'N', q, Pow<q,d-1>::val, q,
              R(1), &realForwardMap[0], q,
                    &realTempWeights0[0], q,
              R(0), &realTempWeights1[0], q );
            Gemm
            ( 'N', 'N', q, Pow<q,d-1>::val, q,
              R(-1), &imagForwardMap[0], q,
                     &imagTempWeights0[0], q,
              R(+1), &realTempWeights1[0], q );
            // Form the imaginary part
            Gemm
            ( 'N', 'N', q, Pow<q,d-1>::val, q,
              R(1), &realForwardMap[0], q,
                    &imagTempWeights0[0], q,
              R(0), &imagTempWeights1[0], q );
            Gemm
            ( 'N', 'N', q, Pow<q,d-1>::val, q,
              R(1), &imagForwardMap[0], q,
                    &realTempWeights0[0], q,
              R(1), &imagTempWeights1[0], q );
        }
        // Postscale
        for( size_t t=0; t<q; ++t )
            postscalingArguments[t] = 
                SignedTwoPi*(x0A[0]+chebyshevNodes[t]*wA[0])*p0Bc[0];
        SinCosBatch( postscalingArguments, imagPostscalings, realPostscalings );
        {
            R* realBuffer = &realTempWeights1[0];
            R* imagBuffer = &imagTempWeights1[0];
            const R* realScalingBuffer = &realPostscalings[0];
            const R* imagScalingBuffer = &imagPostscalings[0];
            for( size_t t=0; t<Pow<q,d-1>::val; ++t )
            {
                for( size_t tPrime=0; tPrime<q; ++tPrime )
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
        for( size_t p=0; p<Pow<q,d-2>::val; ++p )
        {
            const size_t offset = p*q*q; 
            R* offsetRealBuffer = &realTempWeights1[offset];
            R* offsetImagBuffer = &imagTempWeights1[offset];
            const R* realScalingBuffer = &realPrescalings[1][0];
            const R* imagScalingBuffer = &imagPrescalings[1][0];
            for( size_t w=0; w<q; ++w )
            {
                for( size_t t=0; t<q; ++t )
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
            const vector<R>& realForwardMap = 
                context.GetRealForwardMap( 1 );
            const vector<R>& imagForwardMap = 
                context.GetImagForwardMap( 1 );
            for( size_t w=0; w<Pow<q,d-2>::val; ++w )
            {
                // Form the real part
                Gemm
                ( 'N', 'T', q, q, q,
                  R(1), &realTempWeights1[w*q*q], q,
                        &realForwardMap[0], q,
                  R(0), &realTempWeights0[w*q*q], q );
                Gemm
                ( 'N', 'T', q, q, q,
                  R(-1), &imagTempWeights1[w*q*q], q,
                         &imagForwardMap[0], q,
                  R(+1), &realTempWeights0[w*q*q], q );
                // Form the imaginary part
                Gemm
                ( 'N', 'T', q, q, q,
                  R(1), &imagTempWeights1[w*q*q], q,
                        &realForwardMap[0], q,
                  R(0), &imagTempWeights0[w*q*q], q );
                Gemm
                ( 'N', 'T', q, q, q,
                  R(1), &realTempWeights1[w*q*q], q,
                        &imagForwardMap[0], q,
                  R(1), &imagTempWeights0[w*q*q], q );
            }
        }
        // Postscale
        for( size_t t=0; t<q; ++t )
            postscalingArguments[t] =
                SignedTwoPi*(x0A[1]+chebyshevNodes[t]*wA[1])*p0Bc[1];
        SinCosBatch( postscalingArguments, imagPostscalings, realPostscalings );
        for( size_t p=0; p<Pow<q,d-2>::val; ++p )
        {
            const size_t offset = p*q*q;
            R* offsetRealBuffer = &realTempWeights0[offset];
            R* offsetImagBuffer = &imagTempWeights0[offset];
            const R* realScalingBuffer = &realPostscalings[0];
            const R* imagScalingBuffer = &imagPostscalings[0];
            for( size_t w=0; w<q; ++w )
            {
                for( size_t t=0; t<q; ++t )
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
        size_t q_to_j = q*q;
        for( size_t j=2; j<d; ++j )
        {
            const size_t stride = q_to_j;

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

            const vector<R>& realForwardMap = 
                context.GetRealForwardMap( j );
            const vector<R>& imagForwardMap = 
                context.GetImagForwardMap( j );
            const R* realForwardBuffer = &realForwardMap[0];
            const R* imagForwardBuffer = &imagForwardMap[0];

            if( j != d-1 )
            {
                memset( realWriteBuffer, 0, q_to_d*sizeof(R) );
                memset( imagWriteBuffer, 0, q_to_d*sizeof(R) );
            }

            // Prescale, apply the forward map, and then postscale
            vector<R> realTempWeightStrip( q ), imagTempWeightStrip( q );
            for( size_t t=0; t<q; ++t )
                postscalingArguments[t] = 
                    SignedTwoPi*(x0A[j]+chebyshevNodes[t]*wA[j])*p0Bc[j];
            SinCosBatch
            ( postscalingArguments, imagPostscalings, realPostscalings );
            for( size_t p=0; p<q_to_d/(q_to_j*q); ++p )        
            {
                const size_t offset = p*(q_to_j*q);
                R* offsetRealReadBuffer = &realReadBuffer[offset];
                R* offsetImagReadBuffer = &imagReadBuffer[offset];
                R* offsetRealWriteBuffer = &realWriteBuffer[offset];
                R* offsetImagWriteBuffer = &imagWriteBuffer[offset];
                const R* realPrescalingBuffer = &realPrescalings[j][0];
                const R* imagPrescalingBuffer = &imagPrescalings[j][0];
                const R* realPostscalingBuffer = &realPostscalings[0];
                const R* imagPostscalingBuffer = &imagPostscalings[0];
                for( size_t w=0; w<q_to_j; ++w )
                {
                    // Prescale
                    for( size_t t=0; t<q; ++t )
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
                    for( size_t t=0; t<q; ++t )
                    {
                        offsetRealReadBuffer[w+t*stride] = 0;
                        offsetImagReadBuffer[w+t*stride] = 0;
                        for( size_t tPrime=0; tPrime<q; ++tPrime )
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
                    for( size_t t=0; t<q; ++t )
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

} // inuft
} // bfio

#endif // ifndef BFIO_INUFT_FORM_CHECK_POTENTIALS_HPP
