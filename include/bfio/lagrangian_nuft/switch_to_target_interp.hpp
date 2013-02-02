/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of ButterflyFIO and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef BFIO_LAGRANGIAN_NUFT_ADJOINT_SWITCH_TO_TARGET_INTERP_HPP
#define BFIO_LAGRANGIAN_NUFT_ADJOINT_SWITCH_TO_TARGET_INTERP_HPP

#include <array>
#include <cstddef>
#include <complex>
#include <vector>

#include "bfio/structures/box.hpp"
#include "bfio/structures/constrained_htree_walker.hpp"
#include "bfio/structures/plan.hpp"
#include "bfio/structures/weight_grid.hpp"
#include "bfio/structures/weight_grid_list.hpp"

#include "bfio/lagrangian_nuft/context.hpp"

namespace bfio {

using std::complex;
using std::memcpy;
using std::memset;
using std::size_t;

namespace lagrangian_nuft {

// 1d specialization
template<typename R,size_t q>
void
SwitchToTargetInterp
( const lagrangian_nuft::Context<R,1,q>& nuftContext,
  const Plan<1>& plan,
  const Box<R,1>& sourceBox,
  const Box<R,1>& targetBox,
  const Box<R,1>& mySourceBox,
  const Box<R,1>& myTargetBox,
  const size_t log2LocalSourceBoxes,
  const size_t log2LocalTargetBoxes,
  const array<size_t,1>& log2LocalSourceBoxesPerDim,
  const array<size_t,1>& log2LocalTargetBoxesPerDim,
        WeightGridList<R,1,q>& weightGridList )
{ 
    typedef complex<R> C;
    const size_t d = 1;
    const rfio::Context<R,1,q>& rfioContext = 
        nuftContext.GetReducedFIOContext();

    const Direction direction = nuftContext.GetDirection();
    const R SignedTwoPi = ( direction==FORWARD ? -TwoPi : TwoPi );

    // Compute the width of the nodes at level log2N/2
    const size_t N = plan.GetN();
    const size_t log2N = Log2( N );
    const size_t level = log2N/2;
    array<R,d> wA, wB;
    wA[0] = targetBox.widths[0] / (1<<level);
    wB[0] = sourceBox.widths[0] / (1<<(log2N-level));

    // Get the precomputed grid offset evaluations, exp( +-TwoPi i (dx,dp) )
    const array<vector<R>,d>& realOffsetEvals = 
        nuftContext.GetRealOffsetEvaluations();
    const array<vector<R>,d>& imagOffsetEvals = 
        nuftContext.GetImagOffsetEvaluations();

    // Create space for holding the mixed offset evaluations, i.e., 
    // exp( +-TwoPi i (x0,dp) ) and exp( +-TwoPi i (dx,p0) )
    vector<R> phaseEvaluations( q );
    vector<vector<R>> realFixedTargetEvals( d, vector<R>(q) );
    vector<vector<R>> imagFixedTargetEvals( d, vector<R>(q) );
    vector<vector<vector<R>>>
        realFixedSourceEvals
        ( 1<<log2LocalSourceBoxes, vector<vector<R>>( d, vector<R>(q) ) );
    vector<vector<vector<R>>>
        imagFixedSourceEvals
        ( 1<<log2LocalSourceBoxes, vector<vector<R>>( d, vector<R>(q) ) );

    // Create space for holding q weights
    vector<R> realOldWeights( q ), imagOldWeights( q ), 
              realTempWeights( q ), imagTempWeights( q );

    const vector<R>& chebyshevNodes = rfioContext.GetChebyshevNodes();
    ConstrainedHTreeWalker<d> AWalker( log2LocalTargetBoxesPerDim );
    for( size_t i=0; i<(1u<<log2LocalTargetBoxes); ++i, AWalker.Walk() )
    {
        const array<size_t,d> A = AWalker.State();

        // Compute the coordinates and center of this target box
        array<R,d> x0A;
        x0A[0] = myTargetBox.offsets[0] + (A[0]+0.5)*wA[0];

        // Evaluate exp( +-TwoPi i (x0,dp) ) 
        for( size_t t=0; t<q; ++t )
            phaseEvaluations[t] = SignedTwoPi*wB[0]*x0A[0]*chebyshevNodes[t];
        SinCosBatch
        ( phaseEvaluations, 
          imagFixedTargetEvals[0], realFixedTargetEvals[0] );

        ConstrainedHTreeWalker<d> BWalker( log2LocalSourceBoxesPerDim );
        for( size_t k=0; 
             k<(1u<<log2LocalSourceBoxes); 
             ++k, BWalker.Walk() )
        {
            const array<size_t,d> B = BWalker.State();

            // Compute the coordinates and center of this source box
            array<R,d> p0B;
            p0B[0] = mySourceBox.offsets[0] + (B[0]+0.5)*wB[0];

            // Ensure that we've evaluated exp( +-TwoPi i (dx,p0) ) 
            if( i == 0 )
            {
                for( size_t t=0; t<q; ++t )
                {
                    phaseEvaluations[t] = 
                        SignedTwoPi*wA[0]*p0B[0]*chebyshevNodes[t];
                }
                SinCosBatch
                ( phaseEvaluations,
                  imagFixedSourceEvals[k][0], realFixedSourceEvals[k][0] );
            }

            const size_t key = k+(i<<log2LocalSourceBoxes);
            memcpy
            ( &realOldWeights, weightGridList[key].RealBuffer(), q*sizeof(R) );
            memcpy
            ( &imagOldWeights, weightGridList[key].ImagBuffer(), q*sizeof(R) );
            memset( weightGridList[key].Buffer(), 0, 2*q*sizeof(R) );

            // Switch over the first dimension.
            // Scale
            {
                R* realBuffer = &realOldWeights[0];
                R* imagBuffer = &imagOldWeights[0];
                const R* realScalingBuffer = &realFixedTargetEvals[0][0];
                const R* imagScalingBuffer = &imagFixedTargetEvals[0][0];
                for( size_t t=0; t<q; ++t )
                {
                    const R realWeight = realBuffer[t];
                    const R imagWeight = imagBuffer[t];
                    const R realScaling = realScalingBuffer[t];
                    const R imagScaling = imagScalingBuffer[t];
                    realBuffer[t] = 
                        realWeight*realScaling - imagWeight*imagScaling;
                    imagBuffer[t] = 
                        imagWeight*realScaling + realWeight*imagScaling;
                }
            }
            // Form real part
            // TODO: Replace with Gemv's
            Gemm
            ( 'N', 'N', q, 1, q,
              (R)1, &realOffsetEvals[0][0], q,
                    &realOldWeights[0],     q,
              (R)0, &realTempWeights[0],    q );
            Gemm
            ( 'N', 'N', q, 1, q,
              (R)-1, &imagOffsetEvals[0][0], q,
                     &imagOldWeights[0],     q,
              (R)+1, &realTempWeights[0],    q );
            // Form imaginary part
            Gemm
            ( 'N', 'N', q, 1, q,
              (R)1, &realOffsetEvals[0][0], q,
                    &imagOldWeights[0],     q,
              (R)0, &imagTempWeights[0],    q );
            Gemm
            ( 'N', 'N', q, 1, q,
              (R)-1, &imagOffsetEvals[0][0], q,
                     &realOldWeights[0],     q,
              (R)+1, &imagTempWeights[0],    q );

            // Post process scaling
            // Apply the exp( +-TwoPi i (x0,p0) ) and exp( +-TwoPi i (dx,p0) ) 
            // terms
            R phase = SignedTwoPi*x0A[0]*p0B[0];
            const R realPhase = cos(phase);
            const R imagPhase = sin(phase);
            vector<R> realScalings( q );
            vector<R> imagScalings( q );
            for( size_t t=0; t<q; ++t )
            {
                const R realTerm = realFixedSourceEvals[k][0][t];
                const R imagTerm = imagFixedSourceEvals[k][0][t];
                realScalings[t] = realTerm*realPhase - imagTerm*imagPhase;
                imagScalings[t] = imagTerm*realPhase + realTerm*imagPhase;
            }
            R* realBuffer = weightGridList[key].RealBuffer();
            R* imagBuffer = weightGridList[key].ImagBuffer();
            const R* realScalingBuffer = &realScalings[0];
            const R* imagScalingBuffer = &imagScalings[0];
            for( size_t t=0; t<q; ++t )
            {
                const R realWeight = realBuffer[t];
                const R imagWeight = imagBuffer[t];
                const R realScaling = realScalingBuffer[t];
                const R imagScaling = imagScalingBuffer[t];
                realBuffer[t] = realWeight*realScaling - imagWeight*imagScaling;
                imagBuffer[t] = imagWeight*realScaling + realWeight*imagScaling;
            }
        }
    }
}

// 2d specialization
template<typename R,size_t q>
void
SwitchToTargetInterp
( const lagrangian_nuft::Context<R,2,q>& nuftContext,
  const Plan<2>& plan,
  const Box<R,2>& sourceBox,
  const Box<R,2>& targetBox,
  const Box<R,2>& mySourceBox,
  const Box<R,2>& myTargetBox,
  const size_t log2LocalSourceBoxes,
  const size_t log2LocalTargetBoxes,
  const array<size_t,2>& log2LocalSourceBoxesPerDim,
  const array<size_t,2>& log2LocalTargetBoxesPerDim,
        WeightGridList<R,2,q>& weightGridList )
{
    typedef complex<R> C;
    const size_t d = 2;
    const size_t q_to_d = Pow<q,d>::val;
    const rfio::Context<R,2,q>& rfioContext = 
        nuftContext.GetReducedFIOContext();

    const Direction direction = nuftContext.GetDirection();
    const R SignedTwoPi = ( direction==FORWARD ? -TwoPi : TwoPi );

    // Compute the width of the nodes at level log2N/2
    const size_t N = plan.GetN();
    const size_t log2N = Log2( N );
    const size_t level = log2N/2;
    array<R,d> wA, wB;
    for( size_t j=0; j<d; ++j )
    {
        wA[j] = targetBox.widths[j] / (1<<level);
        wB[j] = sourceBox.widths[j] / (1<<(log2N-level));
    }

    // Get the precomputed grid offset evaluations, exp( +-TwoPi i (dx,dp) )
    const array<vector<R>,d>& realOffsetEvals = 
        nuftContext.GetRealOffsetEvaluations();
    const array<vector<R>,d>& imagOffsetEvals = 
        nuftContext.GetImagOffsetEvaluations();

    // Create space for holding the mixed offset evaluations, i.e., 
    // exp( +-TwoPi i (x0,dp) ) and exp( +-TwoPi i (dx,p0) )
    vector<R> phaseEvaluations( q );
    vector<vector<R>> realFixedTargetEvals( d, vector<R>(q) );
    vector<vector<R>> imagFixedTargetEvals( d, vector<R>(q) );
    vector<vector<vector<R>>>
        realFixedSourceEvals
        ( 1<<log2LocalSourceBoxes, vector<vector<R>>( d, vector<R>(q) ) );
    vector<vector<vector<R>>>
        imagFixedSourceEvals
        ( 1<<log2LocalSourceBoxes, vector<vector<R>>( d, vector<R>(q) ) );

    // Create space for holding q^d weights
    vector<R> realOldWeights( q_to_d ), imagOldWeights( q_to_d ),
              realTempWeights( q_to_d ), imagTempWeights( q_to_d );

    const vector<R>& chebyshevNodes = rfioContext.GetChebyshevNodes();
    ConstrainedHTreeWalker<d> AWalker( log2LocalTargetBoxesPerDim );
    for( size_t i=0; i<(1u<<log2LocalTargetBoxes); ++i, AWalker.Walk() )
    {
        const array<size_t,d> A = AWalker.State();

        // Compute the coordinates and center of this target box
        array<R,d> x0A;
        for( size_t j=0; j<d; ++j )
            x0A[j] = myTargetBox.offsets[j] + (A[j]+0.5)*wA[j];

        // Evaluate exp( +-TwoPi i (x0,dp) ) for each coordinate
        for( size_t j=0; j<d; ++j )
        {
            for( size_t t=0; t<q; ++t )
                phaseEvaluations[t] = 
                    SignedTwoPi*wB[j]*x0A[j]*chebyshevNodes[t];
            SinCosBatch
            ( phaseEvaluations, 
              imagFixedTargetEvals[j], realFixedTargetEvals[j] );
        }

        ConstrainedHTreeWalker<d> BWalker( log2LocalSourceBoxesPerDim );
        for( size_t k=0; 
             k<(1u<<log2LocalSourceBoxes); 
             ++k, BWalker.Walk() )
        {
            const array<size_t,d> B = BWalker.State();

            // Compute the coordinates and center of this source box
            array<R,d> p0B;
            for( size_t j=0; j<d; ++j )
                p0B[j] = mySourceBox.offsets[j] + (B[j]+0.5)*wB[j];

            // Evaluate exp( +-TwoPi i (dx,p0) ) for each coord
            if( i == 0 )
            {
                for( size_t j=0; j<d; ++j )
                {
                    for( size_t t=0; t<q; ++t )
                    {
                        phaseEvaluations[t] = 
                            SignedTwoPi*wA[j]*p0B[j]*chebyshevNodes[t];
                    }
                    SinCosBatch
                    ( phaseEvaluations,
                      imagFixedSourceEvals[k][j], realFixedSourceEvals[k][j] );
                }
            }

            const size_t key = k+(i<<log2LocalSourceBoxes);
            memcpy
            ( &realOldWeights[0], weightGridList[key].RealBuffer(), 
              q_to_d*sizeof(R) );
            memcpy
            ( &imagOldWeights[0], weightGridList[key].ImagBuffer(),
              q_to_d*sizeof(R) );
            memset( weightGridList[key].Buffer(), 0, 2*q_to_d*sizeof(R) );

            // Switch over the first dimension.
            // Scale
            {
                R* realBuffer = &realOldWeights[0];
                R* imagBuffer = &imagOldWeights[0];
                const R* realScalingBuffer = &realFixedTargetEvals[0][0];
                const R* imagScalingBuffer = &imagFixedTargetEvals[0][0];
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
            // Form real part
            Gemm
            ( 'N', 'N', q, q, q,
              (R)1, &realOffsetEvals[0][0], q,
                    &realOldWeights[0],     q,
              (R)0, &realTempWeights[0],    q );
            Gemm
            ( 'N', 'N', q, q, q,
              (R)-1, &imagOffsetEvals[0][0], q,
                     &imagOldWeights[0],     q,
              (R)+1, &realTempWeights[0],    q );
            // Form imaginary part
            Gemm
            ( 'N', 'N', q, q, q,
              (R)1, &realOffsetEvals[0][0], q,
                    &imagOldWeights[0],     q,
              (R)0, &imagTempWeights[0],    q );
            Gemm
            ( 'N', 'N', q, q, q,
              (R)+1, &imagOffsetEvals[0][0], q,
                     &realOldWeights[0],     q,
              (R)+1, &imagTempWeights[0],    q );

            // Switch over second dimension
            // Scale
            {
                R* realBuffer = &realTempWeights[0];
                R* imagBuffer = &imagTempWeights[0];
                const R* realScalingBuffer = &realFixedTargetEvals[1][0];
                const R* imagScalingBuffer = &imagFixedTargetEvals[1][0];
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
            // Form the real and imaginary parts
            Gemm
            ( 'N', 'T', q, q, q,
              (R)+1, &realTempWeights[0], q,
                     &realOffsetEvals[1][0], q,
              (R)+1, weightGridList[key].RealBuffer(), q );
            Gemm
            ( 'N', 'T', q, q, q,
              (R)-1, &imagTempWeights[0], q,
                     &imagOffsetEvals[1][0], q,
              (R)+1, weightGridList[key].RealBuffer(), q );

            Gemm
            ( 'N', 'T', q, q, q,
              (R)+1, &imagTempWeights[0], q,
                     &realOffsetEvals[1][0], q,
              (R)+1, weightGridList[key].ImagBuffer(), q );
            Gemm
            ( 'N', 'T', q, q, q,
              (R)+1, &realTempWeights[0], q,
                     &imagOffsetEvals[1][0], q,
              (R)+1, weightGridList[key].ImagBuffer(), q );

            // Post process scaling
            //
            // Apply the exp( +-TwoPi i (x0,p0) ) term by scaling the 
            // exp( +-TwoPi i (dx,p0) ) terms before their application
            size_t q_to_j = 1;
            R* realBuffer = weightGridList[key].RealBuffer();
            R* imagBuffer = weightGridList[key].ImagBuffer();
            vector<R> realScalings( q ), imagScalings( q );
            for( size_t j=0; j<d; ++j )
            {
                const R phase = SignedTwoPi*x0A[j]*p0B[j]; 
                const R realPhase = cos(phase);
                const R imagPhase = sin(phase);
                for( size_t t=0; t<q; ++t )
                {
                    const R realTerm = realFixedSourceEvals[k][j][t];
                    const R imagTerm = imagFixedSourceEvals[k][j][t];
                    realScalings[t] = realTerm*realPhase - imagTerm*imagPhase;
                    imagScalings[t] = imagTerm*realPhase + realTerm*imagPhase;
                }

                const size_t stride = q_to_j;
                for( size_t p=0; p<q/q_to_j; ++p )
                {
                    const size_t offset = p*(q_to_j*q);
                    R* offsetRealBuffer = &realBuffer[offset];
                    R* offsetImagBuffer = &imagBuffer[offset];
                    const R* realScalingBuffer = &realScalings[0];
                    const R* imagScalingBuffer = &imagScalings[0];
                    for( size_t w=0; w<q_to_j; ++w )
                    {
                        for( size_t t=0; t<q; ++t )
                        {
                            const R realWeight = offsetRealBuffer[w+t*stride];
                            const R imagWeight = offsetImagBuffer[w+t*stride];
                            const R realScaling = realScalingBuffer[t];
                            const R imagScaling = imagScalingBuffer[t];
                            offsetRealBuffer[w+t*stride] = 
                                realWeight*realScaling - imagWeight*imagScaling;
                            offsetImagBuffer[w+t*stride] = 
                                imagWeight*realScaling + realWeight*imagScaling;
                        }
                    }
                }
                q_to_j *= q;
            }
        }
    }
}

// Fallback for 3d and above
template<typename R,size_t d,size_t q>
void
SwitchToTargetInterp
( const lagrangian_nuft::Context<R,d,q>& nuftContext,
  const Plan<d>& plan,
  const Box<R,d>& sourceBox,
  const Box<R,d>& targetBox,
  const Box<R,d>& mySourceBox,
  const Box<R,d>& myTargetBox,
  const size_t log2LocalSourceBoxes,
  const size_t log2LocalTargetBoxes,
  const array<size_t,d>& log2LocalSourceBoxesPerDim,
  const array<size_t,d>& log2LocalTargetBoxesPerDim,
        WeightGridList<R,d,q>& weightGridList )
{
    typedef complex<R> C;
    const size_t q_to_d = Pow<q,d>::val;
    const rfio::Context<R,d,q>& rfioContext = 
        nuftContext.GetReducedFIOContext();

    const Direction direction = nuftContext.GetDirection();
    const R SignedTwoPi = ( direction==FORWARD ? -TwoPi : TwoPi );

    // Compute the width of the nodes at level log2N/2
    const size_t N = plan.GetN();
    const size_t log2N = Log2( N );
    const size_t level = log2N/2;
    array<R,d> wA, wB;
    for( size_t j=0; j<d; ++j )
    {
        wA[j] = targetBox.widths[j] / (1<<level);
        wB[j] = sourceBox.widths[j] / (1<<(log2N-level));
    }

    // Get the precomputed grid offset evaluations, exp( +-TwoPi i (dx,dp) )
    const array<vector<R>,d>& realOffsetEvals = 
        nuftContext.GetRealOffsetEvaluations();
    const array<vector<R>,d>& imagOffsetEvals = 
        nuftContext.GetImagOffsetEvaluations();

    // Create space for holding the mixed offset evaluations, i.e., 
    // exp( +-TwoPi i (x0,dp) ) and exp( +-TwoPi i (dx,p0) )
    vector<R> phaseEvaluations( q );
    vector<vector<R>> realFixedTargetEvals( d, vector<R>(q) );
    vector<vector<R>> imagFixedTargetEvals( d, vector<R>(q) );
    vector<vector<vector<R>>>
        realFixedSourceEvals
        ( 1<<log2LocalSourceBoxes, vector<vector<R>>( d, vector<R>(q) ) );
    vector<vector<vector<R>>>
        imagFixedSourceEvals
        ( 1<<log2LocalSourceBoxes, vector<vector<R>>( d, vector<R>(q) ) );

    // Create space for holding q^d weights
    vector<R> realOldWeights( q_to_d ), imagOldWeights( q_to_d ),
              realTempWeights( q_to_d ), imagTempWeights( q_to_d );

    const vector<R>& chebyshevNodes = rfioContext.GetChebyshevNodes();
    ConstrainedHTreeWalker<d> AWalker( log2LocalTargetBoxesPerDim );
    for( size_t i=0; i<(1u<<log2LocalTargetBoxes); ++i, AWalker.Walk() )
    {
        const array<size_t,d> A = AWalker.State();

        // Compute the coordinates and center of this target box
        array<R,d> x0A;
        for( size_t j=0; j<d; ++j )
            x0A[j] = myTargetBox.offsets[j] + (A[j]+0.5)*wA[j];

        // Evaluate exp( +-TwoPi i (x0,dp) ) for each coordinate
        for( size_t j=0; j<d; ++j )
        {
            for( size_t t=0; t<q; ++t )
                phaseEvaluations[t] = +-TwoPi*wB[j]*x0A[j]*chebyshevNodes[t];
            SinCosBatch
            ( phaseEvaluations, 
              imagFixedTargetEvals[j], realFixedTargetEvals[j] );
        }

        ConstrainedHTreeWalker<d> BWalker( log2LocalSourceBoxesPerDim );
        for( size_t k=0; 
             k<(1u<<log2LocalSourceBoxes); 
             ++k, BWalker.Walk() )
        {
            const array<size_t,d> B = BWalker.State();

            // Compute the coordinates and center of this source box
            array<R,d> p0B;
            for( size_t j=0; j<d; ++j )
                p0B[j] = mySourceBox.offsets[j] + (B[j]+0.5)*wB[j];

            // Evaluate exp( +-TwoPi i (dx,p0) ) for each coord
            if( i == 0 )
            {
                for( size_t j=0; j<d; ++j )
                {
                    for( size_t t=0; t<q; ++t )
                    {
                        phaseEvaluations[t] = 
                            SignedTwoPi*wA[j]*p0B[j]*chebyshevNodes[t];
                    }
                    SinCosBatch
                    ( phaseEvaluations,
                      imagFixedSourceEvals[k][j], realFixedSourceEvals[k][j] );
                }
            }

            const size_t key = k+(i<<log2LocalSourceBoxes);
            memcpy
            ( &realOldWeights[0], weightGridList[key].RealBuffer(), 
              q_to_d*sizeof(R) );
            memcpy
            ( &imagOldWeights[0], weightGridList[key].ImagBuffer(),
              q_to_d*sizeof(R) );
            memset( weightGridList[key].Buffer(), 0, 2*q_to_d*sizeof(R) );

            // Switch over the first dimension.
            // Scale
            {
                R* realBuffer = &realOldWeights[0];
                R* imagBuffer = &imagOldWeights[0];
                const R* realScalingBuffer = &realFixedTargetEvals[0][0];
                const R* imagScalingBuffer = &imagFixedTargetEvals[0][0];
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
            // Form real part
            Gemm
            ( 'N', 'N', q, Pow<q,d-1>::val, q,
              (R)1, &realOffsetEvals[0][0], q,
                    &realOldWeights[0],     q,
              (R)0, &realTempWeights[0],    q );
            Gemm
            ( 'N', 'N', q, Pow<q,d-1>::val, q,
              (R)-1, &imagOffsetEvals[0][0], q,
                     &imagOldWeights[0],     q,
              (R)+1, &realTempWeights[0],    q );
            // Form imaginary part
            Gemm
            ( 'N', 'N', q, Pow<q,d-1>::val, q,
              (R)1, &realOffsetEvals[0][0], q,
                    &imagOldWeights[0],     q,
              (R)0, &imagTempWeights[0],    q );
            Gemm
            ( 'N', 'N', q, Pow<q,d-1>::val, q,
              (R)+1, &imagOffsetEvals[0][0], q,
                     &realOldWeights[0],     q,
              (R)+1, &imagTempWeights[0],    q );

            // Switch over second dimension
            // Scale
            for( size_t p=0; p<Pow<q,d-2>::val; ++p )
            {
                const size_t offset = p*q*q;
                R* offsetRealBuffer = &realTempWeights[offset];
                R* offsetImagBuffer = &imagTempWeights[offset];
                const R* realScalingBuffer = &realFixedTargetEvals[1][0];
                const R* imagScalingBuffer = &imagFixedTargetEvals[1][0];
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
            // Form the real and imaginary parts
            for( size_t w=0; w<Pow<q,d-2>::val; ++w )
            {
                Gemm
                ( 'N', 'T', q, q, q,
                  (R)1, &realTempWeights[w*q*q], q,
                        &realOffsetEvals[1][0],  q,
                  (R)0, &realOldWeights[w*q*q],  q );
                Gemm
                ( 'N', 'T', q, q, q,
                  (R)-1, &imagTempWeights[w*q*q], q,
                         &imagOffsetEvals[1][0],  q,
                  (R)+1, &realOldWeights[w*q*q],  q );

                Gemm
                ( 'N', 'T', q, q, q,
                  (R)1, &imagTempWeights[w*q*q], q,
                        &realOffsetEvals[1][0],  q,
                  (R)0, &imagOldWeights[w*q*q],  q );
                Gemm
                ( 'N', 'T', q, q, q,
                  (R)1, &realTempWeights[w*q*q], q,
                        &imagOffsetEvals[1][0],  q,
                  (R)1, &imagOldWeights[w*q*q],  q );
            }

            // Switch over remaining dimensions
            size_t q_to_j = q*q;
            for( size_t j=2; j<d; ++j )
            {
                const size_t stride = q_to_j;

                R* realWriteBuffer = 
                    ( j==d-1 ? weightGridList[key].RealBuffer()
                             : ( j&1 ? &realOldWeights[0]
                                     : &realTempWeights[0] ) );
                R* imagWriteBuffer = 
                    ( j==d-1 ? weightGridList[key].ImagBuffer()
                             : ( j&1 ? &imagOldWeights[0]
                                     : &imagTempWeights[0] ) );
                R* realReadBuffer = 
                    ( j&1 ? &realTempWeights[0] : &realOldWeights[0] );
                R* imagReadBuffer = 
                    ( j&1 ? &imagTempWeights[0] : &imagOldWeights[0] );
                const R* realOffsetBuffer = &realOffsetEvals[j][0];
                const R* imagOffsetBuffer = &imagOffsetEvals[j][0];

                // Scale and transform
                if( j != d-1 )
                {
                    memset( realWriteBuffer, 0, q_to_d*sizeof(R) );
                    memset( imagWriteBuffer, 0, q_to_d*sizeof(R) );
                }
                for( size_t p=0; p<q_to_d/(q_to_j*q); ++p )
                {
                    const size_t offset = p*(q_to_j*q);
                    R* offsetRealReadBuffer = &realReadBuffer[offset];
                    R* offsetImagReadBuffer = &imagReadBuffer[offset];
                    R* offsetRealWriteBuffer = &realWriteBuffer[offset];
                    R* offsetImagWriteBuffer = &imagWriteBuffer[offset];
                    const R* realScalingBuffer = &realFixedTargetEvals[j][0];
                    const R* imagScalingBuffer = &imagFixedTargetEvals[j][0];
                    for( size_t w=0; w<q_to_j; ++w )
                    {
                        for( size_t t=0; t<q; ++t )
                        {
                            const R realWeight = 
                                offsetRealReadBuffer[w+t*stride];
                            const R imagWeight = 
                                offsetImagReadBuffer[w+t*stride];
                            const R realScaling = realScalingBuffer[t];
                            const R imagScaling = imagScalingBuffer[t];
                            offsetRealReadBuffer[w+t*stride] = 
                                realWeight*realScaling - imagWeight*imagScaling;
                            offsetImagReadBuffer[w+t*stride] = 
                                imagWeight*realScaling + realWeight*imagScaling;
                        }
                        for( size_t t=0; t<q; ++t )
                        {
                            for( size_t tPrime=0; tPrime<q; ++tPrime )
                            {
                                offsetRealWriteBuffer[w+t*stride] +=
                                    realOffsetBuffer[t+tPrime*q] *
                                    offsetRealReadBuffer[w+tPrime*stride];
                                offsetRealWriteBuffer[w+t*stride] -=
                                    imagOffsetBuffer[t+tPrime*q] *
                                    offsetImagReadBuffer[w+tPrime*stride];

                                offsetImagWriteBuffer[w+t*stride] +=
                                    imagOffsetBuffer[t+tPrime*q] *
                                    offsetRealReadBuffer[w+tPrime*stride];
                                offsetImagWriteBuffer[w+t*stride] +=
                                    realOffsetBuffer[t+tPrime*q] *
                                    offsetImagReadBuffer[w+tPrime*stride];
                            }
                        }
                    }
                }
                q_to_j *= q;
            }

            // Post process scaling
            //
            // Apply the exp( +-TwoPi i (x0,p0) ) term by scaling the 
            // exp( +-TwoPi i (dx,p0) ) terms before their application
            q_to_j = 1;
            R* realBuffer = weightGridList[key].RealBuffer();
            R* imagBuffer = weightGridList[key].ImagBuffer();
            vector<R> realScalings( q ), imagScalings( q );
            for( size_t j=0; j<d; ++j )
            {
                const R phase = SignedTwoPi*x0A[j]*p0B[j];
                const R realPhase = cos(phase);
                const R imagPhase = sin(phase);
                for( size_t t=0; t<q; ++t )
                {
                    const R realTerm = realFixedSourceEvals[k][j][t];
                    const R imagTerm = imagFixedSourceEvals[k][j][t];
                    realScalings[t] = realTerm*realPhase - imagTerm*imagPhase;
                    imagScalings[t] = imagTerm*realPhase + realTerm*imagPhase;
                }

                const size_t stride = q_to_j;
                for( size_t p=0; p<q_to_d/(q_to_j*q); ++p )
                {
                    const size_t offset = p*(q_to_j*q);
                    R* offsetRealBuffer = &realBuffer[offset];
                    R* offsetImagBuffer = &imagBuffer[offset];
                    const R* realScalingBuffer = &realScalings[0];
                    const R* imagScalingBuffer = &imagScalings[0];
                    for( size_t w=0; w<q_to_j; ++w )
                    {
                        for( size_t t=0; t<q; ++t )
                        {
                            const R realWeight = offsetRealBuffer[w+t*stride];
                            const R imagWeight = offsetImagBuffer[w+t*stride];
                            const R realScaling = realScalingBuffer[t];
                            const R imagScaling = imagScalingBuffer[t];
                            offsetRealBuffer[w+t*stride] = 
                                realWeight*realScaling - imagWeight*imagScaling;
                            offsetImagBuffer[w+t*stride] = 
                                imagWeight*realScaling + realWeight*imagScaling;
                        }
                    }
                }
                q_to_j *= q;
            }
        }
    }
}

} // lagrangian_nuft
} // bfio

#endif // ifndef BFIO_LAGRANGIAN_NUFT_ADJOINT_SWITCH_TO_TARGET_INTERP_HPP
