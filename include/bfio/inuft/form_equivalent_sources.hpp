/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of ButterflyFIO and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef BFIO_INUFT_FORM_EQUIVALENT_SOURCES_HPP
#define BFIO_INUFT_FORM_EQUIVALENT_SOURCES_HPP

#include <array>
#include <cstddef>
#include <vector>

#include "bfio/constants.hpp"

#include "bfio/structures/box.hpp"
#include "bfio/structures/constrained_htree_walker.hpp"
#include "bfio/structures/plan.hpp"
#include "bfio/structures/weight_grid_list.hpp"

#include "bfio/tools/flatten_constrained_htree_index.hpp"
#include "bfio/tools/mpi.hpp"
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
void
FormEquivalentSources
( const Context<R,1,q>& context,
  const Plan<1>& plan,
  const Box<R,1>& mySBox,
  const Box<R,1>& myTBox,
  const size_t log2LocalSBoxes,
  const size_t log2LocalTBoxes,
  const array<size_t,1>& log2LocalSBoxesPerDim,
  const array<size_t,1>& log2LocalTBoxesPerDim,
        WeightGridList<R,1,q>& weightGridList )
{
    const size_t d = 1;
    const vector<R>& chebyshevNodes = context.GetChebyshevNodes();
    const vector<array<R,d>>& chebyshevGrid = context.GetChebyshevGrid();

    const Direction direction = context.GetDirection();
    const R SignedTwoPi = ( direction==FORWARD ? -TwoPi : TwoPi );

    // Store the widths of the source and target boxes
    array<R,d> wA;
    wA[0] = myTBox.widths[0] / (1<<log2LocalTBoxesPerDim[0]);
    array<R,d> wB;
    wB[0] = mySBox.widths[0] / (1<<log2LocalSBoxesPerDim[0]);

    // Iterate over the box pairs, applying M^-1 using the tensor product 
    // structure
    vector<R> realTempWeights( q ), imagTempWeights( q );
    vector<R> scalingArguments( q );
    vector<R> realPrescalings( q ), imagPrescalings( q ),
              realPostscalings( q ), imagPostscalings( q );
    ConstrainedHTreeWalker<d> AWalker( log2LocalTBoxesPerDim );
    for( size_t tIndex=0;
         tIndex<(1u<<log2LocalTBoxes); ++tIndex, AWalker.Walk() )
    {
        const array<size_t,d> A = AWalker.State();

        // Translate the local integer coordinates into the target center
        array<R,d> x0;
        x0[0] = myTBox.offsets[0] + (A[0]+0.5)*wB[0];

        // Store the chebyshev grid on A
        vector<array<R,d>> xPoints( q );
        for( size_t t=0; t<q; ++t )
            xPoints[t][0] = x0[0] + chebyshevGrid[t][0]*wA[0];

        // Compute the postscalings for all of the pairs interacting with A
        for( size_t t=0; t<q; ++t )
            scalingArguments[t] = -SignedTwoPi*x0[0]*chebyshevNodes[t]*wB[0];
        SinCosBatch( scalingArguments, imagPostscalings, realPostscalings );

        ConstrainedHTreeWalker<d> BWalker( log2LocalSBoxesPerDim );
        for( size_t sIndex=0;
             sIndex<(1u<<log2LocalSBoxes); ++sIndex, BWalker.Walk() )
        {
            const array<size_t,d> B = BWalker.State();
            const size_t iIndex = sIndex + (tIndex<<log2LocalSBoxes);
            WeightGrid<R,d,q>& weightGrid = weightGridList[iIndex];

            // Translate the local integer coordinates into the source center
            array<R,d> p0;
            p0[0] = mySBox.offsets[0] + (B[0]+0.5)*wB[0];

            //----------------------------------------------------------------//
            // Solve against the first dimension                              //
            //----------------------------------------------------------------//
            // Prescale
            for( size_t t=0; t<q; ++t )
                scalingArguments[t] = 
                    -SignedTwoPi*(x0[0]+chebyshevNodes[t]*wA[0])*p0[0];
            SinCosBatch( scalingArguments, imagPrescalings, realPrescalings );
            {
                R* realBuffer = weightGrid.RealBuffer();
                R* imagBuffer = weightGrid.ImagBuffer();
                const R* realScalingBuffer = &realPrescalings[0];
                const R* imagScalingBuffer = &imagPrescalings[0];
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
            // Transform with the inverse
            {
                const vector<R>& realInverseMap = 
                    context.GetRealInverseMap( 0 );
                const vector<R>& imagInverseMap = 
                    context.GetImagInverseMap( 0 );
                // Form the real part
                Gemv
                ( 'N', q, q,
                  R(1), &realInverseMap[0], q,
                        weightGrid.RealBuffer(), 1,
                  R(0), &realTempWeights[0], 1 );
                Gemv
                ( 'N', q, q,
                  R(-1), &imagInverseMap[0], q,
                         weightGrid.ImagBuffer(), 1,
                  R(+1), &realTempWeights[0], 1 );
                // Form the imaginary part
                Gemv
                ( 'N', q, q,
                  R(1), &realInverseMap[0], q,
                        weightGrid.ImagBuffer(), 1,
                  R(0), &imagTempWeights[0], 1 );
                Gemv
                ( 'N', q, q,
                  R(1), &imagInverseMap[0], q,
                        weightGrid.RealBuffer(), 1,
                  R(1), &imagTempWeights[0], 1 );
            }
            // Post scale
            {
                R* realWriteBuffer = weightGrid.RealBuffer();
                R* imagWriteBuffer = weightGrid.ImagBuffer();
                const R* realReadBuffer = &realTempWeights[0];
                const R* imagReadBuffer = &imagTempWeights[0];
                const R* realScalingBuffer = &realPostscalings[0];
                const R* imagScalingBuffer = &imagPostscalings[0];
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
        }
    }
}

// 2d specialization
template<typename R,size_t q>
void
FormEquivalentSources
( const Context<R,2,q>& context,
  const Plan<2>& plan,
  const Box<R,2>& mySBox,
  const Box<R,2>& myTBox,
  const size_t log2LocalSBoxes,
  const size_t log2LocalTBoxes,
  const array<size_t,2>& log2LocalSBoxesPerDim,
  const array<size_t,2>& log2LocalTBoxesPerDim,
        WeightGridList<R,2,q>& weightGridList )
{
    const size_t d = 2;
    const size_t q_to_d = Pow<q,d>::val;
    const vector<R>& chebyshevNodes = context.GetChebyshevNodes();
    const vector<array<R,d>>& chebyshevGrid = context.GetChebyshevGrid();

    const Direction direction = context.GetDirection();
    const R SignedTwoPi = ( direction==FORWARD ? -TwoPi : TwoPi );

    // Store the widths of the source and target boxes
    array<R,d> wA;
    for( size_t j=0; j<d; ++j )
        wA[j] = myTBox.widths[j] / (1<<log2LocalTBoxesPerDim[j]);
    array<R,d> wB;
    for( size_t j=0; j<d; ++j )
        wB[j] = mySBox.widths[j] / (1<<log2LocalSBoxesPerDim[j]);

    // Iterate over the box pairs, applying M^-1 using the tensor product 
    // structure
    vector<R> realTempWeights( q_to_d ), imagTempWeights( q_to_d );
    vector<R> scalingArguments( q );
    vector<R> realPrescalings( q ), imagPrescalings( q );
    array<vector<R>,d> realPostscalings, imagPostscalings;
    for( size_t j=0; j<d; ++j )
    {
        realPostscalings[j].resize(q);
        imagPostscalings[j].resize(q);
    }
    ConstrainedHTreeWalker<d> AWalker( log2LocalTBoxesPerDim );
    for( size_t tIndex=0;
         tIndex<(1u<<log2LocalTBoxes); ++tIndex, AWalker.Walk() )
    {
        const array<size_t,d> A = AWalker.State();

        // Translate the local integer coordinates into the target center
        array<R,d> x0;
        for( size_t j=0; j<d; ++j )
            x0[j] = myTBox.offsets[j] + (A[j]+0.5)*wA[j];

        // Store the chebyshev grid on A
        vector<array<R,d>> xPoints( q_to_d );
        for( size_t t=0; t<q_to_d; ++t )
            for( size_t j=0; j<d; ++j )
                xPoints[t][j] = x0[j] + chebyshevGrid[t][j]*wA[j];

        // Store the postscalings for all interactions with A
        for( size_t j=0; j<d; ++j )
        {
            for( size_t t=0; t<q; ++t )
                scalingArguments[t] = -SignedTwoPi*x0[j]*chebyshevNodes[t]*wB[j];
            SinCosBatch
            ( scalingArguments, imagPostscalings[j], realPostscalings[j] );
        }

        ConstrainedHTreeWalker<d> BWalker( log2LocalSBoxesPerDim );
        for( size_t sIndex=0;
             sIndex<(1u<<log2LocalSBoxes); ++sIndex, BWalker.Walk() )
        {
            const array<size_t,d> B = BWalker.State();
            const size_t iIndex = sIndex + (tIndex<<log2LocalSBoxes);
            WeightGrid<R,d,q>& weightGrid = weightGridList[iIndex];

            // Translate the local integer coordinates into the source center
            array<R,d> p0;
            for( size_t j=0; j<d; ++j )
                p0[j] = mySBox.offsets[j] + (B[j]+0.5)*wB[j];

            //----------------------------------------------------------------//
            // Solve against the first dimension                              //
            //----------------------------------------------------------------//
            // Prescale
            for( size_t t=0; t<q; ++t )
                scalingArguments[t] = 
                    -SignedTwoPi*(x0[0]+chebyshevNodes[t]*wA[0])*p0[0];
            SinCosBatch( scalingArguments, imagPrescalings, realPrescalings );
            {
                R* realBuffer = weightGrid.RealBuffer();
                R* imagBuffer = weightGrid.ImagBuffer();
                const R* realScalingBuffer = &realPrescalings[0];
                const R* imagScalingBuffer = &imagPrescalings[0];
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
            // Transform with the inverse
            {
                const vector<R>& realInverseMap = 
                    context.GetRealInverseMap( 0 );
                const vector<R>& imagInverseMap = 
                    context.GetImagInverseMap( 0 );
                // Form the real part
                Gemm
                ( 'N', 'N', q, q, q,
                  R(1), &realInverseMap[0], q,
                        weightGrid.RealBuffer(), q,
                  R(0), &realTempWeights[0], q );
                Gemm
                ( 'N', 'N', q, q, q,
                  R(-1), &imagInverseMap[0], q,
                         weightGrid.ImagBuffer(), q,
                  R(+1), &realTempWeights[0], q );
                // Form the imaginary part
                Gemm
                ( 'N', 'N', q, q, q,
                  R(1), &realInverseMap[0], q,
                        weightGrid.ImagBuffer(), q,
                  R(0), &imagTempWeights[0], q );
                Gemm
                ( 'N', 'N', q, q, q,
                 R(1), &imagInverseMap[0], q,
                       weightGrid.RealBuffer(), q,
                 R(1), &imagTempWeights[0], q );
            }
            // Post scale
            {
                R* realBuffer = &realTempWeights[0];
                R* imagBuffer = &imagTempWeights[0];
                const R* realScalingBuffer = &realPostscalings[0][0];
                const R* imagScalingBuffer = &imagPostscalings[0][0];
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

            //----------------------------------------------------------------//
            // Solve against the second dimension                             //
            //----------------------------------------------------------------//
            // Prescale
            for( size_t t=0; t<q; ++t )
                scalingArguments[t] = 
                    -SignedTwoPi*(x0[1]+chebyshevNodes[t]*wA[1])*p0[1];
            SinCosBatch( scalingArguments, imagPrescalings, realPrescalings );
            {
                R* realBuffer = &realTempWeights[0];
                R* imagBuffer = &imagTempWeights[0];
                const R* realScalingBuffer = &realPrescalings[0];
                const R* imagScalingBuffer = &imagPrescalings[0];
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
            // Transform with the inverse
            {
                const vector<R>& realInverseMap = 
                    context.GetRealInverseMap( 1 );
                const vector<R>& imagInverseMap = 
                    context.GetImagInverseMap( 1 );
                // Form the real part
                Gemm
                ( 'N', 'T', q, q, q,
                  R(1), &realTempWeights[0], q,
                        &realInverseMap[0], q,
                  R(0), weightGrid.RealBuffer(), q );
                Gemm
                ( 'N', 'T', q, q, q,
                  R(-1), &imagTempWeights[0], q,
                         &imagInverseMap[0], q,
                  R(+1), weightGrid.RealBuffer(), q );
                // Form the imaginary part
                Gemm
                ( 'N', 'T', q, q, q,
                  R(1), &imagTempWeights[0], q,
                        &realInverseMap[0], q,
                  R(0), weightGrid.ImagBuffer(), q );
                Gemm
                ( 'N', 'T', q, q, q,
                  R(1), &realTempWeights[0], q,
                        &imagInverseMap[0], q,
                  R(1), weightGrid.ImagBuffer(), q );
            }
            // Postscale
            {
                R* realBuffer = weightGrid.RealBuffer();
                R* imagBuffer = weightGrid.ImagBuffer();
                const R* realScalingBuffer = &realPostscalings[1][0];
                const R* imagScalingBuffer = &imagPostscalings[1][0];
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
        }
    }
}

// Fallback for 3d and above
template<typename R,size_t d,size_t q>
void
FormEquivalentSources
( const Context<R,d,q>& context,
  const Plan<d>& plan,
  const Box<R,d>& mySBox,
  const Box<R,d>& myTBox,
  const size_t log2LocalSBoxes,
  const size_t log2LocalTBoxes,
  const array<size_t,d>& log2LocalSBoxesPerDim,
  const array<size_t,d>& log2LocalTBoxesPerDim,
        WeightGridList<R,d,q>& weightGridList )
{
    const size_t q_to_d = Pow<q,d>::val;
    const vector<R>& chebyshevNodes = context.GetChebyshevNodes();
    const vector<array<R,d>>& chebyshevGrid = context.GetChebyshevGrid();

    const Direction direction = context.GetDirection();
    const R SignedTwoPi = ( direction==FORWARD ? -TwoPi : TwoPi );

    // Store the widths of the source and target boxes
    array<R,d> wA;
    for( size_t j=0; j<d; ++j )
        wA[j] = myTBox.widths[j] / (1<<log2LocalTBoxesPerDim[j]);
    array<R,d> wB;
    for( size_t j=0; j<d; ++j )
        wB[j] = mySBox.widths[j] / (1<<log2LocalSBoxesPerDim[j]);

    // Iterate over the box pairs, applying M^-1 using the tensor product 
    // structure
    vector<R> scalingArguments( q );
    vector<R> realPrescalings( q ), imagPrescalings( q );
    array<vector<R>,d> realPostscalings, imagPostscalings;
    for( size_t j=0; j<d; ++j )
    {
        realPostscalings[j].resize(q);
        imagPostscalings[j].resize(q);
    }
    vector<R> realTempWeights0( q_to_d ), imagTempWeights0( q_to_d ),
              realTempWeights1( q_to_d ), imagTempWeights1( q_to_d );
    ConstrainedHTreeWalker<d> AWalker( log2LocalTBoxesPerDim );
    for( size_t tIndex=0;
         tIndex<(1u<<log2LocalTBoxes); ++tIndex, AWalker.Walk() )
    {
        const array<size_t,d> A = AWalker.State();

        // Translate the local integer coordinates into the target center
        array<R,d> x0;
        for( size_t j=0; j<d; ++j )
            x0[j] = myTBox.offsets[j] + (A[j]+0.5)*wA[j];

        // Store the chebyshev grid on A
        vector<array<R,d>> xPoints( q_to_d );
        for( size_t t=0; t<q_to_d; ++t )
            for( size_t j=0; j<d; ++j )
                xPoints[t][j] = x0[j] + chebyshevGrid[t][j]*wA[j];

        // Store the postscalings for all interactions with A
        for( size_t j=0; j<d; ++j )
        {
            for( size_t t=0; t<q; ++t )
                scalingArguments[t] = -SignedTwoPi*x0[j]*chebyshevNodes[t]*wB[j];
            SinCosBatch
            ( scalingArguments, imagPostscalings[j], realPostscalings[j] );
        }

        ConstrainedHTreeWalker<d> BWalker( log2LocalSBoxesPerDim );
        for( size_t sIndex=0;
             sIndex<(1u<<log2LocalSBoxes); ++sIndex, BWalker.Walk() )
        {
            const array<size_t,d> B = BWalker.State();
            const size_t iIndex = sIndex + (tIndex<<log2LocalSBoxes);
            WeightGrid<R,d,q>& weightGrid = weightGridList[iIndex];

            // Translate the local integer coordinates into the source center
            array<R,d> p0;
            for( size_t j=0; j<d; ++j )
                p0[j] = mySBox.offsets[j] + (B[j]+0.5)*wB[j];

            //----------------------------------------------------------------//
            // Solve against the first dimension                              //
            //----------------------------------------------------------------//
            // Prescale
            for( size_t t=0; t<q; ++t )
                scalingArguments[t] = 
                    -SignedTwoPi*(x0[0]+chebyshevNodes[t]*wA[0])*p0[0];
            SinCosBatch( scalingArguments, imagPrescalings, realPrescalings );
            {
                R* realBuffer = weightGrid.RealBuffer();
                R* imagBuffer = weightGrid.ImagBuffer();
                const R* realScalingBuffer = &realPrescalings[0];
                const R* imagScalingBuffer = &imagPrescalings[0];
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
            // Transform with the inverse
            {
                const vector<R>& realInverseMap = 
                    context.GetRealInverseMap( 0 );
                const vector<R>& imagInverseMap = 
                    context.GetImagInverseMap( 0 );
                // Form the real part
                Gemm
                ( 'N', 'N', q, Pow<q,d-1>::val, q,
                  R(1), &realInverseMap[0], q,
                        weightGrid.RealBuffer(), q,
                  R(0), &realTempWeights0[0], q );
                Gemm
                ( 'N', 'N', q, Pow<q,d-1>::val, q,
                  R(-1), &imagInverseMap[0], q,
                         weightGrid.ImagBuffer(), q,
                  R(+1), &realTempWeights0[0], q );
                // Form the imaginary part
                Gemm
                ( 'N', 'N', q, Pow<q,d-1>::val, q,
                  R(1), &realInverseMap[0], q,
                        weightGrid.ImagBuffer(), q,
                  R(0), &imagTempWeights0[0], q );
                Gemm
                ( 'N', 'N', q, Pow<q,d-1>::val, q,
                 R(1), &imagInverseMap[0], q,
                       weightGrid.RealBuffer(), q,
                 R(1), &imagTempWeights0[0], q );
            }
            // Post scale
            {
                R* realBuffer = &realTempWeights0[0];
                R* imagBuffer = &imagTempWeights0[0];
                const R* realScalingBuffer = &realPostscalings[0][0];
                const R* imagScalingBuffer = &imagPostscalings[0][0];
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

            //----------------------------------------------------------------//
            // Solve against the second dimension                             //
            //----------------------------------------------------------------//
            // Prescale
            for( size_t t=0; t<q; ++t )
                scalingArguments[t] = 
                    -SignedTwoPi*(x0[1]+chebyshevNodes[t]*wA[1])*p0[1];
            SinCosBatch( scalingArguments, imagPrescalings, realPrescalings );
            for( size_t p=0; p<Pow<q,d-2>::val; ++p )
            {
                const size_t offset = p*q*q;
                R* offsetRealBuffer = &realTempWeights0[offset];
                R* offsetImagBuffer = &imagTempWeights0[offset];
                const R* realScalingBuffer = &realPrescalings[0];
                const R* imagScalingBuffer = &imagPrescalings[0];
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
            // Transform with the inverse
            {
                const vector<R>& realInverseMap = 
                    context.GetRealInverseMap( 1 );
                const vector<R>& imagInverseMap = 
                    context.GetImagInverseMap( 1 );
                for( size_t w=0; w<Pow<q,d-2>::val; ++w )
                {
                    // Form the real part
                    Gemm
                    ( 'N', 'T', q, q, q,
                      R(1), &realTempWeights0[w*q*q], q,
                            &realInverseMap[0], q,
                      R(0), &realTempWeights1[w*q*q], q );
                    Gemm
                    ( 'N', 'T', q, q, q,
                      R(-1), &imagTempWeights0[w*q*q], q,
                             &imagInverseMap[0], q,
                      R(+1), &realTempWeights1[w*q*q], q );

                    // Form the imaginary part
                    Gemm
                    ( 'N', 'T', q, q, q,
                      R(1), &imagTempWeights0[w*q*q], q,
                            &realInverseMap[0], q,
                      R(0), &imagTempWeights1[w*q*q], q );
                    Gemm
                    ( 'N', 'T', q, q, q,
                      R(1), &realTempWeights0[w*q*q], q,
                            &imagInverseMap[0], q,
                      R(1), &imagTempWeights1[w*q*q], q );
                }
            }
            // Postscale
            for( size_t p=0; p<Pow<q,d-2>::val; ++p )
            {
                const size_t offset = p*q*q;
                R* offsetRealBuffer = &realTempWeights1[offset];
                R* offsetImagBuffer = &imagTempWeights1[offset];
                const R* realScalingBuffer = &realPostscalings[1][0];
                const R* imagScalingBuffer = &imagPostscalings[1][0];
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

            //----------------------------------------------------------------//
            // Solve against the remaining dimensions                         //
            //----------------------------------------------------------------//
            size_t q_to_j = q*q;
            for( size_t j=2; j<d; ++j )
            {
                const size_t stride = q_to_j;

                R* realWriteBuffer = 
                    ( j==d-1 ? weightGrid.RealBuffer()
                             : ( j&1 ? &realTempWeights1[0]
                                     : &realTempWeights0[0] ) );
                R* imagWriteBuffer = 
                    ( j==d-1 ? weightGrid.ImagBuffer()
                             : ( j&1 ? &imagTempWeights1[0]
                                     : &imagTempWeights0[0] ) );
                R* realReadBuffer = 
                    ( j&1 ? &realTempWeights0[0] : &realTempWeights1[0] );
                R* imagReadBuffer = 
                    ( j&1 ? &imagTempWeights0[0] : &imagTempWeights1[0] );

                const vector<R>& realInverseMap = 
                    context.GetRealInverseMap( j );
                const vector<R>& imagInverseMap = 
                    context.GetImagInverseMap( j );
                const R* realInverseBuffer = &realInverseMap[0];
                const R* imagInverseBuffer = &imagInverseMap[0];

                memset( realWriteBuffer, 0, q_to_d*sizeof(R) );
                memset( imagWriteBuffer, 0, q_to_d*sizeof(R) );

                // Prescale, transform, and postscale
                for( size_t t=0; t<q; ++t )
                    scalingArguments[t] = 
                        -SignedTwoPi*(x0[j]+chebyshevNodes[t]*wA[j])*p0[j];
                SinCosBatch
                ( scalingArguments, imagPrescalings, realPrescalings );
                for( size_t p=0; p<q_to_d/(q_to_j*q); ++p )
                {
                    const size_t offset = p*(q_to_j*q);
                    R* offsetRealReadBuffer = &realReadBuffer[offset];
                    R* offsetImagReadBuffer = &imagReadBuffer[offset];
                    R* offsetRealWriteBuffer = &realWriteBuffer[offset];
                    R* offsetImagWriteBuffer = &imagWriteBuffer[offset];
                    const R* realPrescalingBuffer = &realPrescalings[0];
                    const R* imagPrescalingBuffer = &imagPrescalings[0];
                    const R* realPostscalingBuffer = &realPostscalings[j][0];
                    const R* imagPostscalingBuffer = &imagPostscalings[j][0];
                    for( size_t w=0; w<q_to_j; ++w )
                    {
                        // Prescale
                        for( size_t t=0; t<q; ++t )
                        {
                            const R realWeight = 
                                offsetRealReadBuffer[w+t*stride];
                            const R imagWeight = 
                                offsetImagReadBuffer[w+t*stride];
                            const R realScaling = realPrescalingBuffer[t];
                            const R imagScaling = imagPrescalingBuffer[t];
                            offsetRealReadBuffer[w+t*stride] = 
                                realWeight*realScaling - imagWeight*imagScaling;
                            offsetImagReadBuffer[w+t*stride] = 
                                imagWeight*realScaling + realWeight*imagScaling;
                        }
                        // Transform
                        for( size_t t=0; t<q; ++t )
                        {
                            for( size_t tPrime=0; tPrime<q; ++tPrime )
                            {
                                offsetRealWriteBuffer[w+t*stride] +=
                                    realInverseBuffer[t+tPrime*q] * 
                                    offsetRealReadBuffer[w+tPrime*stride];
                                offsetRealWriteBuffer[w+t*stride] -=
                                    imagInverseBuffer[t+tPrime*q] *
                                    offsetImagReadBuffer[w+tPrime*stride];

                                offsetImagWriteBuffer[w+t*stride] +=
                                    imagInverseBuffer[t+tPrime*q] *
                                    offsetRealReadBuffer[w+tPrime*stride];
                                offsetImagWriteBuffer[w+t*stride] +=
                                    realInverseBuffer[t+tPrime*q] *
                                    offsetImagReadBuffer[w+tPrime*stride];
                            }
                        }
                        // Postscale
                        for( size_t t=0; t<q; ++t )
                        {
                            const R realWeight = 
                                offsetRealWriteBuffer[w+t*stride];
                            const R imagWeight = 
                                offsetImagWriteBuffer[w+t*stride];
                            const R realScaling = realPostscalingBuffer[t];
                            const R imagScaling = imagPostscalingBuffer[t];
                            offsetRealWriteBuffer[w+t*stride] = 
                                realWeight*realScaling - imagWeight*imagScaling;
                            offsetImagWriteBuffer[w+t*stride] = 
                                imagWeight*realScaling + realWeight*imagScaling;
                        }
                    }
                }
                q_to_j *= q;
            }
        }
    }
}

} // inuft
} // bfio

#endif // ifndef BFIO_INUFT_FORM_EQUIVALENT_SOURCES_HPP 
