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
#ifndef BFIO_INTERPOLATIVE_NUFT_FORM_EQUIVALENT_SOURCES_HPP
#define BFIO_INTERPOLATIVE_NUFT_FORM_EQUIVALENT_SOURCES_HPP 1

#include <cstddef>
#include <vector>

#include "bfio/constants.hpp"

#include "bfio/structures/array.hpp"
#include "bfio/structures/box.hpp"
#include "bfio/structures/constrained_htree_walker.hpp"
#include "bfio/structures/plan.hpp"
#include "bfio/structures/weight_grid_list.hpp"

#include "bfio/tools/flatten_constrained_htree_index.hpp"
#include "bfio/tools/mpi.hpp"
#include "bfio/tools/special_functions.hpp"

#include "bfio/interpolative_nuft/context.hpp"

namespace bfio {
namespace interpolative_nuft {

// 1d specialization
template<typename R,std::size_t q>
void
FormEquivalentSources
( const interpolative_nuft::Context<R,1,q>& context,
  const Plan<1>& plan,
  const Box<R,1>& mySourceBox,
  const Box<R,1>& myTargetBox,
  const std::size_t log2LocalSourceBoxes,
  const std::size_t log2LocalTargetBoxes,
  const Array<std::size_t,1>& log2LocalSourceBoxesPerDim,
  const Array<std::size_t,1>& log2LocalTargetBoxesPerDim,
        WeightGridList<R,1,q>& weightGridList )
{
    const std::size_t d = 1;
    const std::vector<R>& chebyshevNodes = context.GetChebyshevNodes();
    const std::vector< Array<R,d> >& chebyshevGrid = context.GetChebyshevGrid();

    // Store the widths of the source and target boxes
    Array<R,d> wA;
    wA[0] = myTargetBox.widths[0] / (1<<log2LocalTargetBoxesPerDim[0]);
    Array<R,d> wB;
    wB[0] = mySourceBox.widths[0] / (1<<log2LocalSourceBoxesPerDim[0]);

    // Iterate over the box pairs, applying M^-1 using the tensor product 
    // structure
    std::vector<R> realTempWeights( q );
    std::vector<R> imagTempWeights( q );
    ConstrainedHTreeWalker<d> AWalker( log2LocalTargetBoxesPerDim );
    for( std::size_t targetIndex=0;
         targetIndex<(1u<<log2LocalTargetBoxes);
         ++targetIndex, AWalker.Walk() )
    {
        const Array<std::size_t,d> A = AWalker.State();

        // Translate the local integer coordinates into the target center
        Array<R,d> x0;
        x0[0] = myTargetBox.offsets[0] + (A[0]+0.5)*wB[0];

        // Store the chebyshev grid on A
        std::vector< Array<R,d> > xPoints( q );
        for( std::size_t t=0; t<q; ++t )
            xPoints[t][0] = x0[0] + chebyshevGrid[t][0]*wA[0];

        std::vector<R> scalingArguments( q );
        std::vector<R> realScalings( q );
        std::vector<R> imagScalings( q );
        std::vector<R> realTempWeights( q );
        std::vector<R> imagTempWeights( q );
        ConstrainedHTreeWalker<d> BWalker( log2LocalSourceBoxesPerDim );
        for( std::size_t sourceIndex=0;
             sourceIndex<(1u<<log2LocalSourceBoxes);
             ++sourceIndex, BWalker.Walk() )
        {
            const Array<std::size_t,d> B = BWalker.State();
            const std::size_t interactionIndex = 
                sourceIndex + (targetIndex<<log2LocalSourceBoxes);
            WeightGrid<R,d,q>& weightGrid = weightGridList[interactionIndex];

            // Translate the local integer coordinates into the source center
            Array<R,d> p0;
            p0[0] = mySourceBox.offsets[0] + (B[0]+0.5)*wB[0];

            //----------------------------------------------------------------//
            // Solve against the first dimension                              //
            //----------------------------------------------------------------//
            // Prescale
            for( std::size_t t=0; t<q; ++t )
                scalingArguments[t] = 
                    -TwoPi*(x0[0]+chebyshevNodes[t]*wA[0])*p0[0];
            SinCosBatch( scalingArguments, imagScalings, realScalings );
            {
                R* realBuffer = weightGrid.RealBuffer();
                R* imagBuffer = weightGrid.ImagBuffer();
                const R* realScalingBuffer = &realScalings[0];
                const R* imagScalingBuffer = &imagScalings[0];
                for( std::size_t t=0; t<q; ++t )
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
                const std::vector<R>& realInverseMap = 
                    context.GetRealInverseMap( 0 );
                const std::vector<R>& imagInverseMap = 
                    context.GetImagInverseMap( 0 );
                // Form the real part
                Gemv
                ( 'N', q, q,
                  (R)1, &realInverseMap[0], q,
                        weightGrid.RealBuffer(), 1,
                  (R)0, &realTempWeights[0], 1 );
                Gemv
                ( 'N', q, q,
                  (R)-1, &imagInverseMap[0], q,
                         weightGrid.ImagBuffer(), 1,
                  (R)+1, &realTempWeights[0], 1 );
                // Form the imaginary part
                Gemv
                ( 'N', q, q,
                  (R)1, &realInverseMap[0], q,
                        weightGrid.ImagBuffer(), 1,
                  (R)0, &imagTempWeights[0], 1 );
                Gemv
                ( 'N', q, q,
                  (R)1, &imagInverseMap[0], q,
                        weightGrid.RealBuffer(), 1,
                  (R)1, &imagTempWeights[0], 1 );
            }
            // Post scale
            for( std::size_t t=0; t<q; ++t )
                scalingArguments[t] = -TwoPi*x0[0]*chebyshevNodes[t]*wB[0];
            SinCosBatch( scalingArguments, imagScalings, realScalings );
            {
                R* realWriteBuffer = weightGrid.RealBuffer();
                R* imagWriteBuffer = weightGrid.ImagBuffer();
                const R* realReadBuffer = &realTempWeights[0];
                const R* imagReadBuffer = &imagTempWeights[0];
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
        }
    }
}

// 2d specialization
template<typename R,std::size_t q>
void
FormEquivalentSources
( const interpolative_nuft::Context<R,2,q>& context,
  const Plan<2>& plan,
  const Box<R,2>& mySourceBox,
  const Box<R,2>& myTargetBox,
  const std::size_t log2LocalSourceBoxes,
  const std::size_t log2LocalTargetBoxes,
  const Array<std::size_t,2>& log2LocalSourceBoxesPerDim,
  const Array<std::size_t,2>& log2LocalTargetBoxesPerDim,
        WeightGridList<R,2,q>& weightGridList )
{
    const std::size_t d = 2;
    const std::size_t q_to_d = Pow<q,d>::val;
    const std::vector<R>& chebyshevNodes = context.GetChebyshevNodes();
    const std::vector< Array<R,d> >& chebyshevGrid = context.GetChebyshevGrid();

    // Store the widths of the source and target boxes
    Array<R,d> wA;
    for( std::size_t j=0; j<d; ++j )
        wA[j] = myTargetBox.widths[j] / (1<<log2LocalTargetBoxesPerDim[j]);
    Array<R,d> wB;
    for( std::size_t j=0; j<d; ++j )
        wB[j] = mySourceBox.widths[j] / (1<<log2LocalSourceBoxesPerDim[j]);

    // Iterate over the box pairs, applying M^-1 using the tensor product 
    // structure
    std::vector<R> scalingArguments( q );
    std::vector<R> realScalings( q );
    std::vector<R> imagScalings( q );
    std::vector<R> realTempWeights( q_to_d );
    std::vector<R> imagTempWeights( q_to_d );
    ConstrainedHTreeWalker<d> AWalker( log2LocalTargetBoxesPerDim );
    for( std::size_t targetIndex=0;
         targetIndex<(1u<<log2LocalTargetBoxes);
         ++targetIndex, AWalker.Walk() )
    {
        const Array<std::size_t,d> A = AWalker.State();

        // Translate the local integer coordinates into the target center
        Array<R,d> x0;
        for( std::size_t j=0; j<d; ++j )
            x0[j] = myTargetBox.offsets[j] + (A[j]+0.5)*wA[j];

        // Store the chebyshev grid on A
        std::vector< Array<R,d> > xPoints( q_to_d );
        for( std::size_t t=0; t<q_to_d; ++t )
            for( std::size_t j=0; j<d; ++j )
                xPoints[t][j] = x0[j] + chebyshevGrid[t][j]*wA[j];

        ConstrainedHTreeWalker<d> BWalker( log2LocalSourceBoxesPerDim );
        for( std::size_t sourceIndex=0;
             sourceIndex<(1u<<log2LocalSourceBoxes);
             ++sourceIndex, BWalker.Walk() )
        {
            const Array<std::size_t,d> B = BWalker.State();
            const std::size_t interactionIndex = 
                sourceIndex + (targetIndex<<log2LocalSourceBoxes);
            WeightGrid<R,d,q>& weightGrid = weightGridList[interactionIndex];

            // Translate the local integer coordinates into the source center
            Array<R,d> p0;
            for( std::size_t j=0; j<d; ++j )
                p0[j] = mySourceBox.offsets[j] + (B[j]+0.5)*wB[j];

            //----------------------------------------------------------------//
            // Solve against the first dimension                              //
            //----------------------------------------------------------------//
            // Prescale
            for( std::size_t t=0; t<q; ++t )
                scalingArguments[t] = 
                    -TwoPi*(x0[0]+chebyshevNodes[t]*wA[0])*p0[0];
            SinCosBatch( scalingArguments, imagScalings, realScalings );
            {
                R* realBuffer = weightGrid.RealBuffer();
                R* imagBuffer = weightGrid.ImagBuffer();
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
            // Transform with the inverse
            {
                const std::vector<R>& realInverseMap = 
                    context.GetRealInverseMap( 0 );
                const std::vector<R>& imagInverseMap = 
                    context.GetImagInverseMap( 0 );
                // Form the real part
                Gemm
                ( 'N', 'N', q, q, q,
                  (R)1, &realInverseMap[0], q,
                        weightGrid.RealBuffer(), q,
                  (R)0, &realTempWeights[0], q );
                Gemm
                ( 'N', 'N', q, q, q,
                  (R)-1, &imagInverseMap[0], q,
                         weightGrid.ImagBuffer(), q,
                  (R)+1, &realTempWeights[0], q );
                // Form the imaginary part
                Gemm
                ( 'N', 'N', q, q, q,
                  (R)1, &realInverseMap[0], q,
                        weightGrid.ImagBuffer(), q,
                  (R)0, &imagTempWeights[0], q );
                Gemm
                ( 'N', 'N', q, q, q,
                 (R)1, &imagInverseMap[0], q,
                       weightGrid.RealBuffer(), q,
                 (R)1, &imagTempWeights[0], q );
            }
            // Post scale
            for( std::size_t t=0; t<q; ++t )
                scalingArguments[t] = -TwoPi*x0[0]*chebyshevNodes[t]*wB[0];
            SinCosBatch( scalingArguments, imagScalings, realScalings );
            {
                R* realBuffer = &realTempWeights[0];
                R* imagBuffer = &imagTempWeights[0];
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

            //----------------------------------------------------------------//
            // Solve against the second dimension                             //
            //----------------------------------------------------------------//
            // Prescale
            for( std::size_t t=0; t<q; ++t )
                scalingArguments[t] = 
                    -TwoPi*(x0[1]+chebyshevNodes[t]*wA[1])*p0[1];
            SinCosBatch( scalingArguments, imagScalings, realScalings );
            {
                R* realBuffer = &realTempWeights[0];
                R* imagBuffer = &imagTempWeights[0];
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
            // Transform with the inverse
            {
                const std::vector<R>& realInverseMap = 
                    context.GetRealInverseMap( 1 );
                const std::vector<R>& imagInverseMap = 
                    context.GetImagInverseMap( 1 );
                // Form the real part
                Gemm
                ( 'N', 'T', q, q, q,
                  (R)1, &realTempWeights[0], q,
                        &realInverseMap[0], q,
                  (R)0, weightGrid.RealBuffer(), q );
                Gemm
                ( 'N', 'T', q, q, q,
                  (R)-1, &imagTempWeights[0], q,
                         &imagInverseMap[0], q,
                  (R)+1, weightGrid.RealBuffer(), q );
                // Form the imaginary part
                Gemm
                ( 'N', 'T', q, q, q,
                  (R)1, &imagTempWeights[0], q,
                        &realInverseMap[0], q,
                  (R)0, weightGrid.ImagBuffer(), q );
                Gemm
                ( 'N', 'T', q, q, q,
                  (R)1, &realTempWeights[0], q,
                        &imagInverseMap[0], q,
                  (R)1, weightGrid.ImagBuffer(), q );
            }
            // Postscale
            for( std::size_t t=0; t<q; ++t )
                scalingArguments[t] = -TwoPi*x0[1]*chebyshevNodes[t]*wB[1];
            SinCosBatch( scalingArguments, imagScalings, realScalings );
            {
                R* realBuffer = weightGrid.RealBuffer();
                R* imagBuffer = weightGrid.ImagBuffer();
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
        }
    }
}

// Fallback for 3d and above
template<typename R,std::size_t d,std::size_t q>
void
FormEquivalentSources
( const interpolative_nuft::Context<R,d,q>& context,
  const Plan<d>& plan,
  const Box<R,d>& mySourceBox,
  const Box<R,d>& myTargetBox,
  const std::size_t log2LocalSourceBoxes,
  const std::size_t log2LocalTargetBoxes,
  const Array<std::size_t,d>& log2LocalSourceBoxesPerDim,
  const Array<std::size_t,d>& log2LocalTargetBoxesPerDim,
        WeightGridList<R,d,q>& weightGridList )
{
    const std::size_t q_to_d = Pow<q,d>::val;
    const std::vector<R>& chebyshevNodes = context.GetChebyshevNodes();
    const std::vector< Array<R,d> >& chebyshevGrid = context.GetChebyshevGrid();

    // Store the widths of the source and target boxes
    Array<R,d> wA;
    for( std::size_t j=0; j<d; ++j )
        wA[j] = myTargetBox.widths[j] / (1<<log2LocalTargetBoxesPerDim[j]);;
    Array<R,d> wB;
    for( std::size_t j=0; j<d; ++j )
        wB[j] = mySourceBox.widths[j] / (1<<log2LocalSourceBoxesPerDim[j]);

    // Iterate over the box pairs, applying M^-1 using the tensor product 
    // structure
    std::vector<R> scalingArguments( q );
    std::vector<R> realScalings( q );
    std::vector<R> imagScalings( q );
    std::vector<R> realTempWeights0( q_to_d );
    std::vector<R> imagTempWeights0( q_to_d );
    std::vector<R> realTempWeights1( q_to_d );
    std::vector<R> imagTempWeights1( q_to_d );
    ConstrainedHTreeWalker<d> AWalker( log2LocalTargetBoxesPerDim );
    for( std::size_t targetIndex=0;
         targetIndex<(1u<<log2LocalTargetBoxes);
         ++targetIndex, AWalker.Walk() )
    {
        const Array<std::size_t,d> A = AWalker.State();

        // Translate the local integer coordinates into the target center
        Array<R,d> x0;
        for( std::size_t j=0; j<d; ++j )
            x0[j] = myTargetBox.offsets[j] + (A[j]+0.5)*wA[j];

        // Store the chebyshev grid on A
        std::vector< Array<R,d> > xPoints( q_to_d );
        for( std::size_t t=0; t<q_to_d; ++t )
            for( std::size_t j=0; j<d; ++j )
                xPoints[t][j] = x0[j] + chebyshevGrid[t][j]*wA[j];

        ConstrainedHTreeWalker<d> BWalker( log2LocalSourceBoxesPerDim );
        for( std::size_t sourceIndex=0;
             sourceIndex<(1u<<log2LocalSourceBoxes);
             ++sourceIndex, BWalker.Walk() )
        {
            const Array<std::size_t,d> B = BWalker.State();
            const std::size_t interactionIndex = 
                sourceIndex + (targetIndex<<log2LocalSourceBoxes);
            WeightGrid<R,d,q>& weightGrid = weightGridList[interactionIndex];

            // Translate the local integer coordinates into the source center
            Array<R,d> p0;
            for( std::size_t j=0; j<d; ++j )
                p0[j] = mySourceBox.offsets[j] + (B[j]+0.5)*wB[j];

            //----------------------------------------------------------------//
            // Solve against the first dimension                              //
            //----------------------------------------------------------------//
            // Prescale
            for( std::size_t t=0; t<q; ++t )
                scalingArguments[t] = 
                    -TwoPi*(x0[0]+chebyshevNodes[t]*wA[0])*p0[0];
            SinCosBatch( scalingArguments, imagScalings, realScalings );
            {
                R* realBuffer = weightGrid.RealBuffer();
                R* imagBuffer = weightGrid.ImagBuffer();
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
            // Transform with the inverse
            {
                const std::vector<R>& realInverseMap = 
                    context.GetRealInverseMap( 0 );
                const std::vector<R>& imagInverseMap = 
                    context.GetImagInverseMap( 0 );
                // Form the real part
                Gemm
                ( 'N', 'N', q, Pow<q,d-1>::val, q,
                  (R)1, &realInverseMap[0], q,
                        weightGrid.RealBuffer(), q,
                  (R)0, &realTempWeights0[0], q );
                Gemm
                ( 'N', 'N', q, Pow<q,d-1>::val, q,
                  (R)-1, &imagInverseMap[0], q,
                         weightGrid.ImagBuffer(), q,
                  (R)+1, &realTempWeights0[0], q );
                // Form the imaginary part
                Gemm
                ( 'N', 'N', q, Pow<q,d-1>::val, q,
                  (R)1, &realInverseMap[0], q,
                        weightGrid.ImagBuffer(), q,
                  (R)0, &imagTempWeights0[0], q );
                Gemm
                ( 'N', 'N', q, Pow<q,d-1>::val, q,
                 (R)1, &imagInverseMap[0], q,
                       weightGrid.RealBuffer(), q,
                 (R)1, &imagTempWeights0[0], q );
            }
            // Post scale
            for( std::size_t t=0; t<q; ++t )
                scalingArguments[t] = -TwoPi*x0[0]*chebyshevNodes[t]*wB[0];
            SinCosBatch( scalingArguments, imagScalings, realScalings );
            {
                R* realBuffer = &realTempWeights0[0];
                R* imagBuffer = &imagTempWeights0[0];
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

            //----------------------------------------------------------------//
            // Solve against the second dimension                             //
            //----------------------------------------------------------------//
            // Prescale
            for( std::size_t t=0; t<q; ++t )
                scalingArguments[t] = 
                    -TwoPi*(x0[1]+chebyshevNodes[t]*wA[1])*p0[1];
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
            // Transform with the inverse
            {
                const std::vector<R>& realInverseMap = 
                    context.GetRealInverseMap( 1 );
                const std::vector<R>& imagInverseMap = 
                    context.GetImagInverseMap( 1 );
                for( std::size_t w=0; w<Pow<q,d-2>::val; ++w )
                {
                    // Form the real part
                    Gemm
                    ( 'N', 'T', q, q, q,
                      (R)1, &realTempWeights0[w*q*q], q,
                            &realInverseMap[0], q,
                      (R)0, &realTempWeights1[w*q*q], q );
                    Gemm
                    ( 'N', 'T', q, q, q,
                      (R)-1, &imagTempWeights0[w*q*q], q,
                             &imagInverseMap[0], q,
                      (R)+1, &realTempWeights1[w*q*q], q );

                    // Form the imaginary part
                    Gemm
                    ( 'N', 'T', q, q, q,
                      (R)1, &imagTempWeights0[w*q*q], q,
                            &realInverseMap[0], q,
                      (R)0, &imagTempWeights1[w*q*q], q );
                    Gemm
                    ( 'N', 'T', q, q, q,
                      (R)1, &realTempWeights0[w*q*q], q,
                            &imagInverseMap[0], q,
                      (R)1, &imagTempWeights1[w*q*q], q );
                }
            }
            // Postscale
            for( std::size_t t=0; t<q; ++t )
                scalingArguments[t] = -TwoPi*x0[1]*chebyshevNodes[t]*wB[1];
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

            //----------------------------------------------------------------//
            // Solve against the remaining dimensions                         //
            //----------------------------------------------------------------//
            std::size_t q_to_j = q*q;
            for( std::size_t j=2; j<d; ++j )
            {
                const std::size_t stride = q_to_j;

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

                const std::vector<R>& realInverseMap = 
                    context.GetRealInverseMap( j );
                const std::vector<R>& imagInverseMap = 
                    context.GetImagInverseMap( j );
                const R* realInverseBuffer = &realInverseMap[0];
                const R* imagInverseBuffer = &imagInverseMap[0];

                std::memset( realWriteBuffer, 0, q_to_d*sizeof(R) );
                std::memset( imagWriteBuffer, 0, q_to_d*sizeof(R) );

                // Prescale, transform, and postscale
                std::vector<R> realPrescalings( q );
                std::vector<R> imagPrescalings( q );
                std::vector<R> realPostscalings( q );
                std::vector<R> imagPostscalings( q );
                for( std::size_t t=0; t<q; ++t )
                    scalingArguments[t] = 
                        -TwoPi*(x0[j]+chebyshevNodes[t]*wA[j])*p0[j];
                SinCosBatch
                ( scalingArguments, imagPrescalings, realPrescalings );
                for( std::size_t t=0; t<q; ++t )
                    scalingArguments[t] = 
                        -TwoPi*x0[j]*chebyshevNodes[t]*wB[j];
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
                        for( std::size_t t=0; t<q; ++t )
                        {
                            for( std::size_t tPrime=0; tPrime<q; ++tPrime )
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
                        for( std::size_t t=0; t<q; ++t )
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

} // interpolative_nuft
} // bfio

#endif // BFIO_INTERPOLATIVE_NUFT_FORM_EQUIVALENT_SOURCES_HPP 

