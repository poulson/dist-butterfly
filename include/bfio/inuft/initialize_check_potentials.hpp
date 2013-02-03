/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of ButterflyFIO and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef BFIO_INUFT_INITIALIZE_CHECK_POTENTIALS_HPP
#define BFIO_INUFT_INITIALIZE_CHECK_POTENTIALS_HPP

#include <array>
#include <cstddef>
#include <vector>

#include "bfio/constants.hpp"

#include "bfio/structures/box.hpp"
#include "bfio/structures/constrained_htree_walker.hpp"
#include "bfio/structures/plan.hpp"
#include "bfio/structures/weight_grid_list.hpp"

#include "bfio/tools/blas.hpp"
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
InitializeCheckPotentials
( const Context<R,1,q>& context,
  const Plan<1>& plan,
  const Box<R,1>& sourceBox,
  const Box<R,1>& targetBox,
  const Box<R,1>& mySourceBox,
  const size_t log2LocalSourceBoxes,
  const array<size_t,1>& log2LocalSourceBoxesPerDim,
  const vector<Source<R,1>>& mySources,
        WeightGridList<R,1,q>& weightGridList )
{
    const size_t N = plan.GetN();
    const size_t d = 1;

    const Direction direction = context.GetDirection();
    const R SignedTwoPi = ( direction==FORWARD ? -TwoPi : TwoPi );

    // Store the widths of the source and target boxes
    array<R,d> wA;
    wA[0] = targetBox.widths[0];
    array<R,d> wB;
    wB[0] = sourceBox.widths[0] / N;

    // Compute the center of the target box
    array<R,d> x0;
    x0[0] = targetBox.offsets[0] + wA[0]/2;

    // Store the Chebyshev grid on the target box
    const vector<array<R,d>>& chebyshevGrid = context.GetChebyshevGrid();
    vector<array<R,d>> xPoints( q );
    for( size_t t=0; t<q; ++t )
        xPoints[t][0] = x0[0] + chebyshevGrid[t][0]*wA[0];

    // Compute the unscaled weights for each local box by looping over our
    // sources and sorting them into the appropriate local box one at a time.
    // We throw an error if a source is outside of our source box.
    const size_t numSources = mySources.size();
    vector<array<R,d>> pPoints( numSources );
    vector<size_t> flattenedSourceBoxIndices( numSources );
    for( size_t s=0; s<numSources; ++s )
    {
        const array<R,d>& p = mySources[s].p;
        pPoints[s] = p;

        // Determine which local box we're in (if any)
        array<size_t,d> B;
        {
            R leftBound = mySourceBox.offsets[0];
            R rightBound = leftBound + mySourceBox.widths[0];
            if( p[0] < leftBound || p[0] >= rightBound )
            {
                std::ostringstream msg;
                msg << "Source " << s << " was at " << p[0]
                    << " in dimension " << 0 << ", but our source box in this "
                    << "dim. is [" << leftBound << "," << rightBound << ").";
                throw std::runtime_error( msg.str() );
            }

            // We must be in the box, so bitwise determine the coord. index
            B[0] = 0;
            for( size_t k=log2LocalSourceBoxesPerDim[0]; k>0; --k )
            {
                const R middle = (rightBound+leftBound)/2.;
                if( p[0] < middle )
                {
                    // implicitly setting bit k-1 of B[0] to 0
                    rightBound = middle;
                }
                else
                {
                    B[0] |= (1<<(k-1));
                    leftBound = middle;
                }
            }
        }

        // Flatten and store the integer coordinates of B
        flattenedSourceBoxIndices[s] = 
            FlattenConstrainedHTreeIndex( B, log2LocalSourceBoxesPerDim );
    }

    // Batch evaluate the dot products and multiply by +-TwoPi
    vector<R> phiResults( q*numSources );
    memset( &phiResults[0], 0, q*numSources*sizeof(R) );
    Ger
    ( q, numSources, 
      SignedTwoPi, &xPoints[0][0], 1, &pPoints[0][0], 1, 
                   &phiResults[0], q );

    // Grab the real and imaginary parts of the phase
    vector<R> sinResults, cosResults;
    SinCosBatch( phiResults, sinResults, cosResults );

    // Form the potentials from each box B on the chebyshev grid of A
    memset
    ( weightGridList.Buffer(), 0, weightGridList.Length()*2*q*sizeof(R) );
    for( size_t s=0; s<numSources; ++s )
    {
        const size_t sourceIndex = flattenedSourceBoxIndices[s];

        R* realBuffer = weightGridList[sourceIndex].RealBuffer();
        R* imagBuffer = weightGridList[sourceIndex].ImagBuffer();
        const R realMagnitude = std::real( mySources[s].magnitude );
        const R imagMagnitude = std::imag( mySources[s].magnitude );
        const R* thisCosBuffer = &cosResults[q*s];
        const R* thisSinBuffer = &sinResults[q*s];
        for( size_t t=0; t<q; ++t )
        {
            const R realPhase = thisCosBuffer[t];
            const R imagPhase = thisSinBuffer[t];
            realBuffer[t] += realPhase*realMagnitude - imagPhase*imagMagnitude;
            imagBuffer[t] += imagPhase*realMagnitude + realPhase*imagMagnitude;
        }
    }
}

// 2d specialization
template<typename R,size_t q>
void
InitializeCheckPotentials
( const Context<R,2,q>& context,
  const Plan<2>& plan,
  const Box<R,2>& sourceBox,
  const Box<R,2>& targetBox,
  const Box<R,2>& mySourceBox,
  const size_t log2LocalSourceBoxes,
  const array<size_t,2>& log2LocalSourceBoxesPerDim,
  const vector<Source<R,2>>& mySources,
        WeightGridList<R,2,q>& weightGridList )
{
    const size_t N = plan.GetN();
    const size_t d = 2;
    const size_t q_to_d = Pow<q,d>::val;

    const Direction direction = context.GetDirection();
    const R SignedTwoPi = ( direction==FORWARD ? -TwoPi : TwoPi );

    // Store the widths of the source and target boxes
    array<R,d> wA;
    for( size_t j=0; j<d; ++j )
        wA[j] = targetBox.widths[j];
    array<R,d> wB;
    for( size_t j=0; j<d; ++j )
        wB[j] = sourceBox.widths[j] / N;

    // Compute the center of the target box
    array<R,d> x0;
    for( size_t j=0; j<d; ++j )
        x0[j] = targetBox.offsets[j] + wA[j]/2;

    // Store the Chebyshev grid on the target box
    const vector<array<R,d>>& chebyshevGrid = context.GetChebyshevGrid();
    vector<array<R,d>> xPoints( q_to_d );
    for( size_t t=0; t<q_to_d; ++t )
        for( size_t j=0; j<d; ++j )
            xPoints[t][j] = x0[j] + chebyshevGrid[t][j]*wA[j];

    // Compute the unscaled weights for each local box by looping over our
    // sources and sorting them into the appropriate local box one at a time.
    // We throw an error if a source is outside of our source box.
    const size_t numSources = mySources.size();
    vector<array<R,d>> pPoints( numSources );
    vector<size_t> flattenedSourceBoxIndices( numSources );
    for( size_t s=0; s<numSources; ++s )
    {
        const array<R,d>& p = mySources[s].p;
        pPoints[s] = p;

        // Determine which local box we're in (if any)
        array<size_t,d> B;
        for( size_t j=0; j<d; ++j )
        {
            R leftBound = mySourceBox.offsets[j];
            R rightBound = leftBound + mySourceBox.widths[j];
            if( p[j] < leftBound || p[j] >= rightBound )
            {
                std::ostringstream msg;
                msg << "Source " << s << " was at " << p[j]
                    << " in dimension " << j << ", but our source box in this "
                    << "dim. is [" << leftBound << "," << rightBound << ").";
                throw std::runtime_error( msg.str() );
            }

            // We must be in the box, so bitwise determine the coord. index
            B[j] = 0;
            for( size_t k=log2LocalSourceBoxesPerDim[j]; k>0; --k )
            {
                const R middle = (rightBound+leftBound)/2.;
                if( p[j] < middle )
                {
                    // implicitly setting bit k-1 of B[j] to 0
                    rightBound = middle;
                }
                else
                {
                    B[j] |= (1<<(k-1));
                    leftBound = middle;
                }
            }
        }

        // Flatten and store the integer coordinates of B
        flattenedSourceBoxIndices[s] = 
            FlattenConstrainedHTreeIndex( B, log2LocalSourceBoxesPerDim );
    }

    // Batch evaluate the dot products and multiply by +-TwoPi
    vector<R> phiResults( q_to_d*numSources );
    Gemm
    ( 'T', 'N', q_to_d, numSources, d,
      SignedTwoPi, &xPoints[0][0], d, &pPoints[0][0], d,
      (R)0, &phiResults[0], q_to_d );

    // Grab the real and imaginary parts of the phase
    vector<R> sinResults, cosResults;
    SinCosBatch( phiResults, sinResults, cosResults );

    // Form the potentials from each box B on the chebyshev grid of A
    memset
    ( weightGridList.Buffer(), 0, weightGridList.Length()*2*q_to_d*sizeof(R) );
    for( size_t s=0; s<numSources; ++s )
    {
        const size_t sourceIndex = flattenedSourceBoxIndices[s];

        R* realBuffer = weightGridList[sourceIndex].RealBuffer();
        R* imagBuffer = weightGridList[sourceIndex].ImagBuffer();
        const R realMagnitude = std::real( mySources[s].magnitude );
        const R imagMagnitude = std::imag( mySources[s].magnitude );
        const R* thisCosBuffer = &cosResults[q_to_d*s];
        const R* thisSinBuffer = &sinResults[q_to_d*s];
        for( size_t t=0; t<q_to_d; ++t )
        {
            const R realPhase = thisCosBuffer[t];
            const R imagPhase = thisSinBuffer[t];
            realBuffer[t] += realPhase*realMagnitude - imagPhase*imagMagnitude;
            imagBuffer[t] += imagPhase*realMagnitude + realPhase*imagMagnitude;
        }
    }
}

// Fallback for 3d and above
template<typename R,size_t d,size_t q>
void
InitializeCheckPotentials
( const Context<R,d,q>& context,
  const Plan<d>& plan,
  const Box<R,d>& sourceBox,
  const Box<R,d>& targetBox,
  const Box<R,d>& mySourceBox,
  const size_t log2LocalSourceBoxes,
  const array<size_t,d>& log2LocalSourceBoxesPerDim,
  const vector<Source<R,d>>& mySources,
        WeightGridList<R,d,q>& weightGridList )
{
    const size_t N = plan.GetN();
    const size_t q_to_d = Pow<q,d>::val;

    const Direction direction = context.GetDirection();
    const R SignedTwoPi = ( direction==FORWARD ? -TwoPi : TwoPi );

    // Store the widths of the source and target boxes
    array<R,d> wA;
    for( size_t j=0; j<d; ++j )
        wA[j] = targetBox.widths[j];
    array<R,d> wB;
    for( size_t j=0; j<d; ++j )
        wB[j] = sourceBox.widths[j] / N;

    // Compute the center of the target box
    array<R,d> x0;
    for( size_t j=0; j<d; ++j )
        x0[j] = targetBox.offsets[j] + wA[j]/2;

    // Store the Chebyshev grid on the target box
    const vector<array<R,d>>& chebyshevGrid = context.GetChebyshevGrid();
    vector<array<R,d>> xPoints( q_to_d );
    for( size_t t=0; t<q_to_d; ++t )
        for( size_t j=0; j<d; ++j )
            xPoints[t][j] = x0[j] + chebyshevGrid[t][j]*wA[j];

    // Compute the unscaled weights for each local box by looping over our
    // sources and sorting them into the appropriate local box one at a time.
    // We throw an error if a source is outside of our source box.
    const size_t numSources = mySources.size();
    vector<array<R,d>> pPoints( numSources );
    vector<size_t> flattenedSourceBoxIndices( numSources );
    for( size_t s=0; s<numSources; ++s )
    {
        const array<R,d>& p = mySources[s].p;
        pPoints[s] = p;

        // Determine which local box we're in (if any)
        array<size_t,d> B;
        for( size_t j=0; j<d; ++j )
        {
            R leftBound = mySourceBox.offsets[j];
            R rightBound = leftBound + mySourceBox.widths[j];
            if( p[j] < leftBound || p[j] >= rightBound )
            {
                std::ostringstream msg;
                msg << "Source " << s << " was at " << p[j]
                    << " in dimension " << j << ", but our source box in this "
                    << "dim. is [" << leftBound << "," << rightBound << ").";
                throw std::runtime_error( msg.str() );
            }

            // We must be in the box, so bitwise determine the coord. index
            B[j] = 0;
            for( size_t k=log2LocalSourceBoxesPerDim[j]; k>0; --k )
            {
                const R middle = (rightBound+leftBound)/2.;
                if( p[j] < middle )
                {
                    // implicitly setting bit k-1 of B[j] to 0
                    rightBound = middle;
                }
                else
                {
                    B[j] |= (1<<(k-1));
                    leftBound = middle;
                }
            }
        }

        // Flatten and store the integer coordinates of B
        flattenedSourceBoxIndices[s] = 
            FlattenConstrainedHTreeIndex( B, log2LocalSourceBoxesPerDim );
    }

    // Batch evaluate the dot products and multiply by +-TwoPi
    vector<R> phiResults( q_to_d*numSources );
    Gemm
    ( 'T', 'N', q_to_d, numSources, d,
      SignedTwoPi, &xPoints[0][0], d, &pPoints[0][0], d,
      (R)0, &phiResults[0], q_to_d );

    // Grab the real and imaginary parts of the phase
    vector<R> sinResults, cosResults;
    SinCosBatch( phiResults, sinResults, cosResults );

    // Form the potentials from each box B on the chebyshev grid of A
    memset
    ( weightGridList.Buffer(), 0, weightGridList.Length()*2*q_to_d*sizeof(R) );
    for( size_t s=0; s<numSources; ++s )
    {
        const size_t sourceIndex = flattenedSourceBoxIndices[s];

        R* realBuffer = weightGridList[sourceIndex].RealBuffer();
        R* imagBuffer = weightGridList[sourceIndex].ImagBuffer();
        const R realMagnitude = std::real( mySources[s].magnitude );
        const R imagMagnitude = std::imag( mySources[s].magnitude );
        const R* thisCosBuffer = &cosResults[q_to_d*s];
        const R* thisSinBuffer = &sinResults[q_to_d*s];
        for( size_t t=0; t<q_to_d; ++t )
        {
            const R realPhase = thisCosBuffer[t];
            const R imagPhase = thisSinBuffer[t];
            realBuffer[t] += realPhase*realMagnitude - imagPhase*imagMagnitude;
            imagBuffer[t] += imagPhase*realMagnitude + realPhase*imagMagnitude;
        }
    }
}

} // inuft
} // bfio

#endif // ifndef BFIO_INUFT_INITIALIZE_CHECK_POTENTIALS_HPP 
