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
#ifndef BFIO_NUFT_INITIALIZE_CHECK_POTENTIALS_HPP
#define BFIO_NUFT_INITIALIZE_CHECK_POTENTIALS_HPP 1

#include <cstddef>
#include <vector>

#include "bfio/constants.hpp"

#include "bfio/structures/array.hpp"
#include "bfio/structures/box.hpp"
#include "bfio/structures/constrained_htree_walker.hpp"
#include "bfio/structures/plan.hpp"
#include "bfio/structures/weight_grid_list.hpp"

#include "bfio/tools/blas.hpp"
#include "bfio/tools/flatten_constrained_htree_index.hpp"
#include "bfio/tools/mpi.hpp"
#include "bfio/tools/special_functions.hpp"

#include "bfio/interpolative_nuft/context.hpp"

namespace bfio {
namespace interpolative_nuft {

// 1d specialization
template<typename R,std::size_t q>
void
InitializeCheckPotentials
( const interpolative_nuft::Context<R,1,q>& context,
  const Plan<1>& plan,
  const Box<R,1>& sourceBox,
  const Box<R,1>& targetBox,
  const Box<R,1>& mySourceBox,
  const std::size_t log2LocalSourceBoxes,
  const Array<std::size_t,1>& log2LocalSourceBoxesPerDim,
  const std::vector< Source<R,1> >& mySources,
        WeightGridList<R,1,q>& weightGridList )
{
    const std::size_t N = plan.GetN();
    const std::size_t d = 1;

    // Store the widths of the source and target boxes
    Array<R,d> wA;
    wA[0] = targetBox.widths[0];
    Array<R,d> wB;
    wB[0] = sourceBox.widths[0] / N;

    // Compute the center of the target box
    Array<R,d> x0;
    x0[0] = targetBox.offsets[0] + wA[0]/2;

    // Store the Chebyshev grid on the target box
    const std::vector< Array<R,d> >& chebyshevGrid = context.GetChebyshevGrid();
    std::vector< Array<R,d> > xPoints( q );
    for( std::size_t t=0; t<q; ++t )
        xPoints[t][0] = x0[0] + chebyshevGrid[t][0]*wA[0];

    // Compute the unscaled weights for each local box by looping over our
    // sources and sorting them into the appropriate local box one at a time.
    // We throw an error if a source is outside of our source box.
    const std::size_t numSources = mySources.size();
    std::vector< Array<R,d> > pPoints( numSources );
    std::vector<std::size_t> flattenedSourceBoxIndices( numSources );
    for( std::size_t s=0; s<numSources; ++s )
    {
        const Array<R,d>& p = mySources[s].p;
        pPoints[s] = p;

        // Determine which local box we're in (if any)
        Array<std::size_t,d> B;
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
            for( std::size_t k=log2LocalSourceBoxesPerDim[0]; k>0; --k )
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

    // Batch evaluate the dot products and multiply by TwoPi
    std::vector<R> phiResults( q*numSources );
    std::memset( &phiResults[0], 0, q*numSources*sizeof(R) );
    Ger
    ( q, numSources, 
      TwoPi, &xPoints[0][0], 1, &pPoints[0][0], 1, 
             &phiResults[0], q );

    // Grab the real and imaginary parts of the phase
    std::vector<R> sinResults;
    std::vector<R> cosResults;
    SinCosBatch( phiResults, sinResults, cosResults );

    // Form the potentials from each box B on the chebyshev grid of A
    std::memset
    ( weightGridList.Buffer(), 0, weightGridList.Length()*2*q*sizeof(R) );
    for( std::size_t s=0; s<numSources; ++s )
    {
        const std::size_t sourceIndex = flattenedSourceBoxIndices[s];

        R* realBuffer = weightGridList[sourceIndex].RealBuffer();
        R* imagBuffer = weightGridList[sourceIndex].ImagBuffer();
        const R realMagnitude = std::real( mySources[s].magnitude );
        const R imagMagnitude = std::imag( mySources[s].magnitude );
        const R* thisCosBuffer = &cosResults[q*s];
        const R* thisSinBuffer = &sinResults[q*s];
        for( std::size_t t=0; t<q; ++t )
        {
            const R realPhase = thisCosBuffer[t];
            const R imagPhase = thisSinBuffer[t];
            realBuffer[t] += realPhase*realMagnitude - imagPhase*imagMagnitude;
            imagBuffer[t] += imagPhase*realMagnitude + realPhase*imagMagnitude;
        }
    }
}

// 2d specialization
template<typename R,std::size_t q>
void
InitializeCheckPotentials
( const interpolative_nuft::Context<R,2,q>& context,
  const Plan<2>& plan,
  const Box<R,2>& sourceBox,
  const Box<R,2>& targetBox,
  const Box<R,2>& mySourceBox,
  const std::size_t log2LocalSourceBoxes,
  const Array<std::size_t,2>& log2LocalSourceBoxesPerDim,
  const std::vector< Source<R,2> >& mySources,
        WeightGridList<R,2,q>& weightGridList )
{
    const std::size_t N = plan.GetN();
    const std::size_t d = 2;
    const std::size_t q_to_d = Pow<q,d>::val;

    // Store the widths of the source and target boxes
    Array<R,d> wA;
    for( std::size_t j=0; j<d; ++j )
        wA[j] = targetBox.widths[j];
    Array<R,d> wB;
    for( std::size_t j=0; j<d; ++j )
        wB[j] = sourceBox.widths[j] / N;

    // Compute the center of the target box
    Array<R,d> x0;
    for( std::size_t j=0; j<d; ++j )
        x0[j] = targetBox.offsets[j] + wA[j]/2;

    // Store the Chebyshev grid on the target box
    const std::vector< Array<R,d> >& chebyshevGrid = context.GetChebyshevGrid();
    std::vector< Array<R,d> > xPoints( q_to_d );
    for( std::size_t t=0; t<q_to_d; ++t )
        for( std::size_t j=0; j<d; ++j )
            xPoints[t][j] = x0[j] + chebyshevGrid[t][j]*wA[j];

    // Compute the unscaled weights for each local box by looping over our
    // sources and sorting them into the appropriate local box one at a time.
    // We throw an error if a source is outside of our source box.
    const std::size_t numSources = mySources.size();
    std::vector< Array<R,d> > pPoints( numSources );
    std::vector<std::size_t> flattenedSourceBoxIndices( numSources );
    for( std::size_t s=0; s<numSources; ++s )
    {
        const Array<R,d>& p = mySources[s].p;
        pPoints[s] = p;

        // Determine which local box we're in (if any)
        Array<std::size_t,d> B;
        for( std::size_t j=0; j<d; ++j )
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
            for( std::size_t k=log2LocalSourceBoxesPerDim[j]; k>0; --k )
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

    // Batch evaluate the dot products and multiply by TwoPi
    std::vector<R> phiResults( q_to_d*numSources );
    Gemm
    ( 'T', 'N', q_to_d, numSources, d,
      TwoPi, &xPoints[0][0], d, &pPoints[0][0], d,
      (R)0, &phiResults[0], q_to_d );

    // Grab the real and imaginary parts of the phase
    std::vector<R> sinResults;
    std::vector<R> cosResults;
    SinCosBatch( phiResults, sinResults, cosResults );

    // Form the potentials from each box B on the chebyshev grid of A
    std::memset
    ( weightGridList.Buffer(), 0, weightGridList.Length()*2*q_to_d*sizeof(R) );
    for( std::size_t s=0; s<numSources; ++s )
    {
        const std::size_t sourceIndex = flattenedSourceBoxIndices[s];

        R* realBuffer = weightGridList[sourceIndex].RealBuffer();
        R* imagBuffer = weightGridList[sourceIndex].ImagBuffer();
        const R realMagnitude = std::real( mySources[s].magnitude );
        const R imagMagnitude = std::imag( mySources[s].magnitude );
        const R* thisCosBuffer = &cosResults[q_to_d*s];
        const R* thisSinBuffer = &sinResults[q_to_d*s];
        for( std::size_t t=0; t<q_to_d; ++t )
        {
            const R realPhase = thisCosBuffer[t];
            const R imagPhase = thisSinBuffer[t];
            realBuffer[t] += realPhase*realMagnitude - imagPhase*imagMagnitude;
            imagBuffer[t] += imagPhase*realMagnitude + realPhase*imagMagnitude;
        }
    }
}

// Fallback for 3d and above
template<typename R,std::size_t d,std::size_t q>
void
InitializeCheckPotentials
( const interpolative_nuft::Context<R,d,q>& context,
  const Plan<d>& plan,
  const Box<R,d>& sourceBox,
  const Box<R,d>& targetBox,
  const Box<R,d>& mySourceBox,
  const std::size_t log2LocalSourceBoxes,
  const Array<std::size_t,d>& log2LocalSourceBoxesPerDim,
  const std::vector< Source<R,d> >& mySources,
        WeightGridList<R,d,q>& weightGridList )
{
    const std::size_t N = plan.GetN();
    const std::size_t q_to_d = Pow<q,d>::val;

    // Store the widths of the source and target boxes
    Array<R,d> wA;
    for( std::size_t j=0; j<d; ++j )
        wA[j] = targetBox.widths[j];
    Array<R,d> wB;
    for( std::size_t j=0; j<d; ++j )
        wB[j] = sourceBox.widths[j] / N;

    // Compute the center of the target box
    Array<R,d> x0;
    for( std::size_t j=0; j<d; ++j )
        x0[j] = targetBox.offsets[j] + wA[j]/2;

    // Store the Chebyshev grid on the target box
    const std::vector< Array<R,d> >& chebyshevGrid = context.GetChebyshevGrid();
    std::vector< Array<R,d> > xPoints( q_to_d );
    for( std::size_t t=0; t<q_to_d; ++t )
        for( std::size_t j=0; j<d; ++j )
            xPoints[t][j] = x0[j] + chebyshevGrid[t][j]*wA[j];

    // Compute the unscaled weights for each local box by looping over our
    // sources and sorting them into the appropriate local box one at a time.
    // We throw an error if a source is outside of our source box.
    const std::size_t numSources = mySources.size();
    std::vector< Array<R,d> > pPoints( numSources );
    std::vector<std::size_t> flattenedSourceBoxIndices( numSources );
    for( std::size_t s=0; s<numSources; ++s )
    {
        const Array<R,d>& p = mySources[s].p;
        pPoints[s] = p;

        // Determine which local box we're in (if any)
        Array<std::size_t,d> B;
        for( std::size_t j=0; j<d; ++j )
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
            for( std::size_t k=log2LocalSourceBoxesPerDim[j]; k>0; --k )
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

    // Batch evaluate the dot products and multiply by TwoPi
    std::vector<R> phiResults( q_to_d*numSources );
    Gemm
    ( 'T', 'N', q_to_d, numSources, d,
      TwoPi, &xPoints[0][0], d, &pPoints[0][0], d,
      (R)0, &phiResults[0], q_to_d );

    // Grab the real and imaginary parts of the phase
    std::vector<R> sinResults;
    std::vector<R> cosResults;
    SinCosBatch( phiResults, sinResults, cosResults );

    // Form the potentials from each box B on the chebyshev grid of A
    std::memset
    ( weightGridList.Buffer(), 0, weightGridList.Length()*2*q_to_d*sizeof(R) );
    for( std::size_t s=0; s<numSources; ++s )
    {
        const std::size_t sourceIndex = flattenedSourceBoxIndices[s];

        R* realBuffer = weightGridList[sourceIndex].RealBuffer();
        R* imagBuffer = weightGridList[sourceIndex].ImagBuffer();
        const R realMagnitude = std::real( mySources[s].magnitude );
        const R imagMagnitude = std::imag( mySources[s].magnitude );
        const R* thisCosBuffer = &cosResults[q_to_d*s];
        const R* thisSinBuffer = &sinResults[q_to_d*s];
        for( std::size_t t=0; t<q_to_d; ++t )
        {
            const R realPhase = thisCosBuffer[t];
            const R imagPhase = thisSinBuffer[t];
            realBuffer[t] += realPhase*realMagnitude - imagPhase*imagMagnitude;
            imagBuffer[t] += imagPhase*realMagnitude + realPhase*imagMagnitude;
        }
    }
}

} // interpolative_nuft
} // bfio

#endif // BFIO_INTERPOLATIVE_NUFT_INITIALIZE_CHECK_POTENTIALS_HPP 

