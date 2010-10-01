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
#ifndef BFIO_FREQ_TO_SPATIAL_INITIALIZE_WEIGHTS_HPP
#define BFIO_FREQ_TO_SPATIAL_INITIALIZE_WEIGHTS_HPP 1

#include <cstddef>
#include <vector>

#include "bfio/constants.hpp"

#include "bfio/structures/array.hpp"
#include "bfio/structures/box.hpp"
#include "bfio/structures/constrained_htree_walker.hpp"
#include "bfio/structures/context.hpp"
#include "bfio/structures/weight_grid_list.hpp"

#include "bfio/tools/flatten_constrained_htree_index.hpp"
#include "bfio/tools/mpi.hpp"
#include "bfio/tools/special_functions.hpp"

#include "bfio/functors/phase_functor.hpp"

namespace bfio {
namespace freq_to_spatial {

template<typename R,std::size_t d,std::size_t q>
void
InitializeWeights
( const PhaseFunctor<R,d>& Phi,
  const std::size_t N,
  const std::vector< Source<R,d> >& mySources,
  const Context<R,d,q>& context,
  const Box<R,d>& freqBox,
  const Box<R,d>& spatialBox,
  const Box<R,d>& myFreqBox,
  const std::size_t log2LocalFreqBoxes,
  const Array<std::size_t,d>& log2LocalFreqBoxesPerDim,
        WeightGridList<R,d,q>& weightGridList
)
{
    const std::size_t q_to_d = Pow<q,d>::val;

    Array<R,d> wB;
    for( std::size_t j=0; j<d; ++j )
        wB[j] = freqBox.widths[j] / N;

    Array<R,d> x0;
    for( std::size_t j=0; j<d; ++j )
        x0[j] = spatialBox.offsets[j] + 0.5*spatialBox.widths[j];

    // Compute the unscaled weights for each local box by looping over our
    // sources and sorting them into the appropriate local box one at a time.
    // We throw an error if a source is outside of our frequency box.
    std::vector<R> phiResults;
    std::vector<R> sinResults;
    std::vector<R> cosResults;
    const std::size_t numSources = mySources.size();
    const std::vector< Array<R,d> > xPoint( 1, x0 );
    std::vector< Array<R,d> > pPoints( numSources );
    std::vector< Array<R,d> > pRefPoints( numSources );
    std::vector<std::size_t> flattenedBoxIndices( numSources );
    for( std::size_t i=0; i<numSources; ++i )
    {
        const Array<R,d>& p = mySources[i].p;
        pPoints[i] = mySources[i].p;

        // Determine which local box we're in (if any)
        Array<std::size_t,d> B;
        for( std::size_t j=0; j<d; ++j )
        {
            R leftBound = myFreqBox.offsets[j];
            R rightBound = leftBound + myFreqBox.widths[j];
            if( p[j] < leftBound || p[j] >= rightBound )
            {
                std::ostringstream msg;
                msg << "Source " << i << " was at " << p[j]
                    << " in dimension " << j << ", but our frequency box"
                    << " in this dim. is [" << leftBound << "," 
                    << rightBound << ").";
                throw std::runtime_error( msg.str() );
            }

            // We must be in the box, so bitwise determine the coord. index
            B[j] = 0;
            for( std::size_t k=log2LocalFreqBoxesPerDim[j]; k>0; --k )
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

        // Translate the local integer coordinates into the freq. center.
        Array<R,d> p0;
        for( std::size_t j=0; j<d; ++j )
            p0[j] = myFreqBox.offsets[j] + (B[j]+0.5)*wB[j];

        // In order to add this point's contribution to the unscaled weights of 
        // B we will evaluate the Lagrangian polynomial on the reference grid,
        // so we need to map p to it first.
        for( std::size_t j=0; j<d; ++j )
            pRefPoints[i][j] = (p[j]-p0[j])/wB[j];
        
        // Flatten the integer coordinates of B
        flattenedBoxIndices[i] = 
            FlattenConstrainedHTreeIndex( B, log2LocalFreqBoxesPerDim );
    }
    Phi.BatchEvaluate( xPoint, pPoints, phiResults );
    {
        R* phiBuffer = &phiResults[0];
        for( std::size_t i=0; i<numSources; ++i )
            phiBuffer[i] *= TwoPi;
    }
    SinCosBatch( phiResults, sinResults, cosResults );
    {
        std::vector<R> realBeta( numSources );
        std::vector<R> imagBeta( numSources );
        R* realBetaBuffer = &realBeta[0];
        R* imagBetaBuffer = &imagBeta[0];
        const R* cosBuffer = &cosResults[0];
        const R* sinBuffer = &sinResults[0];
        for( std::size_t i=0; i<numSources; ++i )
        {
            const R realPhase = cosBuffer[i];
            const R imagPhase = sinBuffer[i];
            const R realMagnitude = real(mySources[i].magnitude);
            const R imagMagnitude = imag(mySources[i].magnitude);
            realBetaBuffer[i] = realPhase*realMagnitude-imagPhase*imagMagnitude;
            imagBetaBuffer[i] = imagPhase*realMagnitude+realPhase*imagMagnitude;
        }

        std::vector<R> lagrangeResults;
        for( std::size_t t=0; t<q_to_d; ++t )
        {
            context.LagrangeBatch( t, pRefPoints, lagrangeResults );
            const R* lagrangeBuffer = &lagrangeResults[0];
            for( std::size_t i=0; i<numSources; ++i )
            {
                const std::size_t k = flattenedBoxIndices[i];
                weightGridList[k].RealWeight(t) += 
                    realBetaBuffer[i]*lagrangeBuffer[i];
                weightGridList[k].ImagWeight(t) +=
                    imagBetaBuffer[i]*lagrangeBuffer[i];
            }
        }
    }

    // Loop over all of the boxes to compute the {p_t^B} and prefactors
    // for each delta weight {delta_t^AB}
    pPoints.resize( q_to_d );
    const std::vector< Array<R,d> >& chebyshevGrid = context.GetChebyshevGrid();
    ConstrainedHTreeWalker<d> BWalker( log2LocalFreqBoxesPerDim );
    for( std::size_t k=0; k<(1u<<log2LocalFreqBoxes); ++k, BWalker.Walk() ) 
    {
        const Array<std::size_t,d> B = BWalker.State();

        // Translate the local integer coordinates into the freq. center 
        Array<R,d> p0;
        for( std::size_t j=0; j<d; ++j )
            p0[j] = myFreqBox.offsets[j] + (B[j]+0.5)*wB[j];

        // Compute the prefactors given this p0 and multiply it by 
        // the corresponding weights
        {
            R* pPointsBuffer = &(pPoints[0][0]);
            const R* p0Buffer = &p0[0];
            const R* wBBuffer = &wB[0];
            const R* chebyshevBuffer = &(chebyshevGrid[0][0]);
            for( std::size_t t=0; t<q_to_d; ++t )
                for( std::size_t j=0; j<d; ++j )
                    pPointsBuffer[t*d+j] = 
                        p0Buffer[j] + wBBuffer[j]*chebyshevBuffer[t*d+j];
        }
        Phi.BatchEvaluate( xPoint, pPoints, phiResults );
        {
            R* phiBuffer = &phiResults[0];
            for( std::size_t t=0; t<q_to_d; ++t )
                phiBuffer[t] *= -TwoPi;
        }
        SinCosBatch( phiResults, sinResults, cosResults );
        {
            R* realBuffer = weightGridList[k].RealBuffer();
            R* imagBuffer = weightGridList[k].ImagBuffer();
            const R* cosBuffer = &cosResults[0];
            const R* sinBuffer = &sinResults[0];
            for( std::size_t t=0; t<q_to_d; ++t )
            {
                const R realPhase = cosBuffer[t];
                const R imagPhase = sinBuffer[t];
                const R realWeight = realBuffer[t];
                const R imagWeight = imagBuffer[t];
                realBuffer[t] = realPhase*realWeight - imagPhase*imagWeight;
                imagBuffer[t] = imagPhase*realWeight + realPhase*imagWeight;
            }
        }
    }
}

} // freq_to_spatial
} // bfio

#endif // BFIO_FREQ_TO_SPATIAL_INITIALIZE_WEIGHTS_HPP 

