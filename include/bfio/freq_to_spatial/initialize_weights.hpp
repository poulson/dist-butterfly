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
#pragma once
#ifndef BFIO_FREQ_TO_SPATIAL_INITIALIZE_WEIGHTS_HPP
#define BFIO_FREQ_TO_SPATIAL_INITIALIZE_WEIGHTS_HPP 1

#include "bfio/structures/data.hpp"
#include "bfio/structures/htree_walker.hpp"
#include "bfio/tools/flatten_htree_index.hpp"
#include "bfio/tools/mpi.hpp"

namespace bfio {
namespace freq_to_spatial {

template<typename R,unsigned d,unsigned q>
void
InitializeWeights
( const AmplitudeFunctor<R,d>& Amp,
  const PhaseFunctor<R,d>& Phi,
  const unsigned N,
  const std::vector< Source<R,d> >& mySources,
  const Context<R,d,q>& context,
  const Box<R,d>& freqBox,
  const Box<R,d>& spatialBox,
  const Box<R,d>& myFreqBox,
  const unsigned log2LocalFreqBoxes,
  const Array<unsigned,d>& log2LocalFreqBoxesPerDim,
        WeightGridList<R,d,q>& weightGridList
)
{
    typedef std::complex<R> C;
    const unsigned q_to_d = Pow<q,d>::val;

    Array<R,d> wB;
    for( unsigned j=0; j<d; ++j )
        wB[j] = freqBox.widths[j] / N;

    Array<R,d> x0;
    for( unsigned j=0; j<d; ++j )
        x0[j] = spatialBox.offsets[j] + 0.5*spatialBox.widths[j];

    // Compute the unscaled weights for each local box by looping over our
    // sources and sorting them into the appropriate local box one at a time.
    // We throw an error if a source is outside of our frequency box.
    const unsigned numSources = mySources.size();
    const std::vector< Array<R,d> > xPoint( 1, x0 );
    std::vector< Array<R,d> > pPoints( numSources );
    std::vector< Array<R,d> > pRefPoints( numSources );
    std::vector<R> phiResults( numSources );
    std::vector<C> imagExpResults( numSources );
    std::vector<unsigned> flattenedBoxIndices( numSources );
    for( unsigned i=0; i<numSources; ++i )
    {
        const Array<R,d>& p = mySources[i].p;
        pPoints[i] = mySources[i].p;

        // Determine which local box we're in (if any)
        Array<unsigned,d> B;
        for( unsigned j=0; j<d; ++j )
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
            for( unsigned k=log2LocalFreqBoxesPerDim[j]; k>0; --k )
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
        for( unsigned j=0; j<d; ++j )
            p0[j] = myFreqBox.offsets[j] + (B[j]+0.5)*wB[j];

        // In order to add this point's contribution to the unscaled weights of 
        // B we will evaluate the Lagrangian polynomial on the reference grid,
        // so we need to map p to it first.
        for( unsigned j=0; j<d; ++j )
            pRefPoints[i][j] = (p[j]-p0[j])/wB[j];
        
        // Flatten the integer coordinates of B
        flattenedBoxIndices[i] = 
            FlattenConstrainedHTreeIndex( B, log2LocalFreqBoxesPerDim );
    }
    Phi.BatchEvaluate( xPoint, pPoints, phiResults );
    for( unsigned i=0; i<numSources; ++i )
        phiResults[i] *= TwoPi;
    ImagExpBatch<R>( phiResults, imagExpResults );
    for( unsigned i=0; i<numSources; ++i )
    {
        const unsigned k = flattenedBoxIndices[i];
        const Array<R,d>& pRef = pRefPoints[i];
        const C beta = imagExpResults[i]*mySources[i].magnitude;
        for( unsigned t=0; t<q_to_d; ++t )
            weightGridList[k][t] += beta*context.Lagrange(t,pRef);
    }

    // Loop over all of the boxes to compute the {p_t^B} and prefactors
    // for each delta weight {delta_t^AB}
    pPoints.resize( q_to_d );
    phiResults.resize( q_to_d );
    imagExpResults.resize( q_to_d );
    const std::vector< Array<R,d> >& chebyshevGrid = context.GetChebyshevGrid();
    ConstrainedHTreeWalker<d> BWalker( log2LocalFreqBoxesPerDim );
    for( unsigned k=0; k<(1u<<log2LocalFreqBoxes); ++k, BWalker.Walk() ) 
    {
        const Array<unsigned,d> B = BWalker.State();

        // Translate the local integer coordinates into the freq. center 
        Array<R,d> p0;
        for( unsigned j=0; j<d; ++j )
            p0[j] = myFreqBox.offsets[j] + (B[j]+0.5)*wB[j];

        // Compute the prefactors given this p0 and multiply it by 
        // the corresponding weights
        for( unsigned t=0; t<q_to_d; ++t )
            for( unsigned j=0; j<d; ++j )
                pPoints[t][j] = p0[j] + wB[j]*chebyshevGrid[t][j];
        Phi.BatchEvaluate( xPoint, pPoints, phiResults );
        for( unsigned t=0; t<q_to_d; ++t )
            phiResults[t] *= TwoPi;
        ImagExpBatch<R>( phiResults, imagExpResults );
        for( unsigned t=0; t<q_to_d; ++t )
            weightGridList[k][t] /= imagExpResults[t];
    }
}

} // freq_to_spatial
} // bfio

#endif // BFIO_FREQ_TO_SPATIAL_INITIALIZE_WEIGHTS_HPP 

