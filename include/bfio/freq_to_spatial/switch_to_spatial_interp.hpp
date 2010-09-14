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
#ifndef BFIO_FREQ_TO_SPATIAL_SWITCH_TO_SPATIAL_INTERP_HPP
#define BFIO_FREQ_TO_SPATIAL_SWITCH_TO_SPATIAL_INTERP_HPP 1

#include "bfio/structures/htree_walker.hpp"
#include "bfio/tools/lagrange.hpp"

namespace bfio {
namespace freq_to_spatial {

template<typename R,unsigned d,unsigned q>
void
SwitchToSpatialInterp
( const AmplitudeFunctor<R,d>& Amp,
  const PhaseFunctor<R,d>& Phi,
  const unsigned log2N, 
  const Box<R,d>& freqBox,
  const Box<R,d>& spatialBox,
  const Box<R,d>& myFreqBox,
  const Box<R,d>& mySpatialBox,
  const unsigned log2LocalFreqBoxes,
  const unsigned log2LocalSpatialBoxes,
  const Array<unsigned,d>& log2LocalFreqBoxesPerDim,
  const Array<unsigned,d>& log2LocalSpatialBoxesPerDim,
  const std::vector< Array<R,d> >& chebyGrid,
        WeightGridList<R,d,q>& weightGridList
)
{
    typedef std::complex<R> C;
    const unsigned q_to_d = Pow<q,d>::val;

    // Compute the width of the nodes at level log2N/2
    const unsigned level = log2N/2;
    Array<R,d> wA, wB;
    for( unsigned j=0; j<d; ++j )
    {
        wA[j] = spatialBox.widths[j] / (1<<level);
        wB[j] = freqBox.widths[j] / (1<<(log2N-level));
    }

    CHTreeWalker<d> AWalker( log2LocalSpatialBoxesPerDim );
    WeightGridList<R,d,q> oldWeightGridList( weightGridList );
    for( unsigned i=0; i<(1u<<log2LocalSpatialBoxes); ++i, AWalker.Walk() )
    {
        const Array<unsigned,d> A = AWalker.State();

        // Compute the coordinates and center of this spatial box
        Array<R,d> x0A;
        for( unsigned j=0; j<d; ++j )
            x0A[j] = mySpatialBox.offsets[j] + (A[j]+0.5)*wA[j];

        std::vector< Array<R,d> > xPoints( q_to_d );
        for( unsigned t=0; t<q_to_d; ++t )
            for( unsigned j=0; j<d; ++j )
                xPoints[t][j] = x0A[j] + wA[j]*chebyGrid[t][j];

        CHTreeWalker<d> BWalker( log2LocalFreqBoxesPerDim );
        for( unsigned k=0; k<(1u<<log2LocalFreqBoxes); ++k, BWalker.Walk() )
        {
            const Array<unsigned,d> B = BWalker.State();

            // Compute the coordinates and center of this freq box
            Array<R,d> p0B;
            for( unsigned j=0; j<d; ++j )
                p0B[j] = myFreqBox.offsets[j] + (B[j]+0.5)*wB[j];

            std::vector< Array<R,d> > pPoints( q_to_d );
            for( unsigned t=0; t<q_to_d; ++t )
                for( unsigned j=0; j<d; ++j )
                    pPoints[t][j] = p0B[j] + wB[j]*chebyGrid[t][j];

            const unsigned key = k+(i<<log2LocalFreqBoxes);
            for( unsigned t=0; t<q_to_d; ++t )
            {
                weightGridList[key][t] = 0;
                for( unsigned tPrime=0; tPrime<q_to_d; ++tPrime )
                {
                    const C beta = 
                        ImagExp( TwoPi*Phi(xPoints[t],pPoints[tPrime]) );
                    weightGridList[key][t] += 
                        Amp(xPoints[t],pPoints[tPrime]) * beta *
                        oldWeightGridList[key][tPrime];
                }
            }
        }
    }
}

} // freq_to_spatial
} // bfio

#endif // BFIO_FREQ_TO_SPATIAL_SWITCH_TO_SPATIAL_INTERP_HPP

