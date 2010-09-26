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
#ifndef BFIO_FREQ_TO_SPATIAL_SWITCH_TO_SPATIAL_INTERP_HPP
#define BFIO_FREQ_TO_SPATIAL_SWITCH_TO_SPATIAL_INTERP_HPP 1

#include "bfio/structures/htree_walker.hpp"

namespace bfio {
namespace freq_to_spatial {

template<typename R,std::size_t d,std::size_t q>
void
SwitchToSpatialInterp
( const AmplitudeFunctor<R,d>& Amp,
  const PhaseFunctor<R,d>& Phi,
  const std::size_t log2N, 
  const Box<R,d>& freqBox,
  const Box<R,d>& spatialBox,
  const Box<R,d>& myFreqBox,
  const Box<R,d>& mySpatialBox,
  const std::size_t log2LocalFreqBoxes,
  const std::size_t log2LocalSpatialBoxes,
  const Array<std::size_t,d>& log2LocalFreqBoxesPerDim,
  const Array<std::size_t,d>& log2LocalSpatialBoxesPerDim,
  const Context<R,d,q>& context,
        WeightGridList<R,d,q>& weightGridList
)
{
    typedef std::complex<R> C;
    const std::size_t q_to_d = Pow<q,d>::val;

    // Compute the width of the nodes at level log2N/2
    const std::size_t level = log2N/2;
    Array<R,d> wA, wB;
    for( std::size_t j=0; j<d; ++j )
    {
        wA[j] = spatialBox.widths[j] / (1<<level);
        wB[j] = freqBox.widths[j] / (1<<(log2N-level));
    }

    const std::vector< Array<R,d> >& chebyshevGrid = 
        context.GetChebyshevGrid();
    ConstrainedHTreeWalker<d> AWalker( log2LocalSpatialBoxesPerDim );
    WeightGridList<R,d,q> oldWeightGridList( weightGridList );
    for( std::size_t i=0; i<(1u<<log2LocalSpatialBoxes); ++i, AWalker.Walk() )
    {
        const Array<std::size_t,d> A = AWalker.State();

        // Compute the coordinates and center of this spatial box
        Array<R,d> x0A;
        for( std::size_t j=0; j<d; ++j )
            x0A[j] = mySpatialBox.offsets[j] + (A[j]+0.5)*wA[j];

        std::vector< Array<R,d> > xPoints( q_to_d );
        for( std::size_t t=0; t<q_to_d; ++t )
            for( std::size_t j=0; j<d; ++j )
                xPoints[t][j] = x0A[j] + wA[j]*chebyshevGrid[t][j];

        std::vector<C> ampResults;
        std::vector<R> phiResults;
        std::vector<R> sinResults;
        std::vector<R> cosResults;
        ConstrainedHTreeWalker<d> BWalker( log2LocalFreqBoxesPerDim );
        for( std::size_t k=0; k<(1u<<log2LocalFreqBoxes); ++k, BWalker.Walk() )
        {
            const Array<std::size_t,d> B = BWalker.State();

            // Compute the coordinates and center of this freq box
            Array<R,d> p0B;
            for( std::size_t j=0; j<d; ++j )
                p0B[j] = myFreqBox.offsets[j] + (B[j]+0.5)*wB[j];

            std::vector< Array<R,d> > pPoints( q_to_d );
            for( std::size_t t=0; t<q_to_d; ++t )
                for( std::size_t j=0; j<d; ++j )
                    pPoints[t][j] = p0B[j] + wB[j]*chebyshevGrid[t][j];

            Amp.BatchEvaluate( xPoints, pPoints, ampResults );
            Phi.BatchEvaluate( xPoints, pPoints, phiResults );
            for( std::size_t j=0; j<phiResults.size(); ++j )
                phiResults[j] *= TwoPi;
            SinCosBatch( phiResults, sinResults, cosResults );

            const std::size_t key = k+(i<<log2LocalFreqBoxes);
            for( std::size_t t=0; t<q_to_d; ++t )
            {
                weightGridList[key].RealWeight(t) = 0;
                weightGridList[key].ImagWeight(t) = 0;
                for( std::size_t tPrime=0; tPrime<q_to_d; ++tPrime )
                {
                    const WeightGrid<R,d,q>& oldGrid = oldWeightGridList[key];
                    const R realWeight = oldGrid.RealWeight(tPrime);
                    const R imagWeight = oldGrid.ImagWeight(tPrime);
                    const R cosResult  = cosResults[t*q_to_d+tPrime];
                    const R sinResult  = sinResults[t*q_to_d+tPrime];
                    const R realAmp    = real(ampResults[t*q_to_d+tPrime]);
                    const R imagAmp    = imag(ampResults[t*q_to_d+tPrime]);
                    const R realBeta=cosResult*realWeight-sinResult*imagWeight;
                    const R imagBeta=sinResult*realWeight+cosResult*imagWeight;
                    weightGridList[key].RealWeight(t) += 
                        realAmp*realBeta - imagAmp*imagBeta;
                    weightGridList[key].ImagWeight(t) +=
                        imagAmp*realBeta + realAmp*imagBeta;
                }
            }
        }
    }
}

} // freq_to_spatial
} // bfio

#endif // BFIO_FREQ_TO_SPATIAL_SWITCH_TO_SPATIAL_INTERP_HPP

