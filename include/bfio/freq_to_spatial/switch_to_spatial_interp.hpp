/*
   Copyright (c) 2010, Jack Poulson
   All rights reserved.

   This file is part of ButterflyFIO.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are met:

    - Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    - Neither the name of the owner nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
   ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
   LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
   CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
   SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
   INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
   CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
   ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
   POSSIBILITY OF SUCH DAMAGE.
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
  const unsigned L, 
  const unsigned log2LocalFreqBoxes,
  const unsigned log2LocalSpatialBoxes,
  const Array<unsigned,d>& log2LocalFreqBoxesPerDim,
  const Array<unsigned,d>& log2LocalSpatialBoxesPerDim,
  const Array<R,d>& myFreqBoxOffsets,
  const Array<R,d>& mySpatialBoxOffsets,
  const std::vector< Array<R,d> >& chebyGrid,
        WeightGridList<R,d,q>& weightGridList
)
{
    typedef std::complex<R> C;
    const unsigned q_to_d = Pow<q,d>::val;

    // Compute the width of the nodes at level l
    const unsigned l = L/2;
    const R wA = static_cast<R>(1)/(1<<l);
    const R wB = static_cast<R>(1)/(1<<(L-l));
    CHTreeWalker<d> AWalker( log2LocalSpatialBoxesPerDim );
    WeightGridList<R,d,q> oldWeightGridList( weightGridList );
    for( unsigned i=0; i<(1u<<log2LocalSpatialBoxes); ++i, AWalker.Walk() )
    {
        const Array<unsigned,d> A = AWalker.State();

        // Compute the coordinates and center of this spatial box
        Array<R,d> x0A;
        for( unsigned j=0; j<d; ++j )
            x0A[j] = mySpatialBoxOffsets[j] + A[j]*wA + wA/2;

        std::vector< Array<R,d> > xPoints( q_to_d );
        for( unsigned t=0; t<q_to_d; ++t )
            for( unsigned j=0; j<d; ++j )
                xPoints[t][j] = x0A[j] + wA*chebyGrid[t][j];

        CHTreeWalker<d> BWalker( log2LocalFreqBoxesPerDim );
        for( unsigned k=0; k<(1u<<log2LocalFreqBoxes); ++k, BWalker.Walk() )
        {
            const Array<unsigned,d> B = BWalker.State();

            // Compute the coordinates and center of this freq box
            Array<R,d> p0B;
            for( unsigned j=0; j<d; ++j )
                p0B[j] = myFreqBoxOffsets[j] + B[j]*wB + wB/2;

            std::vector< Array<R,d> > pPoints( q_to_d );
            for( unsigned t=0; t<q_to_d; ++t )
                for( unsigned j=0; j<d; ++j )
                    pPoints[t][j] = p0B[j] + wB*chebyGrid[t][j];

            const unsigned key = k+(i<<log2LocalFreqBoxes);
            for( unsigned t=0; t<q_to_d; ++t )
            {
                weightGridList[key][t] = 0;
                for( unsigned tPrime=0; tPrime<q_to_d; ++tPrime )
                {
                    R alpha = TwoPi*Phi(xPoints[t],pPoints[tPrime]);
                    weightGridList[key][t] += 
                        Amp(xPoints[t],pPoints[tPrime]) *
                        C(cos(alpha),sin(alpha)) * 
                        oldWeightGridList[key][tPrime];
                }
            }
        }
    }
}

} // freq_to_spatial
} // bfio

#endif /* BFIO_FREQ_TO_SPATIAL_SWITCH_TO_SPATIAL_INTERP_HPP */

