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
  const std::vector< Array<R,d> >& chebyGrid,
  const Array<R,d>& myFreqBoxWidths,
  const Array<unsigned,d>& myFreqBox,
  const unsigned log2LocalFreqBoxes,
  const Array<unsigned,d>& log2LocalFreqBoxesPerDim,
        WeightGridList<R,d,q>& weightGridList
)
{
    typedef std::complex<R> C;
    const unsigned q_to_d = Pow<q,d>::val;

    const R wB = static_cast<R>(1) / N;

    Array<R,d> x0;
    for( unsigned j=0; j<d; ++j )
        x0[j] = 0.5;

    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );

    // Compute the unscaled weights for each local box by looping over 
    // our sources and sorting them into the appropriate local box one 
    // at a time. Bombs if a source is outside of our frequency box.
    for( unsigned i=0; i<mySources.size(); ++i )
    {
        const Array<R,d>& p = mySources[i].p;

        // Determine which local box we're in (if any)
        Array<unsigned,d> B;
        for( unsigned j=0; j<d; ++j )
        {
            const R pj = p[j];
            R leftBound = myFreqBoxWidths[j]*myFreqBox[j];
            R rightBound = myFreqBoxWidths[j]*(myFreqBox[j]+1);
            if( pj < leftBound || pj >= rightBound )
            {
                std::ostringstream msg;
                msg << "Source " << i << " was at " << pj
                    << " in dimension " << j << ", but our frequency box"
                    << " in this dim. is [" << leftBound << "," 
                    << rightBound << ").";
                throw std::runtime_error( msg.str() );
            }

            // We must be in the box, so bitwise determine the coord index
            // by bisection of box B_loc
            B[j] = 0;
            for( unsigned k=log2LocalFreqBoxesPerDim[j]; k>0; --k )
            {
                const R middle = (rightBound+leftBound)/2.;
                if( pj < middle )
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

        // Translate the local integer coordinates into the freq. center
        // of box B (not of B_loc!)
        Array<R,d> p0;
        for( unsigned j=0; j<d; ++j )
            p0[j] = myFreqBoxWidths[j]*myFreqBox[j] + B[j]*wB + wB/2;

        // Flatten the integer coordinates of B
        unsigned k = FlattenCHTreeIndex( B, log2LocalFreqBoxesPerDim );

        // Add this point's contribution to the unscaled weights of B. 
        // We evaluate the Lagrangian polynomial on the reference grid, 
        // so we need to map p to it first.
        Array<R,d> pRef;
        for( unsigned j=0; j<d; ++j )
            pRef[j] = (p[j]-p0[j])/wB;
        const C f = mySources[i].magnitude;
        const R alpha = TwoPi*Phi(x0,p);
        if( Amp.algorithm == MiddleSwitch )
        {
            const C beta = C(cos(alpha),sin(alpha)) * f;
            for( unsigned t=0; t<q_to_d; ++t )
            {
                weightGridList[k][t] += 
                    beta*Lagrange<R,d,q>(t,pRef);
            }
        }
        else if( Amp.algorithm == Prefactor )
        {
            const C beta = Amp(x0,p) * C(cos(alpha),sin(alpha)) * f;
            for( unsigned t=0; t<q_to_d; ++t )
            {
                weightGridList[k][t] +=
                    beta*Lagrange<R,d,q>(t,pRef);
            }
        }
    }

    // Loop over all of the boxes to compute the {p_t^B} and prefactors
    // for each delta weight {delta_t^AB}, exp(-2 Pi i N Phi(x0,p_t^B) ).
    CHTreeWalker<d> BWalker( log2LocalFreqBoxesPerDim );
    for( unsigned k=0; k<(1u<<log2LocalFreqBoxes); ++k, BWalker.Walk() ) 
    {
        const Array<unsigned,d> B = BWalker.State();

        // Translate the local integer coordinates into the freq. center 
        Array<R,d> p0;
        for( unsigned j=0; j<d; ++j )
            p0[j] = myFreqBoxWidths[j]*myFreqBox[j] + B[j]*wB + wB/2;

        // Compute the prefactors given this p0 and multiply it by 
        // the corresponding weights
        for( unsigned t=0; t<q_to_d; ++t )
        {
            // Compute the physical location of pt
            Array<R,d> pt;
            for( unsigned j=0; j<d; ++j )
                pt[j] = p0[j] + wB*chebyGrid[t][j];

            const R alpha = TwoPi*Phi(x0,pt);
            if( Amp.algorithm == MiddleSwitch )
            {
                weightGridList[k][t] /= C(cos(alpha),sin(alpha));
            }
            else if( Amp.algorithm == Prefactor )
            {
                weightGridList[k][t] /= Amp(x0,pt) * C(cos(alpha),sin(alpha)); 
            }
        }
    }
}

} // freq_to_spatial
} // bfio

#endif /* BFIO_FREQ_TO_SPATIAL_INITIALIZE_WEIGHTS_HPP */

