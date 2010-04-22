/*
   Copyright 2010 Jack Poulson

   This file is part of ButterflyFIO.

   This program is free software: you can redistribute it and/or modify it under
   the terms of the GNU Lesser General Public License as published by the
   Free Software Foundation; either version 3 of the License, or 
   (at your option) any later version.

   This program is distributed in the hope that it will be useful, but 
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with this program. If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef BFIO_SWITCH_TO_SPATIAL_INTERP_HPP
#define BFIO_SWITCH_TO_SPATIAL_INTERP_HPP 1

#include "BFIO/Lagrange.hpp"

namespace BFIO
{
    using namespace std;

    template<typename Phi,typename R,unsigned d,unsigned q>
    void
    SwitchToSpatialInterp
    ( const unsigned L, 
      const unsigned log2LocalFreqBoxes,
      const unsigned log2LocalSpatialBoxes,
      const Array<unsigned,d>& log2LocalFreqBoxesPerDim,
      const Array<unsigned,d>& log2LocalSpatialBoxesPerDim,
      const Array<R,d>& myFreqBoxOffsets,
      const Array<R,d>& mySpatialBoxOffsets,
      const vector< Array<R,d> >& chebyGrid,
            WeightSetList<R,d,q>& weightSetList              )
    {
        typedef complex<R> C;

        // Compute the width of the nodes at level l
        const unsigned l = L/2;
        const R wA = static_cast<R>(1)/(1<<l);
        const R wB = static_cast<R>(1)/(1<<(L-l));
        CHTreeWalker<d> AWalker( log2LocalSpatialBoxesPerDim );
        WeightSetList<R,d,q> oldWeightSetList( weightSetList );
        for( unsigned i=0; i<(1u<<log2LocalSpatialBoxes); ++i, AWalker.Walk() )
        {
            const Array<unsigned,d> A = AWalker.State();

            // Compute the coordinates and center of this spatial box
            Array<R,d> x0A;
            for( unsigned j=0; j<d; ++j )
                x0A[j] = mySpatialBoxOffsets[j] + A[j]*wA + wA/2;

            vector< Array<R,d> > xPoints( Pow<q,d>::val );
            for( unsigned t=0; t<Pow<q,d>::val; ++t )
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

                vector< Array<R,d> > pPoints( Pow<q,d>::val );
                for( unsigned t=0; t<Pow<q,d>::val; ++t )
                    for( unsigned j=0; j<d; ++j )
                        pPoints[t][j] = p0B[j] + wB*chebyGrid[t][j];

                const unsigned key = k+(i<<log2LocalFreqBoxes);
                for( unsigned t=0; t<Pow<q,d>::val; ++t )
                {
                    weightSetList[key][t] = 0;
                    for( unsigned tp=0; tp<Pow<q,d>::val; ++tp )
                    {
                        R alpha = TwoPi*Phi::Eval(xPoints[t],pPoints[tp]);
                        weightSetList[key][t] += 
                            C(cos(alpha),sin(alpha)) * 
                            oldWeightSetList[key][tp];
                    }
                }
            }
        }
    }
}

#endif /* BFIO_SWITCH_TO_SPATIAL_INTERP_HPP */

