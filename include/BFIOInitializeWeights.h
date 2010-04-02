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
#ifndef BFIO_INITIALIZE_WEIGHTS_H
#define BFIO_INITIALIZE_WEIGHTS_H 1

#include "BFIOUtil.h"
#include "BFIOTemplate.h"

namespace BFIO
{
    template<typename Psi,typename R,unsigned d,unsigned q>
    inline void
    InitializeWeights
    ( const unsigned N,
      const std::vector< Source<R,d> >& mySources,
      const Array<R,q>& chebyGrid,
      const Array<R,d>& myFreqBoxWidths,
      const Array<unsigned,d>& myFreqBoxCoords,
      const unsigned boxes,
      const Array<unsigned,d>& log2BoxesPerDim,
            std::vector< Array<std::complex<R>,Power<q,d>::value> >& weights )
    {
        using namespace std;
        typedef complex<R> C;

        const R widthOfB = static_cast<R>(1) / N;

        Array<R,d> x0;
        for( unsigned j=0; j<d; ++j )
            x0[j] = 0.5;

        for( unsigned i=0; i<mySources.size(); ++i )
        {
            const Array<R,d>& p = mySources[i].p;

            // Determine which local box we're in (if any)
            Array<unsigned,d> localBoxCoords;
            for( unsigned j=0; j<d; ++j )
            {
                const R pj = p[j];
                R leftBound = myFreqBoxWidths[j]*myFreqBoxCoords[j];
                R rightBound = myFreqBoxWidths[j]*(myFreqBoxCoords[j]+1);
                if( pj < leftBound || pj >= rightBound )
                {
                    cerr << "Source " << i << " was at " << pj
                         << " in dimension " << j << ", but our frequency box"
                         << " in this dim. is [" << leftBound << "," 
                         << rightBound << ")." << endl;
                    throw 0;
                }

                // We must be in the box, so bitwise determine the coord index
                // by bisection of box B_loc
                localBoxCoords[j] = 0;
                for( unsigned k=log2BoxesPerDim[j]; k>0; --k )
                {
                    const R middle = (rightBound-leftBound)/2.;
                    if( pj < middle )
                    {
                        // implicitly setting bit k-1 of localBoxCoords[j] to 0
                        rightBound = middle;
                    }
                    else
                    {
                        localBoxCoords[j] |= (1<<(k-1));
                        leftBound = middle;
                    }
                }
            }

            // Translate the local integer coordinates into the freq. center
            // of box B (not of B_loc!)
            Array<R,d> p0;
            for( unsigned j=0; j<d; ++j )
            {
                p0[j] = myFreqBoxWidths[j]*myFreqBoxCoords[j] + 
                        (localBoxCoords[j]+0.5)*widthOfB;
            }

            // Flatten the integer coordinates of B_loc into a single index
            unsigned k = 0;
            for( unsigned j=0; j<d; ++j )
            {
                static unsigned log2BoxesUpToDim = 0;
                k |= (localBoxCoords[j]<<log2BoxesUpToDim);
                log2BoxesUpToDim += log2BoxesPerDim[j];
            }

            // Add this point's contribution to the unscaled weights of B. 
            // We evaluate the Lagrangian polynomial on the reference grid, 
            // so we need to map p to it first.
            Array<R,d> pRef;
            for( unsigned j=0; j<d; ++j )
                pRef[j] = (p[j]-p0[j])/widthOfB;
            const C f = mySources[i].magnitude;
            const C alpha = exp( C(0.,TwoPi*N)*Psi::Eval(x0,p) ) * f;
            WeightSummation<R,d,q>::Eval(alpha,pRef,chebyGrid,weights[k]);
        }

        // Loop over all of the boxes to compute the {p_t^B} and prefactors
        // for each delta weight {delta_t^AB}, exp(-2 Pi i N Psi(x0,p_t^B) ).
        // Notice that if we are sharing a box with one or more processes, then
        // our local boxes, say B_loc, are subsets of a B, but we still need 
        // to consider our contribution to the weights corresponding to 
        // p_t^B lying in B \ B_loc.
        for( unsigned k=0; k<boxes; ++k ) 
        {
            // Compute the local integer coordinates of box k
            Array<unsigned,d> localBoxCoords;
            for( unsigned j=0; j<d; ++j )
            {
                static unsigned log2BoxesUpToDim = 0;
                unsigned log2BoxesUpToNextDim = 
                    log2BoxesUpToDim+log2BoxesPerDim[j];
                localBoxCoords[j] = (k>>log2BoxesUpToDim) & 
                                    ((1<<log2BoxesUpToNextDim)-1);
                log2BoxesUpToDim = log2BoxesUpToNextDim;
            }

            // Translate the local integer coordinates into the freq. center 
            Array<R,d> p0;
            for( unsigned j=0; j<d; ++j )
            {
                p0[j] = myFreqBoxWidths[j]*myFreqBoxCoords[j] + 
                        (localBoxCoords[j]+0.5)*widthOfB;
            }

            // Compute the prefactors given this p0 and multiply it by 
            // the corresponding weights
            ScaleWeights<Psi,R,d,q>::Eval
            (N,widthOfB,x0,p0,chebyGrid,weights[k]);
        }
    }
}

#endif /* BFIO_INITIALIZE_WEIGHTS_H */

