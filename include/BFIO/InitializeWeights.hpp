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
#ifndef BFIO_INITIALIZE_WEIGHTS_HPP
#define BFIO_INITIALIZE_WEIGHTS_HPP 1

#include "BFIO/Util.hpp"

namespace BFIO
{
    using namespace std;

    // Psi: Phase function in polar frequency coordinates
    // R:   type for real variables (e.g., float or double)
    // d:   dimension of problem
    // q:   number of Chebyshev gridpoints per dimension
    template<typename Psi,typename R,unsigned d,unsigned q>
    void
    InitializeWeights
    ( 
      // Number of frequency boxes in each dimension (must a power of 2)
      const unsigned 
            N,

      // The sources in the polar frequency domain
      const vector< Source<R,d> >& 
            mySources,

      // 1d Chebyshev nodes that we extend to d-dimensions with tensor product
      const Array<R,q>& 
            chebyNodes,

      // The widths of the box in the freq. domain that this process owns
      const Array<R,d>& 
            myFreqBoxWidths,

      // The d-dimensional coordinates of the frequency box for this process
      const Array<unsigned,d>& 
            myFreqBox,

      // Log2 of number of frequency leaf boxes this process should init.
      const unsigned 
            log2LocalFreqBoxes,

      // Log2 of number of local frequency leaf boxes in each dimension 
      const Array<unsigned,d>& 
            log2LocalFreqBoxesPerDim,

      // The resultant equivalent weights for a low-rank expansion in each box
            vector< Array<complex<R>,Power<q,d>::value> >& 
            weights 
    )
    {
        typedef complex<R> C;

        const R wB = static_cast<R>(1) / N;

        Array<R,d> x0;
        for( unsigned j=0; j<d; ++j )
            x0[j] = 0.5;

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
                    cerr << "Source " << i << " was at " << pj
                         << " in dimension " << j << ", but our frequency box"
                         << " in this dim. is [" << leftBound << "," 
                         << rightBound << ")." << endl;
                    throw 0;
                }

                // We must be in the box, so bitwise determine the coord index
                // by bisection of box B_loc
                B[j] = 0;
                for( unsigned k=log2LocalFreqBoxesPerDim[j]; k>0; --k )
                {
                    const R middle = (rightBound-leftBound)/2.;
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

            // Flatten the integer coordinates of B_loc into a single index
            unsigned k = 0;
            for( unsigned j=0; j<d; ++j )
            {
                static unsigned log2LocalFreqBoxesUpToDim = 0;
                k |= (B[j]<<log2LocalFreqBoxesUpToDim);
                log2LocalFreqBoxesUpToDim += log2LocalFreqBoxesPerDim[j];
            }

            // Add this point's contribution to the unscaled weights of B. 
            // We evaluate the Lagrangian polynomial on the reference grid, 
            // so we need to map p to it first.
            Array<R,d> pRef;
            for( unsigned j=0; j<d; ++j )
                pRef[j] = (p[j]-p0[j])/wB;
            const C f = mySources[i].magnitude;
            const R alpha = TwoPi*N*Psi::Eval(x0,p);
            const C beta = C( cos(alpha), sin(alpha) ) * f;
            for( unsigned t=0; t<Power<q,d>::value; ++t )
                weights[i][t] += beta*Lagrange<R,d,q>( t, pRef, chebyNodes );
        }

        // Loop over all of the boxes to compute the {p_t^B} and prefactors
        // for each delta weight {delta_t^AB}, exp(-2 Pi i N Psi(x0,p_t^B) ).
        // Notice that if we are sharing a box with one or more processes, then
        // our local boxes, say B_loc, are subsets of a B, but we still need 
        // to consider our contribution to the weights corresponding to 
        // p_t^B lying in B \ B_loc.
        for( unsigned k=0; k<(1u<<log2LocalFreqBoxes); ++k ) 
        {
            // Compute the local integer coordinates of box k
            Array<unsigned,d> B;
            for( unsigned j=0; j<d; ++j )
            {
                static unsigned log2LocalFreqBoxesUpToDim = 0;
                // B[j] = (k/localFreqBoxesUpToDim) % localFreqBoxesPerDim[j]
                B[j] = (k>>log2LocalFreqBoxesUpToDim) & 
                       ((1u<<log2LocalFreqBoxesPerDim[j])-1);
                log2LocalFreqBoxesUpToDim += log2LocalFreqBoxesPerDim[j];
            }

            // Translate the local integer coordinates into the freq. center 
            Array<R,d> p0;
            for( unsigned j=0; j<d; ++j )
                p0[j] = myFreqBoxWidths[j]*myFreqBox[j] + B[j]*wB + wB/2;

            // Compute the prefactors given this p0 and multiply it by 
            // the corresponding weights
            for( unsigned t=0; t<Power<q,d>::value; ++t )
            {
                // Compute the physical location of pt
                Array<R,d> pt;
                unsigned qToThej = q;
                for( unsigned j=0; j<d; ++j )
                {
                    pt[j] = p0[j] + wB*chebyNodes[(t/qToThej)%q];
                    qToThej *= q;
                }

                const R alpha = -TwoPi*N*Psi::Eval(x0,pt);
                weights[k][t] *= C( cos(alpha), sin(alpha) );
            }
        }
    }
}

#endif /* BFIO_INITIALIZE_WEIGHTS_HPP */

