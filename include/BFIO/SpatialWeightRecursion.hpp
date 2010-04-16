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
#ifndef BFIO_SPATIAL_WEIGHT_RECURSION_HPP
#define BFIO_SPATIAL_WEIGHT_RECURSION_HPP 1

#include "BFIO/Lagrange.hpp"

namespace BFIO
{
    using namespace std;

    template<typename Phi,typename R,unsigned d,unsigned q>
    inline void
    SpatialWeightRecursion
    ( const unsigned log2Procs,
      const unsigned myTeamRank,
      const unsigned N, 
      const vector< Array<R,d> >& chebyGrid,
      const unsigned ARelativeToAp,
      const Array<R,d>& x0A,
      const Array<R,d>& x0Ap,
      const Array<R,d>& p0B,
      const R wA,
      const R wB,
      const unsigned parentOffset,
      const WeightSetList<R,d,q>& oldWeightSetList,
            WeightSet<R,d,q>& weightSet             )
    {
        typedef complex<R> C;

        static bool initialized = false;
        static R lagrangeSpatialLookup[Pow<q,d>::val][1<<d][Pow<q,d>::val];

        if( !initialized )
        {
            for( unsigned t=0; t<Pow<q,d>::val; ++t )
            {
                for( unsigned c=0; c<(1u<<d); ++c )
                {
                    // Map x_t(A) to the reference domain of its parent
                    Array<R,d> xtARefAp;
                    for( unsigned j=0; j<d; ++j )
                    {
                        xtARefAp[j] = 
                            ( (c>>j)&1 ? (2*chebyGrid[t][j]+1)/4 :
                                         (2*chebyGrid[t][j]-1)/4  );
                    }
                    for( unsigned tp=0; tp<Pow<q,d>::val; ++tp )
                    {
                        lagrangeSpatialLookup[t][c][tp] = 
                            Lagrange<R,d,q>( tp, xtARefAp );
                    }
                }
            }
            initialized = true;
        }

        for( unsigned t=0; t<Pow<q,d>::val; ++t )
        {
            // Compute xt(A)
            Array<R,d> xtA;
            for( unsigned j=0; j<d; ++j )
                xtA[j] = x0A[j] + wA*chebyGrid[t][j];

            // Compute the unscaled weight
            weightSet[t] = 0;
            for( unsigned cLocal=0; cLocal<(1u<<(d-log2Procs)); ++cLocal )
            {
                const unsigned c = (cLocal<<log2Procs) + myTeamRank;
                const unsigned parentKey = parentOffset + cLocal;
                C childContribution( 0, 0 );

                // Compute p0(Bc)
                Array<R,d> p0Bc;
                for( unsigned j=0; j<d; ++j )
                    p0Bc[j] = p0B[j] + ( (c>>j)&1 ? wB/4 : -wB/4 );

                for( unsigned tp=0; tp<Pow<q,d>::val; ++tp )        
                {
                    // Compute xtp(Ap)
                    Array<R,d> xtpAp;
                    for( unsigned j=0; j<d; ++j )
                        xtpAp[j] = x0Ap[j] + (wA*2)*chebyGrid[tp][j];

                    const R alpha = -TwoPi*Phi::Eval(xtpAp,p0Bc);
                    childContribution +=
                        lagrangeSpatialLookup[t][ARelativeToAp][tp] *
                        C( cos(alpha), sin(alpha) ) * 
                        oldWeightSetList[parentKey][tp];
                }
                
                // Scale the child contribution and add to weightSet[t]
                const R alpha = TwoPi*Phi::Eval(xtA,p0Bc);
                childContribution *= C( cos(alpha), sin(alpha) );
                weightSet[t] += childContribution;
            }
        }
    }
}

#endif /* BFIO_SPATIAL_WEIGHT_RECURSION_HPP */

