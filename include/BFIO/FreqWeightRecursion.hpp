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
#ifndef BFIO_FREQ_WEIGHT_RECURSION_HPP
#define BFIO_FREQ_WEIGHT_RECURSION_HPP 1

#include "BFIO/Lagrange.hpp"

namespace BFIO
{
    using namespace std;

    template<typename Phi,typename R,unsigned d,unsigned q>
    inline void
    FreqWeightRecursion
    ( const unsigned log2Procs,
      const unsigned myTeamRank,
      const unsigned N, 
      const vector< Array<R,d> >& chebyGrid,
      const Array<R,d>& x0A,
      const Array<R,d>& p0B,
      const R wB,
      const unsigned parentOffset,
      const WeightSetList<R,d,q>& oldWeightSetList,
            WeightSet<R,d,q>& weightSet            )
    {
        typedef complex<R> C;

        static bool initialized = false;
        static R pRefB[1<<d][Pow<q,d>::val][d];
        static R lagrangeFreqLookup[Pow<q,d>::val][1<<d][Pow<q,d>::val];

        if( !initialized )
        {
            for( unsigned t=0; t<Pow<q,d>::val; ++t )
            {
                for( unsigned c=0; c<(1u<<d); ++c )
                {
                    for( unsigned tp=0; tp<Pow<q,d>::val; ++tp )
                    {
                        // Map p_t'(Bc) to the reference domain of B and 
                        // store the Lagrangian evaluation
                        Array<R,d> ptpBcRefB;
                        for( unsigned j=0; j<d; ++j )
                        {
                            pRefB[c][tp][j] = ptpBcRefB[j] = 
                                ( (c>>j)&1 ? (2*chebyGrid[tp][j]+1)/4 :
                                             (2*chebyGrid[tp][j]-1)/4  );
                        }
                        lagrangeFreqLookup[t][c][tp] = 
                            Lagrange<R,d,q>( t, ptpBcRefB );
                    }
                }
            }
            initialized = true;
        }

        for( unsigned t=0; t<Pow<q,d>::val; ++t )
        {
            // Compute the unscaled weight
            weightSet[t] = 0;
            for( unsigned cLocal=0; cLocal<(1u<<(d-log2Procs)); ++cLocal )
            {
                const unsigned c = (cLocal<<log2Procs) + myTeamRank;
                const unsigned parentKey = parentOffset + cLocal;
                for( unsigned tp=0; tp<Pow<q,d>::val; ++tp )        
                {
                    // Scale and translate p_t'(Bc) on ref of B to p_t'
                    Array<R,d> ptp;
                    for( unsigned j=0; j<d; ++j )
                        ptp[j] = p0B[j] + wB*pRefB[c][tp][j];

                    const R alpha = TwoPi*Phi::Eval(x0A,ptp);
                    weightSet[t] += lagrangeFreqLookup[t][c][tp] *
                                    C( cos(alpha), sin(alpha) ) * 
                                    oldWeightSetList[parentKey][tp];
                }
            }

            // Scale the weight
            Array<R,d> ptB;
            for( unsigned j=0; j<d; ++j )
                ptB[j] = p0B[j] + wB*chebyGrid[t][j];
            const R alpha = -TwoPi*Phi::Eval(x0A,ptB);
            weightSet[t] *= C( cos(alpha), sin(alpha) );
        }
    }
}

#endif /* BFIO_FREQ_WEIGHT_RECURSION_HPP */

