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
#include "BFIO/MapToGlobalLoop.hpp"

namespace BFIO
{
    using namespace std;

    template<typename R,unsigned d,unsigned q,unsigned tp,unsigned j>
    struct MapToParentLoop
    {
        static inline void    
        Eval
        ( const Array< Array<R,d>,Power<q,d>::value >& chebyGrid, 
          const unsigned c, Array<R,d>& pRef                     )
        { 
            pRef[j] = ( (c>>j)&1 ? 
                        (2*chebyGrid[tp][j]+1)/4 :
                        (2*chebyGrid[tp][j]-1)/4  );
            MapToParentLoop<R,d,q,tp,j-1>::Eval( chebyGrid, c, pRef );
        }
    };

    template<typename R,unsigned d,unsigned q,unsigned tp>
    struct MapToParentLoop<R,d,q,tp,0>
    {
        static inline void
        Eval
        ( const Array< Array<R,d>,Power<q,d>::value >& chebyGrid, 
          const unsigned c, Array<R,d>& pRef                     )
        {
            pRef[0] = ( c&1 ? 
                        (2*chebyGrid[tp][0]+1)/4 : 
                        (2*chebyGrid[tp][0]-1)/4  );
        }
    };

    template<typename Psi,typename R,unsigned d,
             unsigned q,unsigned t,unsigned c,unsigned tp>
    struct FreqWeightRecursionInnerWeightLoop
    {
        static inline void
        Eval
        ( const unsigned N,
          const Array<R,q>& chebyNodes,
          const Array< Array<R,d>,Power<q,d>::value >& chebyGrid,
          const Array<R,d>& x0,
          const Array<R,d>& p0,
          const R wB,
          const Array<complex<R>,Power<q,d>::value>& oldWeights,
                complex<R>& delta                               )
        {
            typedef complex<R> C;

            // Compute the t-prime point in Bc mapped to the reference grid of B
            Array<R,d> pRef;
            MapToParentLoop<R,d,q,tp,d-1>::Eval( chebyGrid, c, pRef );

            // Scale and translate pRef to its actual location
            Array<R,d> p;
            for( unsigned j=0; j<d; ++j )
                p[j] = p0[j] + wB*pRef[j];

            delta += Lagrange<R,d,q,t>::Eval( pRef, chebyNodes ) * 
                     exp( C(0,2*Pi*N*Psi::Eval(x0,p)) ) * oldWeights[tp];

            FreqWeightRecursionInnerWeightLoop<Psi,R,d,q,t,c,tp-1>::Eval
            ( N, chebyNodes, chebyGrid, x0, p0, wB, oldWeights, delta );
        }
    };

    template<typename Psi,typename R,unsigned d,
             unsigned q,unsigned t,unsigned c   >
    struct FreqWeightRecursionInnerWeightLoop<Psi,R,d,q,t,c,0>
    {
        static inline void
        Eval
        ( const unsigned N,
          const Array<R,q>& chebyNodes,
          const Array< Array<R,d>,Power<q,d>::value >& chebyGrid,
          const Array<R,d>& x0,
          const Array<R,d>& p0,
          const R wB,
          const Array<complex<R>,Power<q,d>::value>& oldWeights,
                complex<R>& delta                               )
        {
            typedef complex<R> C;

            // Compute the t-prime point in Bc mapped to the reference grid of B
            Array<R,d> pRef;
            MapToParentLoop<R,d,q,0,d-1>::Eval( chebyGrid, c, pRef );

            // Scale and translate pRef to its actual location
            Array<R,d> p;
            for( unsigned j=0; j<d; ++j )
                p[j] = p0[j] + wB*pRef[j];

            delta += Lagrange<R,d,q,t>::Eval( pRef, chebyNodes ) * 
                     exp( C(0,2*Pi*N*Psi::Eval(x0,p)) ) * oldWeights[0];
        }
    };

    template<typename Psi,typename R,unsigned d,
             unsigned q,unsigned t,unsigned c   >
    struct FreqWeightRecursionChildLoop
    {
        static inline void
        Eval
        ( const unsigned N,
          const Array<R,q>& chebyNodes,
          const Array< Array<R,d>,Power<q,d>::value >& chebyGrid,
          const Array<R,d>& x0,
          const Array<R,d>& p0,
          const R wB,
          const unsigned parentPairOffset,
          const vector< Array<complex<R>,Power<q,d>::value> >& oldWeights,
                complex<R>& delta                                         )
        {
            const unsigned parentPairKey = parentPairOffset + c;

            FreqWeightRecursionInnerWeightLoop
            <Psi,R,d,q,t,c,Power<q,d>::value-1>::Eval
            ( N, chebyNodes, chebyGrid, x0, p0, wB, 
              oldWeights[parentPairKey], delta     );

            FreqWeightRecursionChildLoop<Psi,R,d,q,t,c-1>::Eval
            ( N, chebyNodes, chebyGrid, x0, p0, wB, 
              parentPairOffset, oldWeights, delta  );
        }
    };

    template<typename Psi,typename R,unsigned d,unsigned q,unsigned t>
    struct FreqWeightRecursionChildLoop<Psi,R,d,q,t,0>
    {
        static inline void
        Eval
        ( const unsigned N,
          const Array<R,q>& chebyNodes,
          const Array< Array<R,d>,Power<q,d>::value >& chebyGrid,
          const Array<R,d>& x0,
          const Array<R,d>& p0,
          const R wB,
          const unsigned parentPairOffset,
          const vector< Array<complex<R>,Power<q,d>::value> >& oldWeights,
                complex<R>& delta                                         )
        {
            const unsigned parentPairKey = parentPairOffset;
            FreqWeightRecursionInnerWeightLoop
            <Psi,R,d,q,t,0,Power<q,d>::value-1>::Eval
            ( N, chebyNodes, chebyGrid, x0, p0, 
              wB, oldWeights[parentPairKey], delta );
        }
    };

    template<typename Psi,typename R,unsigned d,unsigned q,unsigned t>
    struct FreqWeightRecursionOuterWeightLoop
    {
        static inline void
        Eval
        ( const unsigned N,
          const Array<R,q>& chebyNodes,
          const Array< Array<R,d>,Power<q,d>::value >& chebyGrid,
          const Array<R,d>& x0,
          const Array<R,d>& p0,
          const R wB,
          const unsigned parentPairOffset,
          const vector< Array<complex<R>,Power<q,d>::value> >& oldWeights,
                Array<complex<R>,Power<q,d>::value> weights               )
        {
            typedef complex<R> C;

            // Convert a downward loop into an upward loop
            const unsigned weightIdx = Power<q,d>::value-1-t;

            // Compute the unscaled weight
            weights[weightIdx] = static_cast<R>(0);
            FreqWeightRecursionChildLoop<Psi,R,d,q,t,(1<<d)-1>::Eval
            ( N, chebyNodes, chebyGrid, x0, p0, wB, 
              parentPairOffset, oldWeights, weights[weightIdx] );

            // Scale the weight
            Array<R,d> p;
            for( unsigned j=0; j<d; ++j )
                p[j] = p0[j] + wB*chebyGrid[t][j];
            weights[weightIdx] *= exp( C(0,-2*Pi*N*Psi::Eval(x0,p)) );

            // Continue looping over the weights
            FreqWeightRecursionOuterWeightLoop<Psi,R,d,q,t-1>::Eval
            ( N, chebyNodes, chebyGrid, x0, p0, wB, 
              parentPairOffset, oldWeights, weights );
        }
    };

    template<typename Psi,typename R,unsigned d,unsigned q>
    struct FreqWeightRecursionOuterWeightLoop<Psi,R,d,q,0>
    {
        static inline void
        Eval
        ( const unsigned N,
          const Array<R,q>& chebyNodes,
          const Array< Array<R,d>,Power<q,d>::value >& chebyGrid,
          const Array<R,d>& x0,
          const Array<R,d>& p0,
          const R wB,
          const unsigned parentPairOffset,
          const vector< Array<complex<R>,Power<q,d>::value> >& oldWeights,
                Array<complex<R>,Power<q,d>::value> weights               )
        {
            typedef complex<R> C;

            // Convert a downward loop into an upward loop
            const unsigned weightIdx = Power<q,d>::value-1;

            // Compute the unscaled weight
            weights[weightIdx] = static_cast<R>(0);
            FreqWeightRecursionChildLoop<Psi,R,d,q,0,(1<<d)-1>::Eval
            ( N, chebyNodes, chebyGrid, x0, p0, wB,
              parentPairOffset, oldWeights, weights[weightIdx] );
            
            // Scale the weight
            Array<R,d> p;
            for( unsigned j=0; j<d; ++j )
                p[j] = p0[j] + wB*chebyGrid[0][j];
            weights[weightIdx] *= exp( C(0,-2*Pi*N*Psi::Eval(x0,p)) );
        }
    };

    template<typename Psi,typename R,unsigned d,unsigned q>
    inline void
    FreqWeightRecursion
    ( const unsigned N, 
      const Array<R,q>& chebyNodes,
      const Array< Array<R,d>,Power<q,d>::value >& chebyGrid,
      const Array<R,d>& x0,
      const Array<R,d>& p0,
      const R wB,
      const unsigned parentPairOffset,
      const vector< Array<complex<R>,Power<q,d>::value> >& oldWeights,
            Array<complex<R>,Power<q,d>::value> weights               )
    {
        FreqWeightRecursionOuterWeightLoop
        <Psi,R,d,q,Power<q,d>::value-1>::Eval
        ( N, chebyNodes, chebyGrid, x0, p0, wB, 
          parentPairOffset, oldWeights, weights );
    }
}

#endif /* BFIO_FREQ_WEIGHT_RECURSION_HPP */

