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
#ifndef BFIO_SCALE_WEIGHTS_HPP
#define BFIO_SCALE_WEIGHTS_HPP 1

#include "BFIO/ScaleWeights.hpp"

namespace BFIO
{

    template<typename R,unsigned d,unsigned q,unsigned t,unsigned j>
    struct ConstructPointLoop
    {
        static inline void
        Eval
        ( const R wB,
          const Array<R,d>& p0,
          const Array<R,q>& chebyGrid,
                Array<R,d>& pT        )
        {
            pT[j] = wB*chebyGrid[(t/Power<q,j>::value)%q]+p0[j];
            ConstructPointLoop<R,d,q,t,j-1>::Eval(wB,p0,chebyGrid,pT);
        }
    };

    template<typename R,unsigned d,unsigned q,unsigned t>
    struct ConstructPointLoop<R,d,q,t,0>
    {
        static inline void
        Eval
        ( const R wB,
          const Array<R,d>& p0,
          const Array<R,q>& chebyGrid,
                Array<R,d>& pT        )
        { pT[0] = wB*chebyGrid[t%q]+p0[0]; }
    };

    template<typename Psi,typename R,unsigned d,unsigned q,unsigned t>
    struct ScaleWeightsOuter
    {
        static inline void
        Eval
        ( const unsigned N,
          const R wB,
          const Array<R,d>& x0,
          const Array<R,d>& p0,
          const Array<R,q>& chebyGrid,
                Array<std::complex<R>,Power<q,d>::value>& weights )
        {
            using namespace std;
            typedef complex<R> C;

            Array<R,d> pT;
            ConstructPointLoop<R,d,q,t,d-1>::Eval(wB,p0,chebyGrid,pT);
            weights[t] *= exp( C(0.,-TwoPi*N*Psi::Eval(x0,pT)) );

            ScaleWeightsOuter<Psi,R,d,q,t-1>::Eval
            (N,wB,x0,p0,chebyGrid,weights);
        }
    };

    template<typename Psi,typename R,unsigned d,unsigned q>
    struct ScaleWeightsOuter<Psi,R,d,q,0>
    {
        static inline void
        Eval
        ( const unsigned N,
          const R wB,
          const Array<R,d>& x0,
          const Array<R,d>& p0,
          const Array<R,q>& chebyGrid,
                Array<std::complex<R>,Power<q,d>::value>& weights )
        {
            using namespace std; 
            typedef complex<R> C;

            Array<R,d> pT;
            ConstructPointLoop<R,d,q,0,d-1>::Eval(wB,p0,chebyGrid,pT);
            weights[0] *= exp( C(0.,-TwoPi*N*Psi::Eval(x0,pT)) );
        }
    };

    template<typename Psi,typename R,unsigned d,unsigned q>
    struct ScaleWeights
    {
        static inline void
        Eval
        ( const unsigned N,
          const R wB,
          const Array<R,d>& x0,
          const Array<R,d>& p0,
          const Array<R,q>& chebyGrid,
                Array<std::complex<R>,Power<q,d>::value>& weights )
        {
            ScaleWeightsOuter<Psi,R,d,q,Power<q,d>::value-1>::Eval
            ( N, wB, x0, p0, chebyGrid, weights );
        }
    };
}

#endif /* BFIO_SCALE_WEIGHTS_HPP */

