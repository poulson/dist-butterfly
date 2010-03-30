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
#ifndef BFIO_LAGRANGE_H
#define BFIO_LAGRANGE_H 1

#include "BFIOPower.h"
#include "BFIOData.h"

namespace BFIO
{
   template<typename R,unsigned q,unsigned i,unsigned j>
    struct LagrangeKernel
    {
        static inline void 
        Eval( const R z, const Array<R,q>& chebyGrid, R& product )
        { 
            if( i != j ) 
            { product *= (z-chebyGrid[j])/(chebyGrid[i]-chebyGrid[j]); }
        }
    };
    
    template<typename R,unsigned q,unsigned i,unsigned j>
    struct LagrangeInnerLoopCore
    { 
        static inline void
        Eval( const R z, const Array<R,q>& chebyGrid, R& product ) 
        { LagrangeKernel<R,q,i,j>::Eval(z,chebyGrid,product);
          LagrangeInnerLoopCore<R,q,i,j-1>::Eval(z,chebyGrid,product); } 
    };

    template<typename R,unsigned q,unsigned i>
    struct LagrangeInnerLoopCore<R,q,i,0>
    { 
        static inline void 
        Eval( const R z, const Array<R,q>& chebyGrid, R& product ) 
        { LagrangeKernel<R,q,i,0>::Eval(z,chebyGrid,product); } 
    };

    template<typename R,unsigned q,unsigned i>
    struct LagrangeInnerLoop
    { 
        static inline void
        Eval( const R z, const Array<R,q>& chebyGrid, R& product ) 
        { LagrangeInnerLoopCore<R,q,i,q-1>::Eval(z,chebyGrid,product); } 
    };

    template<typename R,unsigned d,unsigned q,unsigned t,unsigned j>
    struct LagrangeOuterLoopCore
    {
        static inline void
        Eval( const Array<R,d>& z, const Array<R,q>& chebyGrid, R& product )
        { 
            // Pluck the j'th dimensional index out of t in order to 
            // accumulate the product in the j'th dimension.
            LagrangeInnerLoop<R,q, (t/Power<q,j>::value)%q >::Eval
            ( z[j], chebyGrid, product );
            LagrangeOuterLoopCore<R,d,q,t,j-1>::Eval( z, chebyGrid, product );
        }
    };

    template<typename R,unsigned d,unsigned q,unsigned t>
    struct LagrangeOuterLoopCore<R,d,q,t,0>
    {
        static inline void
        Eval( const Array<R,d>& z, const Array<R,q>& chebyGrid, R& product )
        {
            // Pluck the 0'th dimensional index out of t in order to
            // accumulate the product in the 0'th dimension
            LagrangeInnerLoop<R,q, t%q >::Eval
            ( z[0], chebyGrid, product );
        }
    };

    // Outer loop for Lagrangian interpolation. 
    template<typename R,unsigned d,unsigned q,unsigned t>
    struct LagrangeOuterLoop 
    {
        static inline void
        Eval( const Array<R,d>& z, const Array<R,q>& chebyGrid, R& product )
        { LagrangeOuterLoopCore<R,d,q,t,d-1>::Eval( z, chebyGrid, product ); }
    };

    template<typename R,unsigned d,unsigned q,unsigned t>
    struct LagrangeInterp
    {
        static inline R
        Eval( const Array<R,d>& z, const Array<R,q>& chebyGrid )
        {
            R product = static_cast<R>(1);        
            LagrangeOuterLoop<R,d,q,t>::Eval( z, chebyGrid, product );
            return product;
        }
    };
}

#endif /* BFIO_LAGRANGE_H */

