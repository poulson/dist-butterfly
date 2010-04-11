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
#ifndef BFIO_LAGRANGE_HPP
#define BFIO_LAGRANGE_HPP 1

#include "BFIO/Power.hpp"
#include "BFIO/Data.hpp"

namespace BFIO
{
    template<typename R,unsigned q,unsigned i,unsigned j>
    struct LagrangeKernel
    {
        static inline void 
        Eval( const R z, const Array<R,q>& chebyNodes, R& product )
        { 
            if( i != j ) 
            { product *= (z-chebyNodes[j])/(chebyNodes[i]-chebyNodes[j]); }
        }
    };
    
    template<typename R,unsigned q,unsigned i,unsigned j>
    struct LagrangeInnerLoop
    { 
        static inline void
        Eval( const R z, const Array<R,q>& chebyNodes, R& product ) 
        { LagrangeKernel<R,q,i,j>::Eval(z,chebyNodes,product);
          LagrangeInnerLoop<R,q,i,j-1>::Eval(z,chebyNodes,product); } 
    };

    template<typename R,unsigned q,unsigned i>
    struct LagrangeInnerLoop<R,q,i,0>
    { 
        static inline void 
        Eval( const R z, const Array<R,q>& chebyNodes, R& product ) 
        { LagrangeKernel<R,q,i,0>::Eval(z,chebyNodes,product); } 
    };

    template<typename R,unsigned d,unsigned q,unsigned t,unsigned j>
    struct LagrangeOuterLoop
    {
        static inline void
        Eval( const Array<R,d>& z, const Array<R,q>& chebyNodes, R& product )
        { 
            // Pluck the j'th dimensional index out of t in order to 
            // accumulate the product in the j'th dimension.
            LagrangeInnerLoop<R,q, (t/Power<q,j>::value)%q, q-1 >::Eval
            ( z[j], chebyNodes, product );
            LagrangeOuterLoop<R,d,q,t,j-1>::Eval( z, chebyNodes, product );
        }
    };

    template<typename R,unsigned d,unsigned q,unsigned t>
    struct LagrangeOuterLoop<R,d,q,t,0>
    {
        static inline void
        Eval( const Array<R,d>& z, const Array<R,q>& chebyNodes, R& product )
        {
            // Pluck the 0'th dimensional index out of t in order to
            // accumulate the product in the 0'th dimension
            LagrangeInnerLoop<R,q, t%q, q-1 >::Eval
            ( z[0], chebyNodes, product );
        }
    };

    template<typename R,unsigned d,unsigned q,unsigned t>
    struct Lagrange
    {
        static inline R
        Eval( const Array<R,d>& z, const Array<R,q>& chebyNodes )
        {
            R product = static_cast<R>(1);        
            LagrangeOuterLoop<R,d,q,t,d-1>::Eval( z, chebyNodes, product );
            return product;
        }
    };
}

#endif /* BFIO_LAGRANGE_HPP */

