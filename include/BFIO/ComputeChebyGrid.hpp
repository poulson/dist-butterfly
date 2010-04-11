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
#ifndef BFIO_COMPUTE_CHEBY_GRID_HPP
#define BFIO_COMPUTE_CHEBY_GRID_HPP 1

#include "BFIO/Power.hpp"
#include "BFIO/Data.hpp"

namespace BFIO
{
    template<typename R,unsigned d,unsigned q,unsigned t,unsigned j>
    struct ComputeChebyGridInnerLoop
    {
        static inline void
        Eval
        ( const Array<R,q>& chebyNodes, Array<R,d>& point )
        {
            const unsigned i = (t/Power<q,j>::value)%q;
            point[j] = chebyNodes[i];

            ComputeChebyGridInnerLoop<R,d,q,t,j-1>::Eval( chebyNodes, point );
        }
    };

    template<typename R,unsigned d,unsigned q,unsigned t>
    struct ComputeChebyGridInnerLoop<R,d,q,t,0>
    {
        static inline void
        Eval
        ( const Array<R,q>& chebyNodes, Array<R,d>& point )
        {
            const unsigned i = t%q;
            point[0] = chebyNodes[i];
        }
    };

    template<typename R,unsigned d,unsigned q,unsigned t>
    struct ComputeChebyGridLoop
    {
        static inline void
        Eval
        ( const Array<R,q>& chebyNodes,
                Array< Array<R,d>,Power<q,d>::value >& chebyGrid )
        {
            ComputeChebyGridInnerLoop<R,d,q,t,d-1>::Eval
            ( chebyNodes, chebyGrid[t] );

            ComputeChebyGridLoop<R,d,q,t-1>::Eval( chebyNodes, chebyGrid );
        }
    };

    template<typename R,unsigned d,unsigned q>
    struct ComputeChebyGridLoop<R,d,q,0>
    {
        static inline void
        Eval
        ( const Array<R,q>& chebyNodes,
                Array< Array<R,d>,Power<q,d>::value >& chebyGrid )
        {
            ComputeChebyGridInnerLoop<R,d,q,0,d-1>::Eval
            ( chebyNodes, chebyGrid[0] );
        }
    };

    template<typename R,unsigned d,unsigned q>
    inline void
    ComputeChebyGrid
    ( const Array<R,q>& chebyNodes, 
            Array< Array<R,d>,Power<q,d>::value >& chebyGrid )
    {
        ComputeChebyGridLoop<R,d,q,Power<q,d>::value-1>::Eval
        ( chebyNodes, chebyGrid );        
    }
}

#endif /* BFIO_COMPUTE_CHEBY_GRID_HPP */

