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
#ifndef BFIO_LRP_HPP
#define BFIO_LRP_HPP 1

#include "BFIO/Data.hpp"
#include "BFIO/Lagrange.hpp"

namespace BFIO
{
    using namespace std;

    // Low-rank potential
    template<typename Phi,typename R,unsigned d,unsigned q>
    struct LRP
    {
        unsigned N;
        Array<R,d> x0;
        Array<R,d> p0;
        Array< Array<R,d>, Pow<q,d>::val > pointSet;
        WeightSet<R,d,q> weightSet;

        LRP() 
        { }
        
        complex<R> operator()( const Array<R,d>& x );
    };
}

// Inline implementations
namespace BFIO
{
    template<typename Phi,typename R,unsigned d,unsigned q>
    inline complex<R>
    LRP<Phi,R,d,q>::operator()( const Array<R,d>& x )
    {
        typedef complex<R> C;

        // Convert x to the reference domain of [-1/2,+1/2]^d
        Array<R,d> xRef;
        for( unsigned j=0; j<d; ++j )
            xRef[j] = (x[j]-0.5)/N;

        C value(0.,0.);
        for( unsigned t=0; t<Pow<q,d>::val; ++t )
        {
            R alpha = -TwoPi*Phi::Eval(pointSet[t],p0);
            value += Lagrange<R,d,q>( t, xRef ) * 
                     C( cos(alpha), sin(alpha) ) * weightSet[t];
        }
        R alpha = TwoPi*Phi::Eval(x,p0);
        value *= C( cos(alpha), sin(alpha) );
        return value;
    }
}

#endif /* BFIO_LRP_HPP */

