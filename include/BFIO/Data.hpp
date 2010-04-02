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
#ifndef BFIO_DATA_HPP
#define BFIO_DATA_HPP 1

#include <complex>

namespace BFIO
{
    static const double Pi    = 3.141592653589793;
    static const double TwoPi = 6.283185307179586;

    // A d-dimensional point over arbitrary datatype T
    template<typename T,unsigned d>
    struct Array
    {
    private:
        T x[d];
    public:
        T& operator[]( unsigned j ) { return x[j]; }
        T operator[]( unsigned j ) const { return x[j]; }
    };

    // A d-dimensional coordinate in the polar frequency domain and the 
    // magnitude of the source located there
    template<typename R,unsigned d>
    struct Source 
    { 
        Array<R,d> p;
        std::complex<R> magnitude;
    };

    // Low-rank potential
    template<typename Psi,typename R,unsigned d,unsigned q>
    struct LRP
    {
        unsigned N;
        Array<R,d> x0;
        Array<R,d> points[ Power<q,d>::value ];
        std::complex<R> weights[ Power<q,d>::value ];
        
        std::complex<R> operator()( const Array<R,d>& x );
    };
}

// Inline implementations
namespace BFIO
{
    template<typename Psi,typename R,unsigned d,unsigned q>
    inline std::complex<R>
    LRP<Psi,R,d,q>::operator()( const Array<R,d>& x )
    {
        using namespace std;
        typedef complex<R> C;

        C value(0.,0.);
        for( unsigned j=0; j<Power<q,d>::value; ++j )
            value += exp( C(0.,TwoPi*N*Psi::Eval(x,points[j])) ) * weights[j];
        return value;
    }
}

#endif /* BFIO_DATA_HPP */

