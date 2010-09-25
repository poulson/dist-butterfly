/*
   ButterflyFIO: a distributed-memory fast algorithm for applying FIOs.
   Copyright (C) 2010 Jack Poulson <jack.poulson@gmail.com>
 
   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
 
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
 
   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef BFIO_TOOLS_IMAG_EXP_HPP
#define BFIO_TOOLS_IMAG_EXP_HPP 1

#include <math.h>
#include <vector>

namespace bfio {

// For single-point imaginary exponentials
template<typename R>
inline std::complex<R>
ImagExp( R alpha );

template<>
inline std::complex<float>
ImagExp( float alpha )
{
    // TODO: Add fast sincos support for various architectures
    const float real = cos(alpha);
    const float imag = sin(alpha);
    return std::complex<float>( real, imag );
}

template<>
inline std::complex<double>
ImagExp( double alpha )
{
    // TODO: Add fast sincos support for various architectures
    const double real = cos(alpha);
    const double imag = sin(alpha);
    return std::complex<double>( real, imag );
}

// For performing many sin(a)/cos(a) pairs
template<typename R>
inline void
SinCosBatch
( const std::vector<R>& a, 
        std::vector<R>& sinResults,
        std::vector<R>& cosResults );

template<>
inline void
SinCosBatch
( const std::vector<float>& a, 
        std::vector<float>& sinResults,
        std::vector<float>& cosResults ) 
{
    sinResults.resize( a.size() );
    cosResults.resize( a.size() );
    // TODO: Add vectorization support here for various architectures
    for( std::size_t j=0; j<a.size(); ++j )
    {
        sinResults[j] = sin(a[j]);
        cosResults[j] = cos(a[j]);
    }
}

template<>
inline void
SinCosBatch
( const std::vector<double>& a, 
        std::vector<double>& sinResults,
        std::vector<double>& cosResults )
{
    sinResults.resize( a.size() );
    cosResults.resize( a.size() );
    // TODO: Add vectorization support here for various architectures
    for( std::size_t j=0; j<a.size(); ++j )
    {
        sinResults[j] = sin(a[j]);
        cosResults[j] = cos(a[j]);
    }
}

} // bfio

#endif // BFIO_TOOLS_IMAG_EXP_HPP

