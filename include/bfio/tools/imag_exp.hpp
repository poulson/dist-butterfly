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

#if defined(IBM)
# include "mass.h"
# include "massv.h"
#elif defined(INTEL)
# include "mkl_vml.h"
#endif

namespace bfio {

// For single-point imaginary exponentials
template<typename R>
inline std::complex<R>
ImagExp( R alpha );

template<>
inline std::complex<float>
ImagExp( float alpha )
{
    const float real = cos(alpha);
    const float imag = sin(alpha);
    return std::complex<float>( real, imag );
}

template<>
inline std::complex<double>
ImagExp( double alpha )
{
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
#if defined(IBM)
    int n = a.size(); 
    vssincos( const_cast<float*>(&a[0]), &sinResults[0], &cosResults[0], &n );
#elif defined(INTEL)
    vsSinCos( a.size(), &a[0], &sinResults[0], &cosResults[0] );
#else
    for( std::size_t j=0; j<a.size(); ++j )
    {
        sinResults[j] = sin(a[j]);
        cosResults[j] = cos(a[j]);
    }
#endif
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
#if defined(IBM)
    int n = a.size();
    vsincos( const_cast<double*>(&a[0]), &sinResults[0], &cosResults[0], &n );
#elif defined(INTEL)
    vdSinCos( a.size(), &a[0], &sinResults[0], &cosResults[0] );
#else
    for( std::size_t j=0; j<a.size(); ++j )
    {
        sinResults[j] = sin(a[j]);
        cosResults[j] = cos(a[j]);
    }
#endif
}

} // bfio

#endif // BFIO_TOOLS_IMAG_EXP_HPP

