/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of ButterflyFIO and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef BFIO_TOOLS_IMAG_EXP_HPP
#define BFIO_TOOLS_IMAG_EXP_HPP

#include <math.h>
#include <vector>

#if defined(MASS)
# include "mass.h"
# include "massv.h"
#elif defined(MKL)
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
std::complex<double>
ImagExp( double alpha )
{
    const double real = cos(alpha);
    const double imag = sin(alpha);
    return std::complex<double>( real, imag );
}

// For performing many sin computations
template<typename R>
void
SinBatch
( const std::vector<R>& a, 
        std::vector<R>& sinResults );

template<>
void
SinBatch
( const std::vector<float>& a, 
        std::vector<float>& sinResults )
{
    const int n = a.size();
    sinResults.resize( n );
#if defined(MASS)
    vssin( const_cast<float*>(&a[0]), &sinResults[0], &n );
#elif defined(MKL)
    vsSin( n, &a[0], &sinResults[0] );
#else
    for( int j=0; j<n; ++j )
        sinResults[j] = sin(a[j]);
#endif
}

template<>
void
SinBatch
( const std::vector<double>& a, 
        std::vector<double>& sinResults )
{
    const int n = a.size();
    sinResults.resize( n );
#if defined(MASS)
    vsin( const_cast<double*>(&a[0]), &sinResults[0], &n );
#elif defined(MKL)
    vdSin( n, &a[0], &sinResults[0] );
#else
    for( int j=0; j<n; ++j )
        sinResults[j] = sin(a[j]);
#endif
}

// For performing many cos computations
template<typename R>
void
CosBatch
( const std::vector<R>& a, 
        std::vector<R>& cosResults );

template<>
void
CosBatch
( const std::vector<float>& a, 
        std::vector<float>& cosResults )
{
    const int n = a.size();
    cosResults.resize( n );
#if defined(MASS)
    vscos( const_cast<float*>(&a[0]), &cosResults[0], &n );
#elif defined(MKL)
    vsCos( n, &a[0], &cosResults[0] );
#else
    for( int j=0; j<n; ++j )
        cosResults[j] = cos(a[j]);
#endif
}

template<>
void
CosBatch
( const std::vector<double>& a, 
        std::vector<double>& cosResults )
{
    const int n = a.size();
    cosResults.resize( n );
#if defined(MASS)
    vcos( const_cast<double*>(&a[0]), &cosResults[0], &n );
#elif defined(MKL)
    vdCos( n, &a[0], &cosResults[0] );
#else
    for( int j=0; j<n; ++j )
        cosResults[j] = cos(a[j]);
#endif
}

// For performing many sin(a)/cos(a) pairs
template<typename R>
void
SinCosBatch
( const std::vector<R>& a, 
        std::vector<R>& sinResults,
        std::vector<R>& cosResults );

template<>
void
SinCosBatch
( const std::vector<float>& a, 
        std::vector<float>& sinResults,
        std::vector<float>& cosResults ) 
{
    const int n = a.size();
    sinResults.resize( n );
    cosResults.resize( n );
#if defined(MASS)
    vssincos( const_cast<float*>(&a[0]), &sinResults[0], &cosResults[0], &n );
#elif defined(MKL)
    vsSinCos( n, &a[0], &sinResults[0], &cosResults[0] );
#else
    for( int j=0; j<n; ++j )
    {
        sinResults[j] = sin(a[j]);
        cosResults[j] = cos(a[j]);
    }
#endif
}

template<>
void
SinCosBatch
( const std::vector<double>& a, 
        std::vector<double>& sinResults,
        std::vector<double>& cosResults )
{
    const int n = a.size();
    sinResults.resize( n );
    cosResults.resize( n );
#if defined(MASS)
    vsincos( const_cast<double*>(&a[0]), &sinResults[0], &cosResults[0], &n );
#elif defined(MKL)
    vdSinCos( n, &a[0], &sinResults[0], &cosResults[0] );
#else
    for( int j=0; j<n; ++j )
    {
        sinResults[j] = sin(a[j]);
        cosResults[j] = cos(a[j]);
    }
#endif
}

// For performing many sqrt computations
template<typename R>
void
SqrtBatch
( const std::vector<R>& a, 
        std::vector<R>& sqrtResults );

template<>
void
SqrtBatch
( const std::vector<float>& a, 
        std::vector<float>& sqrtResults )
{
    const int n = a.size();
    sqrtResults.resize( n );
#if defined(MASS)
    vssqrt( const_cast<float*>(&a[0]), &sqrtResults[0], &n );
#elif defined(MKL)
    vsSqrt( n, &a[0], &sqrtResults[0] );
#else
    for( int j=0; j<n; ++j )
        sqrtResults[j] = sqrt(a[j]);
#endif
}

template<>
void
SqrtBatch
( const std::vector<double>& a, 
        std::vector<double>& sqrtResults )
{
    const int n = a.size();
    sqrtResults.resize( n );
#if defined(MASS)
    vsqrt( const_cast<double*>(&a[0]), &sqrtResults[0], &n );
#elif defined(MKL)
    vdSqrt( n, &a[0], &sqrtResults[0] );
#else
    for( int j=0; j<n; ++j )
        sqrtResults[j] = sqrt(a[j]);
#endif
}

} // bfio

#endif // ifndef BFIO_TOOLS_IMAG_EXP_HPP
