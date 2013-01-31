/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of ButterflyFIO and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef BFIO_TOOLS_LAPACK_HPP
#define BFIO_TOOLS_LAPACK_HPP

#include <complex>

#if defined(LAPACK_POST)
#define LAPACK(name) name ## _
#else
#define LAPACK(name) name
#endif

namespace bfio {

typedef std::complex<float> scomplex;
typedef std::complex<double> dcomplex;

template<typename T>
void LU
( int m, int n, T* A, int lda, int* pivot );

template<typename T>
void InvertLU
( int m, T* A, int lda, const int* pivot, T* work, int lwork );

} // bfio

extern "C" {

void LAPACK(sgetrf)
( const int* m, const int* n, float* A, const int* lda, int* pivot, int* info );

void LAPACK(dgetrf)
( const int* m, const int* n, double* A, const int* lda, int* pivot, 
  int* info );

void LAPACK(cgetrf)
( const int* m, const int* n, bfio::scomplex* A, const int* lda, int* pivot, 
  int* info );

void LAPACK(zgetrf)
( const int* m, const int* n, bfio::dcomplex* A, const int* lda, int* pivot, 
  int* info );

void LAPACK(sgetri)
( const int* m, float* A, const int* lda, const int* pivot, float* work, 
  const int* lwork, int* info );

void LAPACK(dgetri)
( const int* m, double* A, const int* lda, const int* pivot, double* work, 
  const int* lwork, int* info );

void LAPACK(cgetri)
( const int* m, bfio::scomplex* A, const int* lda, const int* pivot, 
  bfio::scomplex* work, const int* lwork, int* info );

void LAPACK(zgetri)
( const int* m, bfio::dcomplex* A, const int* lda, const int* pivot, 
  bfio::dcomplex* work, const int* lwork, int* info );

} // extern "C"

// Implementations
namespace bfio {

template<>
inline void
LU<float>
( int m, int n, float* A, int lda, int* pivot )
{
    int info;
    LAPACK(sgetrf)( &m, &n, A, &lda, pivot, &info );
}

template<>
inline void
LU<double>
( int m, int n, double* A, int lda, int* pivot )
{
    int info;
    LAPACK(dgetrf)( &m, &n, A, &lda, pivot, &info );
}

template<>
inline void
LU<scomplex>
( int m, int n, scomplex* A, int lda, int* pivot )
{
    int info;
    LAPACK(cgetrf)( &m, &n, A, &lda, pivot, &info );
}

template<>
inline void
LU<dcomplex>
( int m, int n, dcomplex* A, int lda, int* pivot )
{
    int info;
    LAPACK(zgetrf)( &m, &n, A, &lda, pivot, &info );
}

template<>
inline void
InvertLU<float>
( int m, float* A, int lda, const int* pivot, float* work, int lwork )
{
    int info;
    LAPACK(sgetri)( &m, A, &lda, pivot, work, &lwork, &info );
}

template<>
inline void
InvertLU<double>
( int m, double* A, int lda, const int* pivot, double* work, int lwork )
{
    int info;
    LAPACK(dgetri)( &m, A, &lda, pivot, work, &lwork, &info );
}

template<>
inline void
InvertLU<scomplex>
( int m, scomplex* A, int lda, const int* pivot, scomplex* work, int lwork )
{
    int info;
    LAPACK(cgetri)( &m, A, &lda, pivot, work, &lwork, &info );
}

template<>
inline void
InvertLU<dcomplex>
( int m, dcomplex* A, int lda, const int* pivot, dcomplex* work, int lwork )
{
    int info;
    LAPACK(zgetri)( &m, A, &lda, pivot, work, &lwork, &info );
}

} // bfio

#endif // ifndef BFIO_TOOLS_LAPACK_HPP
