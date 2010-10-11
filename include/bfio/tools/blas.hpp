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
#ifndef BFIO_TOOLS_BLAS_HPP
#define BFIO_TOOLS_BLAS_HPP 1

#include <complex>

#ifdef FUNDERSCORE
#define C2F( name ) name ## _
#else
#define C2F( name ) name
#endif

namespace bfio {

template<typename R>
void Gemv
( char transa, int m, int n,
  R alpha, const R* A, int lda,
           const R* x, int incx,
  R beta,        R* y, int incy );

template<typename R>
void Gemm
( char transa, char transb, int m, int n, int k,
  R alpha, const R* A, int lda,
           const R* B, int ldb,
  R beta,        R* C, int ldc );

} // bfio

extern "C" {

void C2F(sgemv)
( const char* transa,
  const int* m, const int* n,
  const float* alpha, const float* A, const int* lda,
                      const float* x, const int* incx,
  const float* beta,        float* y, const int* incy );

void C2F(dgemv)
( const char* transa,
  const int* m, const int* n,
  const double* alpha, const double* A, const int* lda,
                       const double* x, const int* incx,
  const double* beta,        double* y, const int* incy );

void C2F(sgemm)
( const char* transA, const char* transB,
  const int* m, const int* n, const int* k,
  const float* alpha, const float* A, const int* lda,
                      const float* B, const int* ldb,
  const float* beta,        float* C, const int* ldc );

void C2F(dgemm)
( const char* transA, const char* transB,
  const int* m, const int* n, const int* k,
  const double* alpha, const double* A, const int* lda,
                       const double* B, const int* ldb,
  const double* beta,        double* C, const int* ldc );

} // extern "C"

// Implementations
namespace bfio {

template<>
inline void
Gemv<float>
( char transa, int m, int n,
  float alpha, const float* A, int lda,
               const float* x, int incx,
  float beta,        float* y, int incy )
{
    C2F(sgemv)
    ( &transa, &m, &n,
      &alpha, A, &lda, x, &incx, &beta, y, &incy );
}

template<>
inline void
Gemv<double>
( char transa, int m, int n,
  double alpha, const double* A, int lda,
                const double* x, int incx,
  double beta,        double* y, int incy )
{
    C2F(dgemv)
    ( &transa, &m, &n,
      &alpha, A, &lda, x, &incx, &beta, y, &incy );
}

template<>
inline void
Gemm<float>
( char transa, char transb, int m, int n, int k,
  float alpha, const float* A, int lda,
               const float* B, int ldb,
  float beta,        float* C, int ldc )
{
    C2F(sgemm)
    ( &transa, &transb, &m, &n, &k,
      &alpha, A, &lda, B, &ldb, &beta, C, &ldc );
}
 
template<>
inline void
Gemm<double>
( char transa, char transb, int m, int n, int k,
  double alpha, const double* A, int lda,
                const double* B, int ldb,
  double beta,        double* C, int ldc )
{
    C2F(dgemm)
    ( &transa, &transb, &m, &n, &k,
      &alpha, A, &lda, B, &ldb, &beta, C, &ldc );
}

} // bfio

#endif // BFIO_TOOLS_BLAS_HPP

