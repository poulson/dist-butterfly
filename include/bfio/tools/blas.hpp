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
void
RealMatrixComplexVec
( int m, int n, 
  const R alpha, const R* A, int lda, 
                 const std::complex<R>* x, 
  const R beta,        std::complex<R>* y );

} // bfio

extern "C" {

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

// Implementations for {float,double,complex<float>,complex<double>}
namespace bfio {

template<>
inline void
RealMatrixComplexVec<float>
( int m, int n,
  const float alpha, const float* A, int lda, 
  const std::complex<float>* x,
  const float beta,  std::complex<float>* y )
{
    const int realsPerComplex = 2;
    const char normal = 'N';
    const char transpose = 'T';
    C2F(sgemm)
    ( &normal, &transpose, &realsPerComplex, &n, &m,
      &alpha, (const float*)x, &realsPerComplex, A, &lda, 
      &beta,  (float*)y, &realsPerComplex );
}
    
template<>
inline void
RealMatrixComplexVec<double>
( int m, int n,
  const double alpha, const double* A, int lda, 
  const std::complex<double>* x,
  const double beta,  std::complex<double>* y )
{
    const int realsPerComplex = 2;
    const char normal = 'N';
    const char transpose = 'T';
    C2F(dgemm)
    ( &normal, &transpose, &realsPerComplex, &n, &m,
      &alpha, (const double*)x, &realsPerComplex, A, &lda,
      &beta,  (double*)y, &realsPerComplex );
}

} // bfio

#endif // BFIO_TOOLS_BLAS_HPP

