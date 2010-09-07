/*
   Copyright (c) 2010, Jack Poulson
   All rights reserved.

   This file is part of ButterflyFIO.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are met:

    - Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    - Neither the name of the owner nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
   ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
   LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
   CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
   SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
   INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
   CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
   ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
   POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once
#ifndef BFIO_BLAS_HPP
#define BFIO_BLAS_HPP 1

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
      &alpha, (float*)x, &realsPerComplex, A, &lda, 
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

#endif /* BFIO_BLAS_HPP */

