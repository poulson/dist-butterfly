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
#ifndef BFIO_MPI_HPP
#define BFIO_MPI_HPP 1

#include <complex>
#include <stdexcept>
#include "mpi.h"

namespace bfio {

template<typename T>
void
SumScatter
( const T* sendBuf, T* recvBuf, int* recvCounts, MPI_Comm comm );

} // bfio

// Implementations for {float,double,complex<float>,complex<double>}
namespace bfio {

template<>
inline void
SumScatter<float>
( const float* sendBuf, float* recvBuf, int* recvCounts, MPI_Comm comm )
{
    int ierror = MPI_Reduce_scatter
    ( const_cast<float*>(sendBuf), recvBuf, recvCounts, MPI_FLOAT, MPI_SUM, 
      comm );
    if( ierror != 0 )
    {
        std::ostringstream msg;
        msg << "ierror from MPI_Reduce_scatter = " << ierror;
        throw std::runtime_error( msg.str() );
    }
}

template<>
inline void
SumScatter<double>
( const double* sendBuf, double* recvBuf, int* recvCounts, MPI_Comm comm )
{
    int ierror = MPI_Reduce_scatter
    ( const_cast<double*>(sendBuf), 
      recvBuf, recvCounts, MPI_DOUBLE, MPI_SUM, comm );
    if( ierror != 0 )
    {
        std::ostringstream msg;
        msg << "ierror from MPI_Reduce_scatter = " << ierror;
        throw std::runtime_error( msg.str() );
    }
}

template<>
inline void
SumScatter< std::complex<float> >
( const std::complex<float>* sendBuf, 
        std::complex<float>* recvBuf, 
  int* recvCounts, MPI_Comm comm )
{
#ifdef AVOID_COMPLEX_MPI
    int size;
    MPI_Comm_size( comm, &size );
    std::vector<int> recvCountsDoubled(size);
    for( int i=0; i<size; ++i )
        recvCountsDoubled[i] = 2*recvCounts[i];
    int ierror = MPI_Reduce_scatter
    ( const_cast<std::complex<float>*>(sendBuf), 
      recvBuf, &recvCountsDoubled[0], MPI_FLOAT, MPI_SUM, comm );
#else
    int ierror = MPI_Reduce_scatter
    ( const_cast<std::complex<float>*>(sendBuf), 
      recvBuf, recvCounts, MPI_COMPLEX, MPI_SUM, comm );
#endif
    if( ierror != 0 )
    {
        std::ostringstream msg;
        msg << "ierror from MPI_Reduce_scatter = " << ierror;
        throw std::runtime_error( msg.str() );
    }
}

template<>
void
inline SumScatter< std::complex<double> >
( const std::complex<double>* sendBuf, 
        std::complex<double>* recvBuf,
  int* recvCounts, MPI_Comm comm )
{
#ifdef AVOID_COMPLEX_MPI
    int size;
    MPI_Comm_size( comm, &size );
    std::vector<int> recvCountsDoubled(size);
    for( int i=0; i<size; ++i )
        recvCountsDoubled[i] = 2*recvCounts[i];
    int ierror = MPI_Reduce_scatter
    ( const_cast<std::complex<double>*>(sendBuf), 
      recvBuf, &recvCountsDoubled[0], MPI_DOUBLE, MPI_SUM, comm );
#else
    int ierror = MPI_Reduce_scatter
    ( const_cast<std::complex<double>*>(sendBuf), 
      recvBuf, recvCounts, MPI_DOUBLE_COMPLEX, MPI_SUM, comm );
#endif
    if( ierror != 0 )
    {
        std::ostringstream msg;
        msg << "ierror from MPI_Reduce_scatter = " << ierror;
        throw std::runtime_error( msg.str() );
    }
}

} // bfio

#endif /* BFIO_MPI_HPP */

