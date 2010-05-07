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
#ifndef BFIO_MPI_HPP
#define BFIO_MPI_HPP 1

#include <complex>
#include "mpi.h"

namespace bfio {

template<typename T>
void
SumScatter
( T* sendBuf, T* recvBuf, int* recvCounts, MPI_Comm comm );

} // bfio

// Implementations for {float,double,complex<float>,complex<double>}
namespace bfio {

template<>
inline void
SumScatter<float>
( float* sendBuf, float* recvBuf, int* recvCounts, MPI_Comm comm )
{
    int ierror = MPI_Reduce_scatter
    ( sendBuf, recvBuf, recvCounts, MPI_FLOAT, MPI_SUM, comm );
    if( ierror != 0 )
    {
        std::ostringstream msg;
        msg << "ierror from MPI_Reduce_scatter = " << ierror;
        const std::string& s = msg.str();
        throw s.c_str();
    }
}

template<>
inline void
SumScatter<double>
( double* sendBuf, double* recvBuf, int* recvCounts, MPI_Comm comm )
{
    int ierror = MPI_Reduce_scatter
    ( sendBuf, recvBuf, recvCounts, MPI_DOUBLE, MPI_SUM, comm );
    if( ierror != 0 )
    {
        std::ostringstream msg;
        msg << "ierror from MPI_Reduce_scatter = " << ierror;
        const std::string& s = msg.str();
        throw s.c_str();
    }
}

template<>
inline void
SumScatter< std::complex<float> >
( std::complex<float>* sendBuf, 
  std::complex<float>* recvBuf, 
  int* recvCounts, MPI_Comm comm )
{
    int ierror = MPI_Reduce_scatter
    ( sendBuf, recvBuf, recvCounts, MPI_COMPLEX, MPI_SUM, comm );
    if( ierror != 0 )
    {
        std::ostringstream msg;
        msg << "ierror from MPI_Reduce_scatter = " << ierror;
        const std::string& s = msg.str();
        throw s.c_str();
    }
}

template<>
void
inline SumScatter< std::complex<double> >
( std::complex<double>* sendBuf, 
  std::complex<double>* recvBuf,
  int* recvCounts, MPI_Comm comm )
{
    int ierror = MPI_Reduce_scatter
    ( sendBuf, recvBuf, recvCounts, MPI_DOUBLE_COMPLEX, MPI_SUM, comm );
    if( ierror != 0 )
    {
        std::ostringstream msg;
        msg << "ierror from MPI_Reduce_scatter = " << ierror;
        const std::string& s = msg.str();
        throw s.c_str();
    }
}

} // bfio

#endif /* BFIO_MPI_HPP */

