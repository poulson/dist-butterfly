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
#ifndef BFIO_TOOLS_MPI_HPP
#define BFIO_TOOLS_MPI_HPP 1

#include <stdexcept>
#include "mpi.h"

namespace bfio {

template<typename T>
void
SumScatter
( const T* sendBuf, T* recvBuf, int* recvCounts, MPI_Comm comm );

} // bfio

// Implementations 
namespace bfio {

template<>
inline void
SumScatter<float>
( const float* sendBuf, float* recvBuf, int* recvCounts, MPI_Comm comm )
{
    int ierror = MPI_Reduce_scatter
    ( const_cast<float*>(sendBuf), 
      recvBuf, recvCounts, MPI_FLOAT, MPI_SUM, comm );
#ifndef RELEASE
    if( ierror != 0 )
    {
        std::ostringstream msg;
        msg << "ierror from MPI_Reduce_scatter = " << ierror;
        throw std::runtime_error( msg.str() );
    }
#endif
}

template<>
inline void
SumScatter<double>
( const double* sendBuf, double* recvBuf, int* recvCounts, MPI_Comm comm )
{
    int ierror = MPI_Reduce_scatter
    ( const_cast<double*>(sendBuf), 
      recvBuf, recvCounts, MPI_DOUBLE, MPI_SUM, comm );
#ifndef RELEASE
    if( ierror != 0 )
    {
        std::ostringstream msg;
        msg << "ierror from MPI_Reduce_scatter = " << ierror;
        throw std::runtime_error( msg.str() );
    }
#endif
}

} // bfio

#endif // BFIO_TOOLS_MPI_HPP

