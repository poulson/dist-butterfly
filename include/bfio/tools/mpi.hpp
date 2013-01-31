/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of ButterflyFIO and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef BFIO_TOOLS_MPI_HPP
#define BFIO_TOOLS_MPI_HPP

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
#ifdef RELEASE
    MPI_Reduce_scatter
    ( const_cast<float*>(sendBuf), 
      recvBuf, recvCounts, MPI_FLOAT, MPI_SUM, comm );
#else
    int ierror = MPI_Reduce_scatter
    ( const_cast<float*>(sendBuf), 
      recvBuf, recvCounts, MPI_FLOAT, MPI_SUM, comm );
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
#ifdef RELEASE
    MPI_Reduce_scatter
    ( const_cast<double*>(sendBuf), 
      recvBuf, recvCounts, MPI_DOUBLE, MPI_SUM, comm );
#else
    int ierror = MPI_Reduce_scatter
    ( const_cast<double*>(sendBuf), 
      recvBuf, recvCounts, MPI_DOUBLE, MPI_SUM, comm );
    if( ierror != 0 )
    {
        std::ostringstream msg;
        msg << "ierror from MPI_Reduce_scatter = " << ierror;
        throw std::runtime_error( msg.str() );
    }
#endif
}

} // bfio

#endif // ifndef BFIO_TOOLS_MPI_HPP
