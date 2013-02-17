/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of DistButterfly and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef DBF_TOOLS_MPI_HPP
#define DBF_TOOLS_MPI_HPP

#include <stdexcept>
#include "mpi.h"

namespace dbf {

template<typename T>
void
SumScatter
( const T* sendBuf, T* recvBuf, int* recvCounts, MPI_Comm comm );

} // dbf

// Implementations 
namespace dbf {

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

} // dbf

#endif // ifndef DBF_TOOLS_MPI_HPP