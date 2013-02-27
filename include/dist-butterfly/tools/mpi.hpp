/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of DistButterfly and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef DBF_TOOLS_MPI_HPP
#define DBF_TOOLS_MPI_HPP

#include <iostream>
#include <stdexcept>
#include "mpi.h"

namespace dbf {

template<typename T>
void
SumScatter( const T* sendBuf, T* recvBuf, int recvSize, MPI_Comm comm );

} // dbf

// Implementations 
namespace dbf {

template<>
inline void
SumScatter<float>
( const float* sendBuf, float* recvBuf, int recvSize, MPI_Comm comm )
{
#ifdef HAVE_MPI_REDUCE_SCATTER_BLOCK
    std::cout << "MPI_Reduce_scatter_block" << std::endl;
    const int ierror = MPI_Reduce_scatter_block
    ( const_cast<float*>(sendBuf), 
      recvBuf, recvSize, MPI_FLOAT, MPI_SUM, comm );
#else
    std::cout << "MPI_Reduce_scatter" << std::endl;
    int commSize; MPI_Comm_size( comm, &commSize );
    std::vector<int> recvCounts( commSize, recvSize );
    const int ierror = MPI_Reduce_scatter
    ( const_cast<float*>(sendBuf), 
      recvBuf, &recvCounts[0], MPI_FLOAT, MPI_SUM, comm );
#endif
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
( const double* sendBuf, double* recvBuf, int recvSize, MPI_Comm comm )
{
#ifdef HAVE_MPI_REDUCE_SCATTER_BLOCK
    std::cout << "MPI_Reduce_scatter_block" << std::endl;
    const int ierror = MPI_Reduce_scatter_block
    ( const_cast<double*>(sendBuf), 
      recvBuf, recvSize, MPI_DOUBLE, MPI_SUM, comm );
#else
    std::cout << "MPI_Reduce_scatter" << std::endl;
    int commSize; MPI_Comm_size( comm, &commSize );
    std::vector<int> recvCounts( commSize, recvSize );
    const int ierror = MPI_Reduce_scatter
    ( const_cast<double*>(sendBuf), 
      recvBuf, &recvCounts[0], MPI_DOUBLE, MPI_SUM, comm );
#endif
    if( ierror != 0 )
    {
        std::ostringstream msg;
        msg << "ierror from MPI_Reduce_scatter = " << ierror;
        throw std::runtime_error( msg.str() );
    }
}

} // dbf

#endif // ifndef DBF_TOOLS_MPI_HPP
