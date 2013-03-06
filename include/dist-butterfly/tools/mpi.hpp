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
SumScatter( T* sendBuf, T* recvBuf, int recvSize, MPI_Comm comm );

} // dbf

// Implementations 
namespace dbf {

template<>
inline void
SumScatter<float>
( float* sendBuf, float* recvBuf, int recvSize, MPI_Comm comm )
{
#ifdef REDUCE_SCATTER_VIA_ALLREDUCE
    int commSize; MPI_Comm_size( comm, &commSize );
    int commRank; MPI_Comm_rank( comm, &commRank );
    const int ierror = MPI_Allreduce
    ( MPI_IN_PLACE, sendBuf, recvSize*commSize, MPI_FLOAT, MPI_SUM, comm );
    std::memcpy( recvBuf, &sendBuf[commRank*recvSize], recvSize*sizeof(float) );
#else
# ifdef HAVE_MPI_REDUCE_SCATTER_BLOCK
    const int ierror = MPI_Reduce_scatter_block
    ( sendBuf, recvBuf, recvSize, MPI_FLOAT, MPI_SUM, comm );
# else
    int commSize; MPI_Comm_size( comm, &commSize );
    std::vector<int> recvCounts( commSize, recvSize );
    const int ierror = MPI_Reduce_scatter
    ( sendBuf, recvBuf, &recvCounts[0], MPI_FLOAT, MPI_SUM, comm );
# endif
#endif
    if( ierror != 0 )
    {
        std::ostringstream msg;
        msg << "ierror in SumScatter = " << ierror;
        throw std::runtime_error( msg.str() );
    }
}

template<>
inline void
SumScatter<double>
( double* sendBuf, double* recvBuf, int recvSize, MPI_Comm comm )
{
#ifdef REDUCE_SCATTER_VIA_ALLREDUCE
    int commSize; MPI_Comm_size( comm, &commSize );
    int commRank; MPI_Comm_rank( comm, &commRank );
    const int ierror = MPI_Allreduce
    ( MPI_IN_PLACE, sendBuf, recvSize*commSize, MPI_DOUBLE, MPI_SUM, comm );
    std::memcpy
    ( recvBuf, &sendBuf[commRank*recvSize], recvSize*sizeof(double) );
#else
# ifdef HAVE_MPI_REDUCE_SCATTER_BLOCK
    const int ierror = MPI_Reduce_scatter_block
    ( sendBuf, recvBuf, recvSize, MPI_DOUBLE, MPI_SUM, comm );
# else
    int commSize; MPI_Comm_size( comm, &commSize );
    std::vector<int> recvCounts( commSize, recvSize );
    const int ierror = MPI_Reduce_scatter
    ( sendBuf, recvBuf, &recvCounts[0], MPI_DOUBLE, MPI_SUM, comm );
# endif
#endif
    if( ierror != 0 )
    {
        std::ostringstream msg;
        msg << "ierror in SumScatter = " << ierror;
        throw std::runtime_error( msg.str() );
    }
}

} // dbf

#endif // ifndef DBF_TOOLS_MPI_HPP
