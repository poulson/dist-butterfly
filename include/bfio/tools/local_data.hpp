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
#ifndef BFIO_LOCAL_DATA_HPP
#define BFIO_LOCAL_DATA_HPP 1

#include <bitset>
#include "mpi.h"
#include "bfio/structures/data.hpp"
#include "bfio/tools/twiddle.hpp"

namespace bfio {

template<unsigned d>
inline unsigned
NumLocalBoxes
( unsigned N, MPI_Comm comm )
{
    int size;
    MPI_Comm_size( comm, &size );
    unsigned L = Log2( N );
    unsigned s = Log2( size );
    return 1<<(d*L-s);
}

template<typename R,unsigned d>
void    
LocalFreqPartitionData
( Array<R,d>& myFreqBoxWidths,
  Array<R,d>& myFreqBoxOffsets,
  MPI_Comm comm )
{
    int rank, size;
    MPI_Comm_rank( comm, &rank );
    MPI_Comm_size( comm, &size );
    std::bitset<8*sizeof(int)> rankBits(rank);
    const unsigned s = Log2( size );

    Array<unsigned,d> myFreqBox(0);
    Array<unsigned,d> log2FreqBoxesPerDim(0);
    for( unsigned j=0; j<d; ++j )
        myFreqBox[j] = 0;
    unsigned nextDim = 0;
    for( unsigned m=s; m>0; --m )
    {
        myFreqBox[nextDim] = (myFreqBox[nextDim]<<1)+rankBits[m-1];
        ++log2FreqBoxesPerDim[nextDim];
        nextDim = (nextDim+1) % d;
    }
    for( unsigned j=0; j<d; ++j )
    {
        myFreqBoxWidths[j] = static_cast<R>(1)/(1<<log2FreqBoxesPerDim[j]);
        myFreqBoxOffsets[j] = myFreqBox[j]*myFreqBoxWidths[j];
    }
}

template<typename R,unsigned d>
void
LocalFreqPartitionData
( Array<unsigned,d>& myFreqBox,
  Array<unsigned,d>& log2FreqBoxesPerDim,
  Array<R,d>& myFreqBoxWidths,
  Array<R,d>& myFreqBoxOffsets,
  MPI_Comm comm )
{
    int rank, size;
    MPI_Comm_rank( comm, &rank );
    MPI_Comm_size( comm, &size );
    std::bitset<8*sizeof(int)> rankBits(rank);
    const unsigned s = Log2( size );

    for( unsigned j=0; j<d; ++j )
        myFreqBox[j] = log2FreqBoxesPerDim[j] = 0;
    unsigned nextDim = 0;
    for( unsigned m=s; m>0; --m )
    {
        myFreqBox[nextDim] = (myFreqBox[nextDim]<<1)+rankBits[m-1];
        ++log2FreqBoxesPerDim[nextDim];
        nextDim = (nextDim+1) % d;
    }
    for( unsigned j=0; j<d; ++j )
    {
        myFreqBoxWidths[j] = static_cast<R>(1)/(1<<log2FreqBoxesPerDim[j]);
        myFreqBoxOffsets[j] = myFreqBox[j]*myFreqBoxWidths[j];
    }
}

} // bfio

#endif /* BFIO_LOCAL_DATA_HPP */

