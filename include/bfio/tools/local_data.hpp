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

