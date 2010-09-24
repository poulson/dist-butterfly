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
#ifndef BFIO_TOOLS_LOCAL_DATA_HPP
#define BFIO_TOOLS_LOCAL_DATA_HPP 1

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
    int numProcesses;
    MPI_Comm_size( comm, &numProcesses );
    unsigned log2N = Log2( N );
    unsigned log2NumProcesses = Log2( numProcesses );
    return 1<<(d*log2N-log2NumProcesses);
}

template<typename R,unsigned d>
void    
LocalFreqPartitionData
( const Box<R,d>& freqBox, Box<R,d>& myFreqBox, MPI_Comm comm )
{
    int rank, numProcesses;
    MPI_Comm_rank( comm, &rank );
    MPI_Comm_size( comm, &numProcesses );
    std::bitset<8*sizeof(int)> rankBits(rank);
    const unsigned log2NumProcesses = Log2( numProcesses );

    Array<unsigned,d> myFreqBoxCoords(0);
    Array<unsigned,d> log2FreqBoxesPerDim(0);
    unsigned nextDim = 0;
    for( unsigned m=log2NumProcesses; m>0; --m )
    {
        myFreqBoxCoords[nextDim] = (myFreqBoxCoords[nextDim]<<1)+rankBits[m-1];
        ++log2FreqBoxesPerDim[nextDim];
        nextDim = (nextDim+1) % d;
    }
    for( unsigned j=0; j<d; ++j )
    {
        myFreqBox.widths[j] = freqBox.widths[j] / (1<<log2FreqBoxesPerDim[j]);
        myFreqBox.offsets[j] = freqBox.offsets[j] + 
                               myFreqBoxCoords[j]*myFreqBox.widths[j];
    }
}

template<typename R,unsigned d>
void
LocalFreqPartitionData
( const Box<R,d>& freqBox,
        Box<R,d>& myFreqBox,
        Array<unsigned,d>& myFreqBoxCoords,
        Array<unsigned,d>& log2FreqBoxesPerDim,
        MPI_Comm comm )
{
    int rank, numProcesses;
    MPI_Comm_rank( comm, &rank );
    MPI_Comm_size( comm, &numProcesses );
    std::bitset<8*sizeof(int)> rankBits(rank);
    const unsigned log2NumProcesses = Log2( numProcesses );

    for( unsigned j=0; j<d; ++j )
        myFreqBoxCoords[j] = log2FreqBoxesPerDim[j] = 0;
    unsigned nextDim = 0;
    for( unsigned m=log2NumProcesses; m>0; --m )
    {
        myFreqBoxCoords[nextDim] = (myFreqBoxCoords[nextDim]<<1)+rankBits[m-1];
        ++log2FreqBoxesPerDim[nextDim];
        nextDim = (nextDim+1) % d;
    }
    for( unsigned j=0; j<d; ++j )
    {
        myFreqBox.widths[j] = freqBox.widths[j] / (1<<log2FreqBoxesPerDim[j]);
        myFreqBox.offsets[j] = freqBox.offsets[j] + 
                               myFreqBoxCoords[j]*myFreqBox.widths[j];
    }
}

} // bfio

#endif // BFIO_TOOLS_LOCAL_DATA_HPP

