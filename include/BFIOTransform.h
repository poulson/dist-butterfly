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
#ifndef BFIO_TRANSFORM_H
#define BFIO_TRANSFORM_H 1

#include <bitset>

#include "BFIOUtil.h"
#include "BFIOTemplate.h"
#include "BFIOInitializeWeights.h"

namespace BFIO
{
    // Applies the butterfly algorithm for the Fourier integral operator 
    // defined by Psi. This allows one to call the function
    // with their own functor, Psi, with potentially no performance penalty. 
    // R is the datatype for representing a Real and d is the spatial and 
    // frequency dimension. q is the number of points in each dimension of the 
    // Chebyshev tensor-product grid (q^d points total).
    template<typename Psi,typename R,unsigned d,unsigned q>
    void
    Transform
    ( const unsigned N, 
      const std::vector< Source<R,d> >& mySources,
            std::vector< LRP<Psi,R,d,q> >& myLRPs,
            MPI_Comm comm                         )
    {
        using namespace std;
        typedef complex<R> C;

        int rank, size;
        MPI_Comm_rank( comm, &rank );
        MPI_Comm_size( comm, &size ); 
        bitset<sizeof(int)*8> rankBits(rank);

        // Assert that N and size are powers of 2
        if( ! IsPowerOfTwo(N) )
            throw "Must use a power of 2 problem size.";
        if( ! IsPowerOfTwo(size) ) 
            throw "Must use a power of 2 number of processes.";
        const unsigned log2N = Log2( N );
        const unsigned log2Size = Log2( size );
        if( log2Size > d*log2N )
            throw "Cannot use more than N^d processes.";

        // Determine the number of partitions in each dimension of the 
        // frequency domain by applying the partitions cyclically over the
        // d dimensions. We can simultaneously compute the indices of our 
        // box in each spatial dimension.
        Array<R,d> myFreqBoxWidths;
        Array<unsigned,d> myFreqBoxCoords;
        Array<unsigned,d> log2NumFreqParts;
        for( unsigned j=0; j<d; ++j )
            myFreqBoxCoords[j] = log2NumFreqParts[j] = 0;
        for( unsigned j=log2Size; j>0; --j )
        {
            static unsigned nextPartDim = 0;
            // Double our current coordinate in the 'nextPartDim' dimension 
            // and then choose the left/right position based on the (j-1)'th
            // bit of our rank
            myFreqBoxCoords[nextPartDim] = 
                (myFreqBoxCoords[nextPartDim]<<1)+rankBits[j-1];

            log2NumFreqParts[nextPartDim]++;
            nextPartDim = (nextPartDim+1) % d;
        }
        for( unsigned j=0; j<d; ++j )
            myFreqBoxWidths[j] = 1. / static_cast<R>(1<<log2NumFreqParts[j]);

        // Compute the number of 1/N width boxes in the frequency domain that 
        // our process is responsible for initializing the weights in
        unsigned log2Boxes = 0;
        Array<unsigned,d> log2BoxesPerDim;
        for( unsigned j=0; j<d; ++j )
        {
            log2BoxesPerDim[j] = log2N-log2NumFreqParts[j];
            log2Boxes += log2BoxesPerDim[j];
        }
        const unsigned boxes = 1<<log2Boxes;

        // Compute {zi} for the Chebyshev grid of order q over [-1/2,+1/2]
        Array<R,q> chebyGrid;
        for( unsigned i=0; i<q; ++i )
            chebyGrid[i] = 0.5*cos(i*Pi/(q-1));

        vector< Array<C,Power<q,d>::value> > weights(boxes);
        InitializeWeights<Psi,R,d,q>
        ( N, mySources, chebyGrid, myFreqBoxWidths,
          myFreqBoxCoords, boxes, log2BoxesPerDim, weights );

        // First half of algorithm

        // Switch to spatial interpolation

        // Second half of algorithm

        // Copy weights into LRPs
        {
            myLRPs.resize( 1<<(d*log2N-log2Size) );
            // Fill in the LRPs
        }
    }
}

#endif /* BFIO_TRANSFORM_H */

