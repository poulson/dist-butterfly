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

        // Assert that N and size are powers of 2
        if( ! IsPowerOfTwo(N) )
        {
            if( rank == 0 )
                cerr << "Must use a power of 2 problem size." << endl;
            throw 0;
        }
        if( ! IsPowerOfTwo(size) ) 
	{
            if( rank == 0 )
                cerr << "Must use a power of 2 number of processes." << endl;
            throw 0;
	}
        const unsigned log2N = Log2( N );
        const unsigned log2Size = Log2( size );

        // Determine the number of partitions in each dimension of the 
        // frequency domain by applying the partitions cyclically over the
        // d dimensions. We can simultaneously compute the indices of our 
        // box in each spatial dimension.
        bitset<sizeof(int)*8> rankBits(rank);
        if( rank == 31 )
        {
            for( unsigned j=0; j<8; ++j )
                cout << rankBits[j];
            cout << endl;
        }
        Array<R,d> myFreqBoxWidths;
        Array<unsigned,d> myFreqBoxCoords;
        Array<unsigned,d> log2FreqParts;
        for( unsigned j=0; j<d; ++j )
        {
            myFreqBoxCoords[j] = 0;
            log2FreqParts[j] = 0;
        }
        unsigned nextPartDim = 0;
        for( unsigned j=log2Size; j>0; --j )
        {
            // Double our current coordinate in the 'nextPartDim' dimension 
            // and then choose the left/right position based on the (j-1)'th
            // bit of our rank
            myFreqBoxCoords[nextPartDim] = 
                (myFreqBoxCoords[nextPartDim]<<1)+rankBits[j-1];

            log2FreqParts[nextPartDim]++;
            nextPartDim = (nextPartDim+1) % d;
        }
        for( unsigned j=0; j<d; ++j )
            myFreqBoxWidths[j] = 1. / static_cast<R>(1<<log2FreqParts[j]);

        // Compute the balance of the frequency partitioning among processes
        // with the log2N partitions in each direction. This determines both 
        // the number of interactions we must participate in, as well as the 
        // number of processes we must communicate with in order to form these
        // interactions.
        unsigned log2Boxes = 0;
        unsigned log2Procs = 0;
        Array<int,d> log2BoxesPerDim;
        Array<int,d> log2ProcsPerDim;
        for( unsigned j=0; j<d; ++j )
        {
            if( log2N >= log2FreqParts[j] )
            {
                log2BoxesPerDim[j] = log2N-log2FreqParts[j];
                log2ProcsPerDim[j] = 0;
                log2Boxes += log2BoxesPerDim[j];
            }
            else
            {
                log2BoxesPerDim[j] = 0;
                log2ProcsPerDim[j] = log2FreqParts[j]-log2N;
                log2Procs += log2ProcsPerDim[j];
            }
        }
        const unsigned boxes = 1<<log2Boxes;
        const unsigned procs = 1<<log2Procs;

        // Initialize the location of the spatial root node's center
        Array<R,d> x0;
        for( unsigned j=0; j<d; ++j )
            x0[j] = 0.5;

        // Store the width of B
        R widthOfB = 1. / static_cast<R>(N);

        // Compute {zi} for the Chebyshev grid of order q over [-1/2,+1/2]
        Array<R,q> chebyGrid;
        for( unsigned i=0; i<q; ++i )
            chebyGrid[i] = 0.5*cos(i*Pi/(q-1));

        // Compute the unscaled weights for each local box by looping over 
        // our sources and sorting them into the appropriate local box one 
        // at a time. Bombs if a source is outside of our frequency box.
        vector< Array<C,Power<q,d>::value> > weights(boxes);
        for( unsigned i=0; i<mySources.size(); ++i )
        {
            const Array<R,d>& p = mySources[i].p;

            // Determine which local box we're in (if any)
            Array<unsigned,d> localBoxCoords;
            for( unsigned j=0; j<d; ++j )
            {
                const R pj = p[j];
                R leftBound = myFreqBoxWidths[j]*myFreqBoxCoords[j];
                R rightBound = myFreqBoxWidths[j]*(myFreqBoxCoords[j]+1);
                if( pj < leftBound || pj >= rightBound )
                {
                    cerr << "Source " << i << " was at " << pj
                         << " in dimension " << j << ", but our frequency box"
                         << " in this dim. is [" << leftBound << "," 
                         << rightBound << ")." << endl;
                    throw 0;
                }

                // We must be in the box, so bitwise determine the coord index
                // by bisection of box B_loc
                localBoxCoords[j] = 0;
                for( unsigned k=log2BoxesPerDim[j]; k>0; --k )
                {
                    const R middle = (rightBound-leftBound)/2.;
                    if( pj < middle )
                    {
                        // implicitly setting bit k-1 of localBoxCoords[j] to 0
                        rightBound = middle;
                    }
                    else
                    {
                        localBoxCoords[j] |= (1<<(k-1));
                        leftBound = middle;
                    }
                }
            }

            // Translate the local integer coordinates into the freq. center
            // of box B (not of B_loc!)
            Array<R,d> p0;
            for( unsigned j=0; j<d; ++j )
            {
                p0[j] = myFreqBoxWidths[j]*myFreqBoxCoords[j] + 
                        (localBoxCoords[j]+0.5)*widthOfB;
            }

            // Flatten the integer coordinates of B_loc into a single index
            unsigned k = 0;
            for( unsigned j=0; j<d; ++j )
            {
                static unsigned log2BoxesUpToDim = 0;
                k |= (localBoxCoords[j]<<log2BoxesUpToDim);
                log2BoxesUpToDim += log2BoxesPerDim[j];
            }

            // Add this point's contribution to the unscaled weights of B. 
            // We evaluate the Lagrangian polynomial on the reference grid, 
            // so we need to map p to it first.
            Array<R,d> pRef;
            for( unsigned j=0; j<d; ++j )
                pRef[j] = (p[j]-p0[j])/widthOfB;
            const C f = mySources[i].magnitude;
            const C alpha = exp( C(0.,TwoPi*N)*Psi::Eval(x0,p) ) * f;
            WeightSummation<R,d,q>::Eval(alpha,pRef,chebyGrid,weights[k]);
        }

        // Loop over all of the boxes to compute the {p_t^B} and prefactors
        // for each delta weight {delta_t^AB}, exp(-2 Pi i N Psi(x0,p_t^B) ).
        // Notice that if we are sharing a box with one or more processes, then
        // our local boxes, say B_loc, are subsets of a B, but we still need 
        // to consider our contribution to the weights corresponding to 
        // p_t^B lying in B \ B_loc.
        for( unsigned k=0; k<boxes; ++k ) 
        {
            // Compute the local integer coordinates of box k
            Array<unsigned,d> localBoxCoords;
            for( unsigned j=0; j<d; ++j )
            {
                static unsigned log2BoxesUpToDim = 0;
                unsigned log2BoxesUpToNextDim = 
                    log2BoxesUpToDim+log2BoxesPerDim[j];
                localBoxCoords[j] = (k>>log2BoxesUpToDim) & 
                                    ((1<<log2BoxesUpToNextDim)-1);
                log2BoxesUpToDim = log2BoxesUpToNextDim;
            }

            // Translate the local integer coordinates into the freq. center 
            Array<R,d> p0;
            for( unsigned j=0; j<d; ++j )
            {
                p0[j] = myFreqBoxWidths[j]*myFreqBoxCoords[j] + 
                        (localBoxCoords[j]+0.5)*widthOfB;
            }

            // Compute the prefactors given this p0 and multiply it by 
            // the corresponding weights
            ScaleWeights<Psi,R,d,q>::Eval
            (N,widthOfB,x0,p0,chebyGrid,weights[k]);
        }

        // If we only contributed to the weight calculations, sum the results
        // over this team
        if( procs != 1 )
        {
            // Create the list of participating processes
            const int teamRank = rank & (procs-1);
            const int startRank = rank-teamRank;
            vector<int> ranks(procs);
            for( unsigned j=0; j<procs; ++j )
                ranks[j] = j+startRank;

            // Construct the communicator
            MPI_Group group;
            MPI_Comm_group( comm, &group );
            MPI_Group teamGroup;
            MPI_Group_incl( group, procs, &ranks[0], &teamGroup );
            MPI_Comm teamComm;
            MPI_Comm_create( comm, teamGroup, &teamComm );

            // Sum weights over new communicator
            vector< Array<C,Power<q,d>::value> > weightsCopy = weights;
            Sum
            ( &weightsCopy[0][0], &weights[0][0], 
              boxes*Power<q,d>::value, teamComm  );

            // Destroys communicator
            MPI_Comm_free( &teamComm );
            MPI_Group_free( &teamGroup );
            MPI_Group_free( &group );
        }

        // Print out the number of partitions in each direction as well as 
        // the box coordinates of each process
        if( rank == 0 )
        {
            cout << "Number of partitions in each dimension:" << endl;
            for( unsigned j=0; j<d; ++j )
                cout << j << ": " << log2FreqParts[j] << endl;
        }
        for( int j=0; j<size; ++j )
        {
            if( rank == j )
            {
                cout << "Location of process " << rank << ": ";
                for( unsigned i=0; i<d; ++i )
                    cout << myFreqBoxCoords[i] << " ";  
                cout << endl;
            }
            sleep( 1 );
        }
    }
}

#endif /* BFIO_TRANSFORM_H */

