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
#ifndef BFIO_TRANSFORM_HPP
#define BFIO_TRANSFORM_HPP 1

#include <bitset>

#include "BFIO/Util.hpp"
#include "BFIO/LRP.hpp"
#include "BFIO/InitializeWeights.hpp"
#include "BFIO/FreqWeightRecursion.hpp"
#include "BFIO/SwitchToSpatialInterp.hpp"
#include "BFIO/SpatialWeightRecursion.hpp"

namespace BFIO
{
    using namespace std;

    // Applies the butterfly algorithm for the Fourier integral operator 
    // defined by the phase function Phi. This allows one to call the function
    // with their own functor, Phi, with potentially no performance penalty. 
    // R is the datatype for representing a Real and d is the spatial and 
    // frequency dimension. q is the number of points in each dimension of the 
    // Chebyshev tensor-product grid (q^d points total).
    template<typename Phi,typename R,unsigned d,unsigned q>
    void
    Transform
    ( const unsigned N, 
      const vector< Source<R,d> >& mySources,
            vector< LRP<Phi,R,d,q> >& myLRPs,
            MPI_Comm comm                    )
    {
        typedef complex<R> C;

        int rank, S;
        MPI_Comm_rank( comm, &rank );
        MPI_Comm_size( comm, &S    ); 
        bitset<sizeof(int)*8> rankBits(rank); 

        double startTime = MPI_Wtime();

        // Assert that N and size are powers of 2
        if( ! IsPowerOfTwo(N) )
            throw "Must use a power of 2 problem size.";
        if( ! IsPowerOfTwo(S) ) 
            throw "Must use a power of 2 number of processes.";
        const unsigned L = Log2( N );
        const unsigned s = Log2( S );
        if( s > d*L )
            throw "Cannot use more than N^d processes.";

        if( rank == 0 )
        {
            cout << MPI_Wtime()-startTime << " seconds." << endl;
            cout << "L = " << L << endl;
            cout << "s = " << s << endl;
        }

        // Determine the number of boxes in each dimension of the frequency
        // domain by applying the partitions cyclically over the d dimensions.
        // We can simultaneously compute the indices of our box.
        Array<unsigned,d> myFreqBox;
        Array<unsigned,d> mySpatialBox;
        Array<unsigned,d> log2FreqBoxesPerDim;
        Array<unsigned,d> log2SpatialBoxesPerDim;
        for( unsigned j=0; j<d; ++j )
        {
            myFreqBox[j] = mySpatialBox[j] = 0;
            log2FreqBoxesPerDim[j] = log2SpatialBoxesPerDim[j] = 0;
        }
        unsigned nextDim = 0;
        for( unsigned j=s; j>0; --j )
        {
            // Double our current coordinate in the 'nextDim' dimension 
            // and then choose the left/right position based on the (j-1)'th
            // bit of our rank
            myFreqBox[nextDim] = (myFreqBox[nextDim]<<1)+rankBits[j-1];

            log2FreqBoxesPerDim[nextDim]++;
            nextDim = (nextDim+1) % d;
        }

        // Initialize the widths of the boxes in the spatial and frequency 
        // domains that our process is responsible for
        Array<R,d> myFreqBoxWidths;
        Array<R,d> mySpatialBoxWidths;
        for( unsigned j=0; j<d; ++j )
        {
            myFreqBoxWidths[j] = static_cast<R>(1)/(1<<log2FreqBoxesPerDim[j]);
            mySpatialBoxWidths[j] = static_cast<R>(1);
        }
        
        // Compute the offsets for our frequency and spatial box
        Array<R,d> myFreqBoxOffsets;
        Array<R,d> mySpatialBoxOffsets;
        for( unsigned j=0; j<d; ++j )
        {
            myFreqBoxOffsets[j] = myFreqBox[j]*myFreqBoxWidths[j];
            mySpatialBoxOffsets[j] = mySpatialBox[j]*mySpatialBoxWidths[j];
        }

        // Compute the number of 1/N width boxes in the frequency domain that 
        // our process is responsible for initializing the weights in. Also
        // initialize each box being responsible for all of the spatial domain.
        unsigned log2LocalFreqBoxes = 0;
        unsigned log2LocalSpatialBoxes = 0;
        Array<unsigned,d> log2LocalFreqBoxesPerDim;
        Array<unsigned,d> log2LocalSpatialBoxesPerDim;
        for( unsigned j=0; j<d; ++j )
        {
            log2LocalFreqBoxesPerDim[j] = L-log2FreqBoxesPerDim[j];
            log2LocalSpatialBoxesPerDim[j] = 0;
            log2LocalFreqBoxes += log2LocalFreqBoxesPerDim[j];
        }

        // Compute the Chebyshev grid over [-1/2,+1/2]^d
        if( rank == 0 )
        {
            cout << MPI_Wtime()-startTime << " seconds." << endl;
            cout << "Initializing Chebyshev grid...";
        }
        vector< Array<R,d> > chebyGrid( Pow<q,d>::val );
        for( unsigned t=0; t<Pow<q,d>::val; ++t )
        {
            unsigned qToThej = 1;
            for( unsigned j=0; j<d; ++j )
            {
                unsigned i = (t/qToThej)%q;
                chebyGrid[t][j] = 0.5*cos(i*Pi/(q-1));
                qToThej *= q;
            }
        }
        if( rank == 0 )
        {
            cout << "done." << endl;
            cout << MPI_Wtime()-startTime << " seconds." << endl;
        }

        // Initialize the weights using Lagrangian interpolation on the 
        // smooth component of the kernel.
        if( rank == 0 )
        {
            cout << "Initializing weights...";
            cout.flush();
        }
        WeightSetList<R,d,q> weightSetList( 1<<log2LocalFreqBoxes );
        InitializeWeights<Phi,R,d,q>
        ( N, mySources, chebyGrid, myFreqBoxWidths, myFreqBox,
          log2LocalFreqBoxes, log2LocalFreqBoxesPerDim, weightSetList );
        if( rank == 0 )
        {
            cout << "done." << endl;
            cout << MPI_Wtime()-startTime << " seconds." << endl;
        }

        // Start the main recursion loop
        if( rank == 0 )
            cout << "Starting algorithm." << endl;
        for( unsigned l=1; l<=L; ++l )
        {
            // Compute the width of the nodes at level l
            const R wA = static_cast<R>(1) / (1<<l);
            const R wB = static_cast<R>(1) / (1<<(L-l));
            if( rank == 0 )
                cout << "(wA,wB)=(" << wA << "," << wB << ")" << endl;

            // Print the state at the beginning of the loop
            MPI_Barrier( comm );
            if( rank == 0 )
            {
                cout << "At the beginning of level " << l << endl;
                cout << "=========================================" << endl;
            }
            for( int m=0; m<S; ++m )
            {
                if( rank == m )
                {
                    cout << "  Rank: " << m << endl;
                    cout << "  mySpatialBox:        ";
                    for( unsigned j=0; j<d; ++j )
                        cout << mySpatialBox[j] << " ";
                    cout << endl;
                    cout << "  mySpatialBoxOffsets: ";
                    for( unsigned j=0; j<d; ++j )
                        cout << mySpatialBoxOffsets[j] << " ";
                    cout << endl;
                    cout << "  mySpatialBoxWidths:  ";
                    for( unsigned j=0; j<d; ++j )
                        cout << mySpatialBoxWidths[j] << " ";
                    cout << endl;
                    cout << "  log2LocalSpatialBoxesPerDim: ";
                    for( unsigned j=0; j<d; ++j )
                        cout << log2LocalSpatialBoxesPerDim[j] << " ";
                    cout << endl << endl;
                }
                MPI_Barrier( comm );
            }
            for( int m=0; m<S; ++m )
            {
                if( rank == m )
                {
                    cout << "  Rank: " << m << endl;
                    cout << "  myFreqBox:        ";
                    for( unsigned j=0; j<d; ++j )
                        cout << myFreqBox[j] << " ";
                    cout << endl;
                    cout << "  myFreqBoxOffsets: ";
                    for( unsigned j=0; j<d; ++j )
                        cout << myFreqBoxOffsets[j] << " ";
                    cout << endl;
                    cout << "  myFreqBoxWidths:  ";
                    for( unsigned j=0; j<d; ++j )
                        cout << myFreqBoxWidths[j] << " ";
                    cout << endl;
                    cout << "  log2LocalFreqBoxesPerDim: ";
                    for( unsigned j=0; j<d; ++j )
                        cout << log2LocalFreqBoxesPerDim[j] << " ";
                    cout << endl << endl;
                }
                MPI_Barrier( comm );
            }
            for( int m=0; m<S; ++m )
            {
                if( rank == m )
                {
                    cout << "  Rank: " << m << endl; 
                    for( unsigned i=0; i<(1u<<log2LocalSpatialBoxes); ++i )
                    {
                        cout << "  Space box: " << i << endl;
                        for( unsigned j=0; j<(1u<<log2LocalFreqBoxes); ++j )
                        {
                            cout << "    Freq box: " << j << endl;
                            for( unsigned t=0; t<Pow<q,d>::val; ++t )
                            {
                                unsigned index = j+(i<<log2LocalFreqBoxes);
                                cout << "      Weight " << t << ": " 
                                     << weightSetList[index][t] << endl;
                            }
                        }
                    }
                }
                MPI_Barrier( comm );
            }

            if( log2LocalFreqBoxes >= d )
            {
                // Refine the spatial domain and coursen the frequency domain
                for( unsigned j=0; j<d; ++j )
                {
                    --log2LocalFreqBoxesPerDim[j];
                    ++log2LocalSpatialBoxesPerDim[j];
                }
                log2LocalFreqBoxes -= d;
                log2LocalSpatialBoxes += d;

                // Form the N^d/S = 2^(d*L) / 2^s = 2^(d*L-s) weights

                // Loop over A boxes in spatial domain. 'i' will represent the 
                // leaf number w.r.t. the tree implied by cyclically assigning
                // the spatial bisections across the d dimensions. Thus if we 
                // distribute the data cyclically in the _reverse_ order over 
                // the d dimensions, then the ReduceScatter will not require 
                // any packing or unpacking.
                WeightSetList<R,d,q> oldWeightSetList( weightSetList );
                Array<unsigned,d> A( 0 );
                for( unsigned i=0; i<(1u<<log2LocalSpatialBoxes); ++i )
                {
                    // Compute the coordinates and center of this spatial box
                    Array<R,d> x0A;

                    for( unsigned j=0; j<d; ++j )
                        x0A[j] = mySpatialBoxOffsets[j] + A[j]*wA + wA/2;

                    // Loop over the B boxes in frequency domain
                    Array<unsigned,d> B( 0 );
                    for( unsigned k=0; k<(1u<<log2LocalFreqBoxes); ++k )
                    {
                        // Compute the coordinates and center of this freq box
                        Array<R,d> p0B;
                        for( unsigned j=0; j<d; ++j )
                            p0B[j] = myFreqBoxOffsets[j] + B[j]*wB + wB/2;

                        const unsigned key = k + (i<<log2LocalFreqBoxes);
                        const unsigned parentOffset = 
                            ((i>>d)<<(log2LocalFreqBoxes+d)) + (k<<d);
                        if( l <= L/2 )
                        {
                            FreqWeightRecursion<Phi,R,d,q>
                            ( 0, 0, N, chebyGrid, 
                              x0A, p0B, wB, parentOffset,
                              oldWeightSetList, weightSetList[key] );
                        }
                        else
                        {
                            unsigned ARelativeToAp = 0;
                            Array<R,d> x0Ap;
                            Array<unsigned,d> globalA;
                            for( unsigned j=0; j<d; ++j )
                            {
                                globalA[j] = 
                                    (mySpatialBox[j]<<
                                     log2LocalSpatialBoxesPerDim[j])+A[j];
                                x0Ap[j] = (globalA[j]/2)*2*wA + wA;
                                ARelativeToAp |= (globalA[j]&1)<<j;
                            }
                            SpatialWeightRecursion<Phi,R,d,q>
                            ( 0, 0, N, chebyGrid, 
                              ARelativeToAp, x0A, x0Ap, p0B, wA, wB,
                              parentOffset, oldWeightSetList, 
                              weightSetList[key] );
                        }
                        TraverseHTree( log2LocalFreqBoxesPerDim, B );
                    }
                    TraverseHTree( log2LocalSpatialBoxesPerDim, A );
                }
            }
            else 
            {
                // There are currently 2^(d*(L-l)) leaves. The frequency 
                // partitioning is implied by reading the rank bits left-to-
                // right starting with bit s-1, but the spatial partitioning 
                // is implied by reading the rank bits right-to-left. 
                //
                // We notice that our consistency in the cyclic bisection of 
                // the frequency domain means that if log2Procs=a, then 
                // we communicate with 1 other process in each of the first 
                // a of d dimensions. Getting these ranks is implicit in the
                // tree structure.
                const unsigned log2Procs = d-log2LocalFreqBoxes;
                log2LocalFreqBoxes = 0; 
                for( unsigned j=0; j<d; ++j )
                    log2LocalFreqBoxesPerDim[j] = 0;

                // Pull the group out of the global communicator
                MPI_Group group;
                MPI_Comm_group( comm, &group );

                // Construct the group for our local team
                static unsigned numSpaceCuts = 0;
                MPI_Group teamGroup;
                unsigned myTeamRank = 0; // initialize to avoid warnings
                const int startRank = rank-
                    (((rank>>numSpaceCuts)&((1<<log2Procs)-1))<<numSpaceCuts);
                vector<int> ranks( 1<<log2Procs );
                for( unsigned j=0; j<(1u<<log2Procs); ++j )
                {
                    // We need to reverse the order of the last log2Procs
                    // bits of j and add the result onto the startRank
                    unsigned jReversed = 0;
                    for( unsigned k=0; k<log2Procs; ++k )
                        jReversed |= ((j>>k)&1)<<(log2Procs-1-k);
                    ranks[j] = startRank+jReversed;
                    if( ranks[j] == rank )
                        myTeamRank = j;
                }
                MPI_Group_incl( group, 1<<log2Procs, &ranks[0], &teamGroup );

                MPI_Barrier( comm );
                if( rank == 0 )
                {
                    cout << "Team ranks:" << endl;
                    cout << "==========================" << endl;
                }
                for( int m=0; m<S; ++m )
                {
                    if( rank == m )
                    {
                        cout << "Rank " << m << ": ";
                        for( unsigned a=0; a<(1u<<log2Procs); ++a )
                            cout << ranks[a] << " ";
                        cout << endl;
                    }
                    MPI_Barrier( comm );
                }
                
                // Construct the local team communicator from the team group
                MPI_Comm  teamComm;
                MPI_Comm_create( comm, teamGroup, &teamComm );

                // Fully refine the spatial domain and coarsen frequency domain.
                // We will partition the spatial domain after the SumScatter.
                for( unsigned j=0; j<d; ++j )
                {
                    ++log2LocalSpatialBoxesPerDim[j];
                    ++log2LocalSpatialBoxes;
                    
                    if( myFreqBox[j] & 1 )
                    {
                        myFreqBoxOffsets[j] *= 
                            static_cast<R>(myFreqBox[j]-1)/myFreqBox[j];
                    }
                    myFreqBox[j] >>= 1;
                    myFreqBoxWidths[j] *= 2;
                }
                
                // Compute the coordinates and center of this freq box
                Array<R,d> p0B;
                for( unsigned j=0; j<d; ++j )
                    p0B[j] = myFreqBoxOffsets[j] + wB/2;
                
                // Form the partial weights. 
                //
                // Loop over A boxes in spatial domain. 'i' will represent the 
                // leaf number w.r.t. the tree implied by cyclically assigning
                // the spatial bisections across the d dimensions. Thus if we 
                // distribute the data cyclically in the _reverse_ order over 
                // the d dimensions, then the ReduceScatter will not require 
                // any packing or unpacking.
                WeightSetList<R,d,q> partialWeightSetList
                ( 1<<log2LocalSpatialBoxes );
                Array<unsigned,d> A( 0 );
                for( unsigned i=0; i<(1u<<log2LocalSpatialBoxes); ++i )
                {
                    // Compute the coordinates and center of this spatial box
                    Array<R,d> x0A;
                    for( unsigned j=0; j<d; ++j )
                        x0A[j] = mySpatialBoxOffsets[j] + A[j]*wA + wA/2;

                    const unsigned parentOffset = ((i>>d)<<(d-log2Procs));
                    MPI_Barrier( comm );
                    if( rank == 0 )
                    {
                        cout << "Entering partial recursion...";    
                        cout.flush();
                    }
                    if( l <= L/2 )
                    {
                        FreqWeightRecursion<Phi,R,d,q>
                        ( log2Procs, myTeamRank, 
                          N, chebyGrid, x0A, p0B, wB, parentOffset,
                          weightSetList, partialWeightSetList[i]   );
                    }
                    else
                    {
                        unsigned ARelativeToAp = 0;
                        Array<R,d> x0Ap;
                        Array<unsigned,d> globalA;
                        for( unsigned j=0; j<d; ++j )
                        {
                            globalA[j] = 
                                (mySpatialBox[j]<<
                                 log2LocalSpatialBoxesPerDim[j])+A[j];
                            x0Ap[j] = (globalA[j]/2)*2*wA + wA;
                            ARelativeToAp |= (globalA[j]&1)<<j;
                        }
                        SpatialWeightRecursion<Phi,R,d,q>
                        ( log2Procs, myTeamRank,
                          N, chebyGrid, ARelativeToAp,
                          x0A, x0Ap, p0B, wA, wB, parentOffset,
                          weightSetList, partialWeightSetList[i] );
                    }
                    MPI_Barrier( comm );
                    if( rank == 0 )
                        cout << "done." << endl;
                    TraverseHTree( log2LocalSpatialBoxesPerDim, A );
                }

                MPI_Barrier( comm );
                if( rank == 0 )
                {
                    cout << "About to communicate...";    
                    cout.flush();
                }
                // Scatter the summation of the weights
                vector<int> recvCounts( 1<<log2Procs );
                for( unsigned j=0; j<(1u<<log2Procs); ++j )
                    recvCounts[j] = weightSetList.Length()*Pow<q,d>::val;
                SumScatter
                ( &(partialWeightSetList[0][0]), &(weightSetList[0][0]), 
                  &recvCounts[0], teamComm                              );
                MPI_Barrier( comm );
                if( rank == 0 )
                    cout << "done." << endl;
 
                // There is at most 1 case where multiple processes communicate
                // with a team size not equal to 2^d, so we can wrap backwards
                // over the d dimensions by always starting this loop from d
                for( unsigned j=0; j<log2Procs; ++j )
                {
                    const unsigned dim = d-j-1;
                    mySpatialBoxWidths[dim] /= 2;
                    mySpatialBox[dim] <<= 1;
                    if( rankBits[numSpaceCuts++] ) 
                    {
                        mySpatialBox[dim] |= 1;
                        mySpatialBoxOffsets[dim] += mySpatialBoxWidths[dim];    
                    }

                    --log2LocalSpatialBoxesPerDim[dim];
                    --log2LocalSpatialBoxes;
                }

                // Tear down the new communicator
                MPI_Comm_free( &teamComm );
                MPI_Group_free( &teamGroup );
                MPI_Group_free( &group );
            }

            if( l == L/2 )
            {
                MPI_Barrier( comm );
                if( rank == 0 )
                {
                    cout << "Right before switching: " << endl;
                    cout << "=====================================" << endl;
                }
                for( int m=0; m<S; ++m )
                {
                    if( rank == m )
                    {
                        cout << "  Rank: " << m << endl; 
                        for( unsigned i=0; i<(1u<<log2LocalSpatialBoxes); ++i )
                        {
                            cout << "  Space box: " << i << endl;
                            for( unsigned j=0; j<(1u<<log2LocalFreqBoxes); ++j )
                            {
                                cout << "    Freq box: " << j << endl;
                                for( unsigned t=0; t<Pow<q,d>::val; ++t )
                                {
                                    unsigned index = 
                                        j+(i<<log2LocalFreqBoxes);
                                    cout << "      Weight " << t << ": " 
                                         << weightSetList[index][t] << endl;
                                }
                            }
                        }
                    }
                    MPI_Barrier( comm );
                }

                if( rank == 0 )
                    cout << "Switching to spatial interpolation...";
                SwitchToSpatialInterp<Phi,R,d,q>
                ( L, log2LocalFreqBoxes, log2LocalSpatialBoxes,
                  log2LocalFreqBoxesPerDim, log2LocalSpatialBoxesPerDim,
                  myFreqBoxOffsets, mySpatialBoxOffsets, chebyGrid, 
                  weightSetList  );
                if( rank == 0 )
                    cout << "done." << endl;
            }
            
            // Print the state at the end of the loop
            MPI_Barrier( comm );
            if( rank == 0 )
            {
                cout << "At the end of level " << l << endl;
                cout << "=========================================" << endl;
            }
            for( int m=0; m<S; ++m )
            {
                if( rank == m )
                {
                    cout << "  Rank: " << m << endl;
                    cout << "  mySpatialBox:        ";
                    for( unsigned j=0; j<d; ++j )
                        cout << mySpatialBox[j] << " ";
                    cout << endl;
                    cout << "  mySpatialBoxOffsets: ";
                    for( unsigned j=0; j<d; ++j )
                        cout << mySpatialBoxOffsets[j] << " ";
                    cout << endl;
                    cout << "  mySpatialBoxWidths:  ";
                    for( unsigned j=0; j<d; ++j )
                        cout << mySpatialBoxWidths[j] << " ";
                    cout << endl;
                    cout << "  log2LocalSpatialBoxesPerDim: ";
                    for( unsigned j=0; j<d; ++j )
                        cout << log2LocalSpatialBoxesPerDim[j] << " ";
                    cout << endl << endl;
                }
                MPI_Barrier( comm );
            }
            for( int m=0; m<S; ++m )
            {
                if( rank == m )
                {
                    cout << "  Rank: " << m << endl;
                    cout << "  myFreqBox:        ";
                    for( unsigned j=0; j<d; ++j )
                        cout << myFreqBox[j] << " ";
                    cout << endl;
                    cout << "  myFreqBoxOffsets: ";
                    for( unsigned j=0; j<d; ++j )
                        cout << myFreqBoxOffsets[j] << " ";
                    cout << endl;
                    cout << "  myFreqBoxWidths:  ";
                    for( unsigned j=0; j<d; ++j )
                        cout << myFreqBoxWidths[j] << " ";
                    cout << endl;
                    cout << "  log2LocalFreqBoxesPerDim: ";
                    for( unsigned j=0; j<d; ++j )
                        cout << log2LocalFreqBoxesPerDim[j] << " ";
                    cout << endl << endl;
                }
                MPI_Barrier( comm );
            }
            for( int m=0; m<S; ++m )
            {
                if( rank == m )
                {
                    cout << "  Rank: " << m << endl; 
                    for( unsigned i=0; i<(1u<<log2LocalSpatialBoxes); ++i )
                    {
                        cout << "  Space box: " << i << endl;
                        for( unsigned j=0; j<(1u<<log2LocalFreqBoxes); ++j )
                        {
                            cout << "    Freq box: " << j << endl;
                            for( unsigned t=0; t<Pow<q,d>::val; ++t )
                            {
                                unsigned index = j+(i<<log2LocalFreqBoxes);
                                cout << "      Weight " << t << ": " 
                                     << weightSetList[index][t] << endl;
                            }
                        }
                    }
                }
                MPI_Barrier( comm );
            }
        }
        if( rank == 0 )
        {
            cout << "Finished forming low-rank approximations." << endl;
            cout<<"  "<<MPI_Wtime()-startTime<<" seconds."<<endl;
        }
        
        // Construct Low-Rank Potentials (LRPs) from weights
        {
            const R wA = static_cast<R>(1)/N;

            MPI_Barrier( comm );
            if( rank == 0 )
            {
                cout << "Low Rank Potentials" << endl;
                cout << "============================" << endl;
            }
            for( int m=0; m<S; ++m )
            {
                if( rank == m )
                {
                    cout << "Rank " << rank << ":" << endl;
                    // Fill in the LRPs
                    myLRPs.resize( 1<<(d*L-s) );
                    Array<unsigned,d> A( 0 );
                    for( unsigned i=0; i<myLRPs.size(); ++i )
                    {
                        cout << "  i=" << i << endl;
                        myLRPs[i].N = N;
                        cout << "  N=" << myLRPs[i].N << endl;

                        Array<R,d> x0A;

                        cout << "  A=";
                        for( unsigned j=0; j<d; ++j )
                            cout << A[j] << " ";
                        cout << endl;
                        for( unsigned j=0; j<d; ++j )
                            x0A[j] = mySpatialBoxOffsets[j] + A[j]*wA + wA/2;
                        for( unsigned j=0; j<d; ++j )
                            myLRPs[i].x0[j] = x0A[j];
                        cout << "  x0=";
                        for( unsigned j=0; j<d; ++j )
                            cout << myLRPs[i].x0[j] << " ";
                        cout << endl;
    
                        // Fill in the grid points of the box
                        for( unsigned t=0; t<Pow<q,d>::val; ++t )             
                            for( unsigned j=0; j<d; ++j )    
                                myLRPs[i].pointSet[t][j] = 
                                    x0A[j] + wA*chebyGrid[t][j];

                        // Fill in the weights for the grid points
                        myLRPs[i].weightSet = weightSetList[i];
                        cout << "  weights:" << endl;
                        for( unsigned t=0; t<Pow<q,d>::val; ++t )
                            cout << "    " << myLRPs[i].weightSet[t] << endl;
                        cout << endl;

                        TraverseHTree( log2LocalSpatialBoxesPerDim, A );
                    }
                }
            }
        }
    }
}

#endif /* BFIO_TRANSFORM_HPP */

