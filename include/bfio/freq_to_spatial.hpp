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
#ifndef BFIO_FREQ_TO_SPATIAL_HPP
#define BFIO_FREQ_TO_SPATIAL_HPP 1

#include <iostream>
#include <stdexcept>

#include "bfio/structures.hpp"
#include "bfio/tools.hpp"

#include "bfio/freq_to_spatial/initialize_weights.hpp"
#include "bfio/freq_to_spatial/freq_weight_recursion.hpp"
#include "bfio/freq_to_spatial/switch_to_spatial_interp.hpp"
#include "bfio/freq_to_spatial/spatial_weight_recursion.hpp"

namespace bfio {

// Applies the butterfly algorithm for the Fourier integral operator 
// defined by the mapped phase function, Phi. This allows one to call the 
// function with their own functor with potentially no performance penalty. 
// R is the datatype for representing a Real and d is the spatial and 
// frequency dimension. q is the number of points in each dimension of the 
// Chebyshev tensor-product grid (q^d points total).
template<typename R,unsigned d,unsigned q>
void
FreqToSpatial
( const AmplitudeFunctor<R,d>& Amp,
  const PhaseFunctor<R,d>& Phi,
  const unsigned N,
  const std::vector< Source<R,d> >& mySources,
        std::vector< LowRankPotential<R,d,q> >& myLRPs,
        MPI_Comm comm )
{
    typedef std::complex<R> C;

    int rank, S;
    MPI_Comm_rank( comm, &rank );
    MPI_Comm_size( comm, &S    ); 
    std::bitset<sizeof(int)*8> rankBits(rank); 

    // Assert that N and size are powers of 2
    if( ! IsPowerOfTwo(N) )
        throw std::runtime_error( "Must use a power of 2 problem size" );
    if( ! IsPowerOfTwo(S) ) 
        throw std::runtime_error( "Must use a power of 2 number of processes" );
    if( myLRPs.size() != NumLocalBoxes<d>( N, comm ) )
        throw std::runtime_error( "Incorrect length for vector of LRPs" );
    const unsigned L = Log2( N );
    const unsigned s = Log2( S );
    if( s > d*L )
        throw std::runtime_error( "Cannot use more than N^d processes" );

    // Determine the number of boxes in each dimension of the frequency
    // domain by applying the partitions cyclically over the d dimensions.
    // We can simultaneously compute the indices of our box.
    Array<unsigned,d> myFreqBox;
    Array<unsigned,d> log2FreqBoxesPerDim;
    Array<R,d> myFreqBoxWidths;
    Array<R,d> myFreqBoxOffsets;
    LocalFreqPartitionData
    ( myFreqBox, log2FreqBoxesPerDim, 
      myFreqBoxWidths, myFreqBoxOffsets, comm );

    Array<unsigned,d> mySpatialBox(0);
    Array<unsigned,d> log2SpatialBoxesPerDim(0);
    Array<R,d> mySpatialBoxWidths(1);
    Array<R,d> mySpatialBoxOffsets(0);

    // Compute the number of 1/N width boxes in the frequency domain that 
    // our process is responsible for initializing the weights in. Also
    // initialize each box being responsible for all of the spatial domain.
    unsigned log2LocalFreqBoxes = 0;
    unsigned log2LocalSpatialBoxes = 0;
    Array<unsigned,d> log2LocalFreqBoxesPerDim;
    Array<unsigned,d> log2LocalSpatialBoxesPerDim(0);
    for( unsigned j=0; j<d; ++j )
    {
        log2LocalFreqBoxesPerDim[j] = L-log2FreqBoxesPerDim[j];
        log2LocalFreqBoxes += log2LocalFreqBoxesPerDim[j];
    }

    // Compute the Chebyshev grid over [-1/2,+1/2]^d
    std::vector< Array<R,d> > chebyGrid( Pow<q,d>::val );
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

    // Initialize the weights using Lagrangian interpolation on the 
    // smooth component of the kernel.
    WeightSetList<R,d,q> weightSetList( 1<<log2LocalFreqBoxes );
    freq_to_spatial::InitializeWeights<R,d,q>
    ( Phi, N, mySources, chebyGrid, myFreqBoxWidths, myFreqBox,
      log2LocalFreqBoxes, log2LocalFreqBoxesPerDim, weightSetList );

    // Start the main recursion loop
    unsigned numSpaceCuts = 0;
    unsigned nextSpatialDimToCut = d-1;
    if( L == 0 || L == 1 )
    {
        freq_to_spatial::SwitchToSpatialInterp<R,d,q>
        ( Amp, Phi, L, log2LocalFreqBoxes, log2LocalSpatialBoxes,
          log2LocalFreqBoxesPerDim, log2LocalSpatialBoxesPerDim,
           myFreqBoxOffsets, mySpatialBoxOffsets, chebyGrid, 
           weightSetList  );
    }
    for( unsigned l=1; l<=L; ++l )
    {
        // Compute the width of the nodes at level l
        const R wA = static_cast<R>(1) / (1<<l);
        const R wB = static_cast<R>(1) / (1<<(L-l));

        if( log2LocalFreqBoxes >= d )
        {
            // Refine spatial domain and coursen the frequency domain
            for( unsigned j=0; j<d; ++j )
            {
                --log2LocalFreqBoxesPerDim[j];
                ++log2LocalSpatialBoxesPerDim[j];
            }
            log2LocalFreqBoxes -= d;
            log2LocalSpatialBoxes += d;

            // Loop over boxes in spatial domain. 'i' will represent the
            // leaf # w.r.t. the tree implied by cyclically assigning
            // the spatial bisections across the d dims.
            CHTreeWalker<d> AWalker( log2LocalSpatialBoxesPerDim );
            WeightSetList<R,d,q> oldWeightSetList( weightSetList );
            for( unsigned i=0; 
                 i<(1u<<log2LocalSpatialBoxes); 
                 ++i, AWalker.Walk()           )
            {
                const Array<unsigned,d> A = AWalker.State();

                // Compute coordinates and center of this spatial box
                Array<R,d> x0A;
                for( unsigned j=0; j<d; ++j )
                    x0A[j] = mySpatialBoxOffsets[j] + A[j]*wA + wA/2;

                // Loop over the B boxes in frequency domain
                CHTreeWalker<d> BWalker( log2LocalFreqBoxesPerDim );
                for( unsigned k=0; 
                     k<(1u<<log2LocalFreqBoxes); 
                     ++k, BWalker.Walk()        )
                {
                    const Array<unsigned,d> B = BWalker.State();

                    // Compute coordinates and center of this freq box
                    Array<R,d> p0B;
                    for( unsigned j=0; j<d; ++j )
                        p0B[j] = myFreqBoxOffsets[j] + B[j]*wB + wB/2;

                    const unsigned key = k + (i<<log2LocalFreqBoxes);
                    const unsigned parentOffset = 
                        ((i>>d)<<(log2LocalFreqBoxes+d)) + (k<<d);
                    if( l <= L/2 )
                    {
                        freq_to_spatial::FreqWeightRecursion<R,d,q>
                        ( Phi, 0, 0, N, chebyGrid, 
                          x0A, p0B, wB, parentOffset,
                          oldWeightSetList, weightSetList[key] );
                    }
                    else
                    {
                        Array<R,d> x0Ap;
                        Array<unsigned,d> globalA;
                        unsigned ARelativeToAp = 0;
                        for( unsigned j=0; j<d; ++j )
                        {
                            globalA[j] = 
                                (mySpatialBox[j]<<
                                 log2LocalSpatialBoxesPerDim[j])+A[j];
                            x0Ap[j] = (globalA[j]/2)*2*wA + wA;
                            ARelativeToAp |= (globalA[j]&1)<<j;
                        }
                        freq_to_spatial::SpatialWeightRecursion<R,d,q>
                        ( Phi, 0, 0, N, chebyGrid, 
                          ARelativeToAp, x0A, x0Ap, p0B, wA, wB,
                          parentOffset, oldWeightSetList, 
                          weightSetList[key] );
                    }
                }
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
            MPI_Group teamGroup;
            int myTeamRank = 0;
            // Mask log2Procs bits offset by numSpaceCuts bits
            const int startRank = 
                rank & ~(((1<<log2Procs)-1)<<numSpaceCuts);
            const unsigned log2Stride = numSpaceCuts;
            
            std::vector<int> ranks( 1<<log2Procs );
            for( unsigned j=0; j<(1u<<log2Procs); ++j )
            {
                // We need to reverse the order of the last log2Procs
                // bits of j and add the result multiplied by the stride
                // onto the startRank
                unsigned jReversed = 0;
                for( unsigned k=0; k<log2Procs; ++k )
                    jReversed |= ((j>>k)&1)<<(log2Procs-1-k);
                ranks[j] = startRank+(jReversed<<log2Stride);
                if( ranks[j] == rank )
                    myTeamRank = j;
            }
            MPI_Group_incl
            ( group, 1<<log2Procs, &ranks[0], &teamGroup );

            // Construct the local team communicator from the team group
            MPI_Comm  teamComm;
            MPI_Comm_create( comm, teamGroup, &teamComm );

            // Fully refine spatial domain and coarsen frequency domain.
            // We partition the spatial domain after the SumScatter.
            for( unsigned j=0; j<d; ++j )
            {
                ++log2LocalSpatialBoxesPerDim[j];
                ++log2LocalSpatialBoxes;
            }
            for( unsigned j=0; j<log2Procs; ++j ) 
            {
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
            // Loop over boxes in spatial domain. 'i' will represent the
            // leaf # w.r.t. the tree implied by cyclically assigning
            // the spatial bisections across the d dims. Thus if we 
            // distribute the data cyclically in the reverse order over 
            // the d dims, then the ReduceScatter will not require any
            // packing or unpacking.
            CHTreeWalker<d> AWalker( log2LocalSpatialBoxesPerDim );
            WeightSetList<R,d,q> partialWeightSetList
            ( 1<<log2LocalSpatialBoxes );
            for( unsigned i=0; 
                 i<(1u<<log2LocalSpatialBoxes); 
                 ++i, AWalker.Walk()           )
            {
                const Array<unsigned,d> A = AWalker.State();

                // Compute coordinates and center of this spatial box
                Array<R,d> x0A;
                for( unsigned j=0; j<d; ++j )
                    x0A[j] = mySpatialBoxOffsets[j] + A[j]*wA + wA/2;

                const unsigned parentOffset = ((i>>d)<<(d-log2Procs));
                if( l <= L/2 )
                {
                    freq_to_spatial::FreqWeightRecursion<R,d,q>
                    ( Phi, log2Procs, myTeamRank, 
                      N, chebyGrid, x0A, p0B, wB, parentOffset,
                      weightSetList, partialWeightSetList[i]    );
                }
                else
                {
                    Array<R,d> x0Ap;
                    Array<unsigned,d> globalA;
                    unsigned ARelativeToAp = 0;
                    for( unsigned j=0; j<d; ++j )
                    {
                        globalA[j] = 
                            (mySpatialBox[j]<<
                             log2LocalSpatialBoxesPerDim[j])+A[j];
                        x0Ap[j] = (globalA[j]/2)*2*wA + wA;
                        ARelativeToAp |= (globalA[j]&1)<<j;
                    }
                    freq_to_spatial::SpatialWeightRecursion<R,d,q>
                    ( Phi, log2Procs, myTeamRank,
                      N, chebyGrid, ARelativeToAp,
                      x0A, x0Ap, p0B, wA, wB, parentOffset,
                      weightSetList, partialWeightSetList[i] );
                }
            }

            // Scatter the summation of the weights
            std::vector<int> recvCounts( 1<<log2Procs );
            for( unsigned j=0; j<(1u<<log2Procs); ++j )
                recvCounts[j] = weightSetList.Length()*Pow<q,d>::val;
            SumScatter
            ( &(partialWeightSetList[0][0]), &(weightSetList[0][0]), 
              &recvCounts[0], teamComm );

            for( unsigned j=0; j<log2Procs; ++j )
            {
                mySpatialBoxWidths[nextSpatialDimToCut] /= 2;
                mySpatialBox[nextSpatialDimToCut] <<= 1;
                if( rankBits[numSpaceCuts] ) 
                {
                    mySpatialBox[nextSpatialDimToCut] |= 1;
                    mySpatialBoxOffsets[nextSpatialDimToCut] 
                        += mySpatialBoxWidths[nextSpatialDimToCut];
                }
                --log2LocalSpatialBoxesPerDim[nextSpatialDimToCut];
                --log2LocalSpatialBoxes;
                ++numSpaceCuts;
                nextSpatialDimToCut = (nextSpatialDimToCut+d-1) % d;
            }

            // Tear down the new communicator
            MPI_Comm_free( &teamComm );
            MPI_Group_free( &teamGroup );
            MPI_Group_free( &group );
        }
        if( l==L/2 )
        {
            freq_to_spatial::SwitchToSpatialInterp<R,d,q>
            ( Amp, Phi, L, log2LocalFreqBoxes, log2LocalSpatialBoxes,
              log2LocalFreqBoxesPerDim, log2LocalSpatialBoxesPerDim,
              myFreqBoxOffsets, mySpatialBoxOffsets, chebyGrid, 
              weightSetList );
        }
    }
    
    // Construct Low-Rank Potentials (LRPs) from weights
    {
        const R wA = static_cast<R>(1)/N;

        CHTreeWalker<d> AWalker( log2LocalSpatialBoxesPerDim );
        for( unsigned i=0; i<myLRPs.size(); ++i, AWalker.Walk() )
        {
            const Array<unsigned,d> A = AWalker.State();

            Array<R,d> x0A;
            for( unsigned j=0; j<d; ++j )
                x0A[j] = mySpatialBoxOffsets[j] + A[j]*wA + wA/2;
            myLRPs[i].SetSpatialCenter( x0A );

            Array<R,d> p0B;
            for( unsigned j=0; j<d; ++j )
                p0B[j] = static_cast<R>(1)/2;
            myLRPs[i].SetFreqCenter( p0B );

            PointSet<R,d,q> pointSet;
            for( unsigned t=0; t<Pow<q,d>::val; ++t )             
                for( unsigned j=0; j<d; ++j )    
                    pointSet[t][j] = x0A[j] + wA*chebyGrid[t][j];
            myLRPs[i].SetPointSet( pointSet );

            myLRPs[i].SetWeightSet( weightSetList[i] );
        }
    }
}

} // bfio

#endif /* BFIO_FREQ_TO_SPATIAL_HPP */

