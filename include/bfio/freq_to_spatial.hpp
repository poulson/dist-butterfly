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
#pragma once
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
// defined by the mapped amplitude and phase functions, Amp and Phi. This 
// allows one to call the function with their own functor with potentially no 
// performance penalty. R is the datatype for representing a Real and d is the 
// spatial and frequency dimension. q is the number of points in each dimension 
// of the Chebyshev tensor-product grid (q^d points total).
template<typename R,unsigned d,unsigned q>
void
FreqToSpatial
( const unsigned N,
  const Box<R,d>& freqBox,
  const Box<R,d>& spatialBox,
  const std::vector< Source<R,d> >& mySources,
        std::vector< LowRankPotential<R,d,q> >& myLRPs,
        MPI_Comm comm )
{
    typedef std::complex<R> C;
    const unsigned q_to_d = Pow<q,d>::val;

    // Extract our communicator and its size
    int rank, numProcesses;
    MPI_Comm_rank( comm, &rank );
    MPI_Comm_size( comm, &numProcesses ); 
    std::bitset<sizeof(int)*8> rankBits(rank); 

    // Extract our amplitude and phase functions, as well as N
    const AmplitudeFunctor<R,d>& Amp = myLRPs[0].GetAmplitudeFunctor();
    const PhaseFunctor<R,d>& Phi = myLRPs[0].GetPhaseFunctor();

    // Assert that N and size are powers of 2
    if( ! IsPowerOfTwo(N) )
        throw std::runtime_error( "Must use a power of 2 problem size" );
    if( ! IsPowerOfTwo(numProcesses) ) 
        throw std::runtime_error( "Must use a power of 2 number of processes" );
    if( myLRPs.size() != NumLocalBoxes<d>( N, comm ) )
        throw std::runtime_error( "Incorrect length for vector of LRPs" );
    const unsigned log2N = Log2( N );
    const unsigned log2NumProcesses = Log2( numProcesses );
    if( log2NumProcesses > d*log2N )
        throw std::runtime_error( "Cannot use more than N^d processes" );

    // Determine the number of boxes in each dimension of the frequency
    // domain by applying the partitions cyclically over the d dimensions.
    // We can simultaneously compute the indices of our box.
    Array<unsigned,d> myFreqBoxCoords;
    Array<unsigned,d> log2FreqBoxesPerDim;
    Box<R,d> myFreqBox;
    LocalFreqPartitionData
    ( freqBox, myFreqBox, myFreqBoxCoords, log2FreqBoxesPerDim, comm );

    Array<unsigned,d> mySpatialBoxCoords(0);
    Array<unsigned,d> log2SpatialBoxesPerDim(0);
    Box<R,d> mySpatialBox;
    mySpatialBox = spatialBox;

    // Compute the number of leaf-level boxes in the frequency domain that 
    // our process is responsible for initializing the weights in. 
    unsigned log2LocalFreqBoxes = 0;
    unsigned log2LocalSpatialBoxes = 0;
    Array<unsigned,d> log2LocalFreqBoxesPerDim;
    Array<unsigned,d> log2LocalSpatialBoxesPerDim(0);
    for( unsigned j=0; j<d; ++j )
    {
        log2LocalFreqBoxesPerDim[j] = log2N-log2FreqBoxesPerDim[j];
        log2LocalFreqBoxes += log2LocalFreqBoxesPerDim[j];
    }

    // Compute the Chebyshev grid over [-1/2,+1/2]^d
    std::vector< Array<R,d> > chebyGrid( q_to_d );
    for( unsigned t=0; t<q_to_d; ++t )
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
    WeightGridList<R,d,q> weightGridList( 1<<log2LocalFreqBoxes );
    freq_to_spatial::InitializeWeights<R,d,q>
    ( Amp, Phi, N, mySources, chebyGrid, freqBox, spatialBox, myFreqBox,
      log2LocalFreqBoxes, log2LocalFreqBoxesPerDim, weightGridList );

    // Start the main recursion loop
    unsigned numSpaceCuts = 0;
    unsigned nextSpatialDimToCut = d-1;
    if( log2N == 0 || log2N == 1 )
    {
        freq_to_spatial::SwitchToSpatialInterp<R,d,q>
        ( Amp, Phi, log2N, freqBox, spatialBox, myFreqBox, mySpatialBox,
          log2LocalFreqBoxes, log2LocalSpatialBoxes,
          log2LocalFreqBoxesPerDim, log2LocalSpatialBoxesPerDim,
          chebyGrid, weightGridList );
    }
    for( unsigned level=1; level<=log2N; ++level )
    {
        // Compute the width of the nodes at this level
        Array<R,d> wA;
        Array<R,d> wB;
        for( unsigned j=0; j<d; ++j )
        {
            wA[j] = spatialBox.widths[j] / (1<<level);
            wB[j] = freqBox.widths[j] / (1<<(log2N-level));
        }

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
            WeightGridList<R,d,q> oldWeightGridList( weightGridList );
            for( unsigned i=0; 
                 i<(1u<<log2LocalSpatialBoxes); 
                 ++i, AWalker.Walk() )
            {
                const Array<unsigned,d> A = AWalker.State();

                // Compute coordinates and center of this spatial box
                Array<R,d> x0A;
                for( unsigned j=0; j<d; ++j )
                    x0A[j] = mySpatialBox.offsets[j] + (A[j]+0.5)*wA[j];

                // Loop over the B boxes in frequency domain
                CHTreeWalker<d> BWalker( log2LocalFreqBoxesPerDim );
                for( unsigned k=0; 
                     k<(1u<<log2LocalFreqBoxes); 
                     ++k, BWalker.Walk() )
                {
                    const Array<unsigned,d> B = BWalker.State();

                    // Compute coordinates and center of this freq box
                    Array<R,d> p0B;
                    for( unsigned j=0; j<d; ++j )
                        p0B[j] = myFreqBox.offsets[j] + (B[j]+0.5)*wB[j];

                    const unsigned key = k + (i<<log2LocalFreqBoxes);
                    const unsigned parentOffset = 
                        ((i>>d)<<(log2LocalFreqBoxes+d)) + (k<<d);
                    if( level <= log2N/2 )
                    {
                        freq_to_spatial::FreqWeightRecursion<R,d,q>
                        ( Amp, Phi, 0, 0, N, chebyGrid, 
                          x0A, p0B, wB, parentOffset,
                          oldWeightGridList, weightGridList[key] );
                    }
                    else
                    {
                        Array<R,d> x0Ap;
                        Array<unsigned,d> globalA;
                        unsigned ARelativeToAp = 0;
                        for( unsigned j=0; j<d; ++j )
                        {
                            globalA[j] = 
                                (mySpatialBoxCoords[j]<<
                                 log2LocalSpatialBoxesPerDim[j])+A[j];
                            x0Ap[j] = spatialBox.offsets[j] + 
                                      (globalA[j]|1)*wA[j];
                            ARelativeToAp |= (globalA[j]&1)<<j;
                        }
                        freq_to_spatial::SpatialWeightRecursion<R,d,q>
                        ( Amp, Phi, 0, 0, N, chebyGrid, 
                          ARelativeToAp, x0A, x0Ap, p0B, wA, wB,
                          parentOffset, oldWeightGridList, 
                          weightGridList[key] );
                    }
                }
            }
        }
        else 
        {
            // There are currently 2^(d*(log2N-level)) leaves. The frequency 
            // partitioning is implied by reading the rank bits left-to-
            // right starting with bit log2NumProcesses-1, but the spatial 
            // partitioning is implied by reading the rank bits right-to-left. 
            //
            // We notice that our consistency in the cyclic bisection of
            // the frequency domain means that if log2NumMergingProcesses=a, 
            // then we communicate with 1 other process in each of the first 
            // a of d dimensions. Getting these ranks is implicit in the
            // tree structure.
            const unsigned log2NumMergingProcesses = d-log2LocalFreqBoxes;
            const unsigned numMergingProcesses = 1u<<log2NumMergingProcesses;

            log2LocalFreqBoxes = 0; 
            for( unsigned j=0; j<d; ++j )
                log2LocalFreqBoxesPerDim[j] = 0;

            // Pull the group out of the global communicator
            MPI_Group group;
            MPI_Comm_group( comm, &group );

            // Construct the group for our local team
            MPI_Group teamGroup;
            int myTeamRank = 0;
            // Mask log2NumMergingProcesses bits offset by numSpaceCuts bits
            const int startRank = 
                rank & ~((numMergingProcesses-1)<<numSpaceCuts);
            const unsigned log2Stride = numSpaceCuts;
            
            std::vector<int> ranks( numMergingProcesses );
            for( unsigned j=0; j<numMergingProcesses; ++j )
            {
                // We need to reverse the order of the last 
                // log2NumMergingProcesses bits of j and add the result 
                // multiplied by the stride onto the startRank
                unsigned jReversed = 0;
                for( unsigned k=0; k<log2NumMergingProcesses; ++k )
                    jReversed |= ((j>>k)&1)<<(log2NumMergingProcesses-1-k);
                ranks[j] = startRank+(jReversed<<log2Stride);
                if( ranks[j] == rank )
                    myTeamRank = j;
            }
            MPI_Group_incl
            ( group, numMergingProcesses, &ranks[0], &teamGroup );

            // Construct the local team communicator from the team group
            MPI_Comm teamComm;
            MPI_Comm_create( comm, teamGroup, &teamComm );

            // Fully refine spatial domain and coarsen frequency domain.
            // We partition the spatial domain after the SumScatter.
            for( unsigned j=0; j<d; ++j )
            {
                ++log2LocalSpatialBoxesPerDim[j];
                ++log2LocalSpatialBoxes;
            }
            for( unsigned j=0; j<log2NumMergingProcesses; ++j ) 
            {
                if( myFreqBoxCoords[j] & 1 )
                {
                    myFreqBox.offsets[j] -= freqBox.offsets[j];
                    myFreqBox.offsets[j] *= 
                        static_cast<R>(myFreqBoxCoords[j]-1)/myFreqBoxCoords[j];
                    myFreqBox.offsets[j] += freqBox.offsets[j];
                }
                myFreqBoxCoords[j] >>= 1;
                myFreqBox.widths[j] *= 2;
            }

            // Compute the coordinates and center of this freq box
            Array<R,d> p0B;
            for( unsigned j=0; j<d; ++j )
                p0B[j] = myFreqBox.offsets[j] + 0.5*wB[j];

            // Form the partial weights. 
            //
            // Loop over boxes in spatial domain. 'i' will represent the
            // leaf # w.r.t. the tree implied by cyclically assigning
            // the spatial bisections across the d dims. Thus if we 
            // distribute the data cyclically in the reverse order over 
            // the d dims, then the ReduceScatter will not require any
            // packing or unpacking.
            CHTreeWalker<d> AWalker( log2LocalSpatialBoxesPerDim );
            WeightGridList<R,d,q> partialWeightGridList
            ( 1<<log2LocalSpatialBoxes );
            for( unsigned i=0; 
                 i<(1u<<log2LocalSpatialBoxes); 
                 ++i, AWalker.Walk() )
            {
                const Array<unsigned,d> A = AWalker.State();

                // Compute coordinates and center of this spatial box
                Array<R,d> x0A;
                for( unsigned j=0; j<d; ++j )
                    x0A[j] = mySpatialBox.offsets[j] + (A[j]+0.5)*wA[j];

                const unsigned parentOffset = 
                    ((i>>d)<<(d-log2NumMergingProcesses));
                if( level <= log2N/2 )
                {
                    freq_to_spatial::FreqWeightRecursion<R,d,q>
                    ( Amp, Phi, log2NumMergingProcesses, myTeamRank, 
                      N, chebyGrid, x0A, p0B, wB, parentOffset,
                      weightGridList, partialWeightGridList[i] );
                }
                else
                {
                    Array<R,d> x0Ap;
                    Array<unsigned,d> globalA;
                    unsigned ARelativeToAp = 0;
                    for( unsigned j=0; j<d; ++j )
                    {
                        globalA[j] = 
                            (mySpatialBoxCoords[j]<<
                             log2LocalSpatialBoxesPerDim[j])+A[j];
                        x0Ap[j] = spatialBox.offsets[j] + (globalA[j]|1)*wA[j];
                        ARelativeToAp |= (globalA[j]&1)<<j;
                    }
                    freq_to_spatial::SpatialWeightRecursion<R,d,q>
                    ( Amp, Phi, log2NumMergingProcesses, myTeamRank,
                      N, chebyGrid, ARelativeToAp,
                      x0A, x0Ap, p0B, wA, wB, parentOffset,
                      weightGridList, partialWeightGridList[i] );
                }
            }

            // Scatter the summation of the weights
            std::vector<int> recvCounts( numMergingProcesses );
            for( unsigned j=0; j<numMergingProcesses; ++j )
                recvCounts[j] = weightGridList.Length()*q_to_d;
            SumScatter
            ( &(partialWeightGridList[0][0]), &(weightGridList[0][0]), 
              &recvCounts[0], teamComm );

            for( unsigned j=0; j<log2NumMergingProcesses; ++j )
            {
                mySpatialBox.widths[nextSpatialDimToCut] *= 0.5;
                mySpatialBoxCoords[nextSpatialDimToCut] *= 2;
                if( rankBits[numSpaceCuts] ) 
                {
                    mySpatialBoxCoords[nextSpatialDimToCut] |= 1;
                    mySpatialBox.offsets[nextSpatialDimToCut]
                        += mySpatialBox.widths[nextSpatialDimToCut];
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
        if( level==log2N/2 )
        {
            freq_to_spatial::SwitchToSpatialInterp<R,d,q>
            ( Amp, Phi, log2N, freqBox, spatialBox, myFreqBox, mySpatialBox,
              log2LocalFreqBoxes, log2LocalSpatialBoxes,
              log2LocalFreqBoxesPerDim, log2LocalSpatialBoxesPerDim,
              chebyGrid, weightGridList );
        }
    }
    
    // Construct Low-Rank Potentials (LRPs) from weights
    {
        Array<R,d> wA;
        for( unsigned j=0; j<d; ++j )
            wA[j] = spatialBox.widths[j] / N;

        CHTreeWalker<d> AWalker( log2LocalSpatialBoxesPerDim );
        for( unsigned i=0; i<myLRPs.size(); ++i, AWalker.Walk() )
        {
            const Array<unsigned,d> A = AWalker.State();

            Array<R,d> x0A;
            for( unsigned j=0; j<d; ++j )
                x0A[j] = mySpatialBox.offsets[j] + (A[j]+0.5)*wA[j];
            myLRPs[i].SetSpatialWidths( wA );
            myLRPs[i].SetSpatialCenter( x0A );

            Array<R,d> p0B;
            for( unsigned j=0; j<d; ++j )
                p0B[j] = freqBox.widths[j] / 2;
            myLRPs[i].SetFreqCenter( p0B );

            PointGrid<R,d,q> pointGrid;
            for( unsigned t=0; t<q_to_d; ++t )             
                for( unsigned j=0; j<d; ++j )    
                    pointGrid[t][j] = x0A[j] + wA[j]*chebyGrid[t][j];
            myLRPs[i].SetPointGrid( pointGrid );

            myLRPs[i].SetWeightGrid( weightGridList[i] );
        }
    }
}

} // bfio

#endif // BFIO_FREQ_TO_SPATIAL_HPP

