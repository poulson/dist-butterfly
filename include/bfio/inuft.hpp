/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of ButterflyFIO and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef BFIO_INUFT_HPP
#define BFIO_INUFT_HPP

#include <array>
#include <complex>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "bfio/structures.hpp"
#include "bfio/tools.hpp"

#ifdef TIMING
namespace bfio {

using std::array;
using std::memcpy;
using std::size_t;
using std::vector;

namespace inuft {

static bool alreadyTimed = false;

static bfio::Timer timer;
static bfio::Timer initializeCheckPotentialsTimer;
static bfio::Timer formCheckPotentialsTimer;
static bfio::Timer formEquivalentSourcesTimer;
static bfio::Timer sumScatterTimer;

static inline void
ResetTimers()
{
    timer.Reset();
    initializeCheckPotentialsTimer.Reset();
    formCheckPotentialsTimer.Reset();
    formEquivalentSourcesTimer.Reset();
    sumScatterTimer.Reset();
}

static inline void
PrintTimings()
{
#ifndef RELEASE
    if( !alreadyTimed )
        throw std::logic_error("You have not yet run INUFT.");
#endif
    std::cout << "INUFT timings:\n"
              << "------------------------------------------\n"
              << "InitializeCheckPotentials: "
              << initializeCheckPotentialsTimer.Total() << " seconds.\n"
              << "FormCheckPotentials:       "
              << formCheckPotentialsTimer.Total() << " seconds.\n"
              << "FormEquivalentSources:     "
              << formEquivalentSourcesTimer.Total() << " seconds.\n"
              << "SumScatter:                "
              << sumScatterTimer.Total() << " seconds.\n"
              << "Total: " << timer.Total() << " seconds.\n" << std::endl;
}

} // inuft
} // bfio
#endif

#include "bfio/inuft/context.hpp"
#include "bfio/inuft/form_equivalent_sources.hpp"
#include "bfio/inuft/form_check_potentials.hpp"
#include "bfio/inuft/initialize_check_potentials.hpp"
#include "bfio/inuft/potential_field.hpp"

namespace bfio {

template<typename R,size_t d,size_t q>
std::unique_ptr<const inuft::PotentialField<R,d,q>>
INUFT
( const inuft::Context<R,d,q>& context,
  const Plan<d>& plan,
  const Box<R,d>& sBox,
  const Box<R,d>& tBox,
  const vector<Source<R,d>>& mySources )
{
#ifdef TIMING
    inuft::ResetTimers();
    inuft::timer.Start();
#endif
    typedef complex<R> C;
    const size_t q_to_d = Pow<q,d>::val;

    const Direction direction = context.GetDirection();
    const R SignedTwoPi = ( direction==FORWARD ? -TwoPi<R>() : TwoPi<R>() );

    // Extract our communicator and its size
    MPI_Comm comm = plan.GetComm();
    int rank, numProcesses;
    MPI_Comm_rank( comm, &rank );
    MPI_Comm_size( comm, &numProcesses ); 

    // Get the problem-specific parameters
    const size_t N = plan.GetN();
    const size_t log2N = Log2( N );
    const array<size_t,d>& myInitialSBoxCoords = 
        plan.GetMyInitialSourceBoxCoords();
    const array<size_t,d>& log2InitialSBoxesPerDim = 
        plan.GetLog2InitialSourceBoxesPerDim();
    array<size_t,d> mySBoxCoords = myInitialSBoxCoords;
    array<size_t,d> log2SBoxesPerDim = log2InitialSBoxesPerDim;
    Box<R,d> mySBox;
    for( size_t j=0; j<d; ++j )
    {
        mySBox.widths[j] = sBox.widths[j] / (1u<<log2SBoxesPerDim[j]);
        mySBox.offsets[j] = sBox.offsets[j] + mySBox.widths[j]*mySBoxCoords[j];
    }

    array<size_t,d> myTBoxCoords, log2TBoxesPerDim;
    myTBoxCoords.fill(0);
    log2TBoxesPerDim.fill(0);
    Box<R,d> myTBox;
    myTBox = tBox;

    // Compute the number of leaf-level boxes in the source domain that 
    // our process is responsible for initializing the weights in. 
    size_t log2LocalSBoxes = 0;
    size_t log2LocalTBoxes = 0;
    array<size_t,d> log2LocalSBoxesPerDim, log2LocalTBoxesPerDim;
    log2LocalTBoxesPerDim.fill(0);
    for( size_t j=0; j<d; ++j )
    {
        log2LocalSBoxesPerDim[j] = log2N-log2SBoxesPerDim[j];
        log2LocalSBoxes += log2LocalSBoxesPerDim[j];
    }

    WeightGridList<R,d,q> weightGridList( 1<<log2LocalSBoxes );
#ifdef TIMING
    inuft::initializeCheckPotentialsTimer.Start();
#endif
    inuft::InitializeCheckPotentials
    ( context, plan, sBox, tBox, mySBox, 
      log2LocalSBoxes, log2LocalSBoxesPerDim, mySources, weightGridList );
#ifdef TIMING
    inuft::initializeCheckPotentialsTimer.Stop();
    inuft::formEquivalentSourcesTimer.Start();
#endif
    inuft::FormEquivalentSources
    ( context, plan, mySBox, myTBox,
      log2LocalSBoxes, log2LocalTBoxes, 
      log2LocalSBoxesPerDim, log2LocalTBoxesPerDim, weightGridList );
#ifdef TIMING
    inuft::formEquivalentSourcesTimer.Stop();
#endif

    // Start the main recursion loop
    for( size_t level=1; level<=log2N; ++level )
    {
        // Compute the width of the nodes at this level
        array<R,d> wA;
        array<R,d> wB;
        for( size_t j=0; j<d; ++j )
        {
            wA[j] = tBox.widths[j] / (1<<level);
            wB[j] = sBox.widths[j] / (1<<(log2N-level));
        }

        if( log2LocalSBoxes >= d )
        {
            // Refine target domain and coursen the source domain
            for( size_t j=0; j<d; ++j )
            {
                --log2LocalSBoxesPerDim[j];
                ++log2LocalTBoxesPerDim[j];
            }
            log2LocalSBoxes -= d;
            log2LocalTBoxes += d;

            // Loop over boxes in target domain. 
            vector<R> prescalingArguments( q );
            array<vector<R>,d> realPrescalings, imagPrescalings;
            for( size_t j=0; j<d; ++j )
            {
                realPrescalings[j].resize(q);
                imagPrescalings[j].resize(q);
            }
            const vector<R>& chebyshevNodes = context.GetChebyshevNodes();
            ConstrainedHTreeWalker<d> AWalker( log2LocalTBoxesPerDim );
            WeightGridList<R,d,q> oldWeightGridList( weightGridList );
            for( size_t tIndex=0; 
                 tIndex<(1u<<log2LocalTBoxes); ++tIndex, AWalker.Walk() )
            {
                const array<size_t,d> A = AWalker.State();

                // Compute coordinates and center of this target box
                array<R,d> x0A;
                for( size_t j=0; j<d; ++j )
                    x0A[j] = myTBox.offsets[j] + (A[j]+0.5)*wA[j];

                // Store the prescaling factors for forming the check potentials
                for( size_t j=0; j<d; ++j )
                {
                    for( size_t t=0; t<q; ++t )
                        prescalingArguments[t] = 
                            SignedTwoPi*x0A[j]*chebyshevNodes[t]*wB[j]/2;
                    SinCosBatch
                    ( prescalingArguments, 
                      imagPrescalings[j], realPrescalings[j] );
                }

                // Loop over the B boxes in source domain
                ConstrainedHTreeWalker<d> BWalker( log2LocalSBoxesPerDim );
                for( size_t sIndex=0; 
                     sIndex<(1u<<log2LocalSBoxes); ++sIndex, BWalker.Walk() )
                {
                    const array<size_t,d> B = BWalker.State();

                    // Compute coordinates and center of this source box
                    array<R,d> p0B;
                    for( size_t j=0; j<d; ++j )
                        p0B[j] = mySBox.offsets[j] + (B[j]+0.5)*wB[j];

                    // We are storing the interaction pairs source-major
                    const size_t iIndex = sIndex + (tIndex<<log2LocalSBoxes);

                    // Grab the interaction offset for the parent of target box 
                    // i interacting with the children of source box k
                    const size_t parentIOffset = 
                        ((tIndex>>d)<<(log2LocalSBoxes+d)) + (sIndex<<d);

#ifdef TIMING
		    inuft::formCheckPotentialsTimer.Start();
#endif
                    inuft::FormCheckPotentials
                    ( context, plan, level, realPrescalings, imagPrescalings,
                      x0A, p0B, wA, wB, parentIOffset,
                      oldWeightGridList, weightGridList[iIndex] );
#ifdef TIMING
		    inuft::formCheckPotentialsTimer.Stop();
#endif
                }
            }
#ifdef TIMING
	    inuft::formEquivalentSourcesTimer.Start();
#endif
            inuft::FormEquivalentSources
            ( context, plan, mySBox, myTBox, log2LocalSBoxes, log2LocalTBoxes,
              log2LocalSBoxesPerDim, log2LocalTBoxesPerDim, weightGridList );
#ifdef TIMING
	    inuft::formEquivalentSourcesTimer.Stop();
#endif
        }
        else 
        {
            const size_t log2NumMergingProcesses = d-log2LocalSBoxes;
            const size_t numMergingProcesses = 1u<<log2NumMergingProcesses;

            log2LocalSBoxes = 0; 
            for( size_t j=0; j<d; ++j )
                log2LocalSBoxesPerDim[j] = 0;

            // Fully refine target domain and coarsen source domain.
            // We partition the target domain after the SumScatter.
            const vector<size_t>& sDimsToMerge = 
                plan.GetSourceDimsToMerge( level );
            for( size_t i=0; i<log2NumMergingProcesses; ++i )
            {
                const size_t j = sDimsToMerge[i];
                if( mySBoxCoords[j] & 1 )
                    mySBox.offsets[j] -= mySBox.widths[j];
                mySBoxCoords[j] >>= 1;
                mySBox.widths[j] *= 2;
            }
            for( size_t j=0; j<d; ++j )
            {
                ++log2LocalTBoxesPerDim[j];
                ++log2LocalTBoxes;
            }

            // Compute the coordinates and center of this source box
            array<R,d> p0B;
            for( size_t j=0; j<d; ++j )
                p0B[j] = mySBox.offsets[j] + wB[j]/2;

            // Form the partial weights by looping over the boxes in the  
            // target domain.
            vector<R> prescalingArguments( q );
            array<vector<R>,d> realPrescalings;
            array<vector<R>,d> imagPrescalings;
            for( size_t j=0; j<d; ++j )
            {
                realPrescalings[j].resize(q);
                imagPrescalings[j].resize(q);
            }
            const vector<R>& chebyshevNodes = context.GetChebyshevNodes();
            ConstrainedHTreeWalker<d> AWalker( log2LocalTBoxesPerDim );
            WeightGridList<R,d,q> partialWeightGridList( 1<<log2LocalTBoxes );
            for( size_t tIndex=0; 
                 tIndex<(1u<<log2LocalTBoxes); ++tIndex, AWalker.Walk() )
            {
                const array<size_t,d> A = AWalker.State();

                // Compute coordinates and center of this target box
                array<R,d> x0A;
                for( size_t j=0; j<d; ++j )
                    x0A[j] = myTBox.offsets[j] + (A[j]+0.5)*wA[j];

                // Store the prescaling factors for forming the check potentials
                for( size_t j=0; j<d; ++j )
                {
                    for( size_t t=0; t<q; ++t )
                        prescalingArguments[t] =
                            SignedTwoPi*x0A[j]*chebyshevNodes[t]*wB[j]/2;
                    SinCosBatch
                    ( prescalingArguments, 
                      imagPrescalings[j], realPrescalings[j] );
                }

                // Compute the interaction offset of A's parent interacting 
                // with the remaining local source boxes
                const size_t parentIOffset = 
                    ((tIndex>>d)<<(d-log2NumMergingProcesses));

#ifdef TIMING
		inuft::formCheckPotentialsTimer.Start();
#endif
                inuft::FormCheckPotentials
                ( context, plan, level, realPrescalings, imagPrescalings,
                  x0A, p0B, wA, wB, parentIOffset,
                  weightGridList, partialWeightGridList[tIndex] );
#ifdef TIMING
		inuft::formCheckPotentialsTimer.Stop();
#endif
            }

            // Scatter the summation of the weights
#ifdef TIMING
            inuft::sumScatterTimer.Start();
#endif
            vector<int> recvCounts( numMergingProcesses );
            for( size_t j=0; j<numMergingProcesses; ++j )
                recvCounts[j] = 2*weightGridList.Length()*q_to_d;
            // Currently two types of planned communication are supported, as 
            // they are the only required types for transforming and inverting 
            // the transform:
            //  1) partitions of dimensions 0 -> c
            //  2) partitions of dimensions c -> d-1
            // Both 1 and 2 include partitioning 0 -> d-1, but, in general, 
            // the second category never requires packing.
            const size_t log2SubclusterSize = 
                plan.GetLog2SubclusterSize( level );
            if( log2SubclusterSize == 0 )
            {
                MPI_Comm clusterComm = plan.GetClusterComm( level );
                SumScatter    
                ( partialWeightGridList.Buffer(), weightGridList.Buffer(),
                  &recvCounts[0], clusterComm );
            }
            else
            {
                const size_t log2NumSubclusters = 
                    log2NumMergingProcesses-log2SubclusterSize;
                const size_t numSubclusters = 1u<<log2NumSubclusters;
                const size_t subclusterSize = 1u<<log2SubclusterSize;

                const size_t recvSize = recvCounts[0];
                const size_t sendSize = recvSize*numMergingProcesses;
                const size_t numChunksPerProcess = subclusterSize;
                const size_t chunkSize = recvSize / numChunksPerProcess;
                const R* partialBuffer = partialWeightGridList.Buffer();
                vector<R> sendBuffer( sendSize );
                for( size_t sc=0; sc<numSubclusters; ++sc )
                {
                    R* subclusterSendBuffer = 
                        &sendBuffer[sc*subclusterSize*recvSize];
                    const R* subclusterPartialBuffer = 
                        &partialBuffer[sc*subclusterSize*recvSize];
                    for( size_t p=0; p<subclusterSize; ++p )
                    {
                        R* processSend = &subclusterSendBuffer[p*recvSize];
                        for( size_t c=0; c<numChunksPerProcess; ++c )
                        {
                            memcpy 
                            ( &processSend[c*chunkSize],
                              &subclusterPartialBuffer
                              [(p+c*subclusterSize)*chunkSize],
                              chunkSize*sizeof(R) );
                        }
                    }
                }
                MPI_Comm clusterComm = plan.GetClusterComm( level );
                SumScatter
                ( &sendBuffer[0], weightGridList.Buffer(), 
                  &recvCounts[0], clusterComm );
            }
#ifdef TIMING
            inuft::sumScatterTimer.Stop();
#endif

            // Adjust our local target box
            const vector<size_t>& tDimsToCut = plan.GetTargetDimsToCut( level );
            const vector<bool>& rightSideOfCut = 
                plan.GetRightSideOfCut( level );
            for( size_t i=0; i<log2NumMergingProcesses; ++i )
            {
                const size_t j = tDimsToCut[i];
                myTBox.widths[j] *= 0.5;
                myTBoxCoords[j] *= 2;
                if( rightSideOfCut[i] )
                {
                    myTBoxCoords[j] |= 1;
                    myTBox.offsets[j] += myTBox.widths[j];
                }
                --log2LocalTBoxesPerDim[j];
                --log2LocalTBoxes;
            }
            
            // Backtransform all of the potentials into equivalent sources
#ifdef TIMING
	    inuft::formEquivalentSourcesTimer.Start();
#endif
            inuft::FormEquivalentSources
            ( context, plan, mySBox, myTBox, log2LocalSBoxes, log2LocalTBoxes,
              log2LocalSBoxesPerDim, log2LocalTBoxesPerDim, weightGridList );
#ifdef TIMING
	    inuft::formEquivalentSourcesTimer.Stop();
#endif
        }
    }

    // Construct the PotentialField
    std::unique_ptr<const inuft::PotentialField<R,d,q>> 
        potentialField(
            new inuft::PotentialField<R,d,q>
                ( context, sBox, myTBox, log2LocalTBoxesPerDim, weightGridList )
        );

#ifdef TIMING
    inuft::timer.Stop();
    inuft::alreadyTimed = true;
#endif
    return potentialField;
}

} // bfio

#endif // ifndef BFIO_INUFT_HPP
