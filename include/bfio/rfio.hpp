/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of ButterflyFIO and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef BFIO_RFIO_HPP
#define BFIO_RFIO_HPP

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
using std::complex;
using std::size_t;
using std::vector;

namespace rfio {

static bool alreadyTimed = false;

static bfio::Timer timer;
static bfio::Timer initializeWeightsTimer;
static bfio::Timer sourceWeightRecursionTimer;
static bfio::Timer switchToTargetInterpTimer;
static bfio::Timer targetWeightRecursionTimer;
static bfio::Timer sumScatterTimer;

static inline void
ResetTimers()
{
    timer.Reset();
    initializeWeightsTimer.Reset();
    sourceWeightRecursionTimer.Reset();
    switchToTargetInterpTimer.Reset();
    targetWeightRecursionTimer.Reset();
    sumScatterTimer.Reset();
}

static inline void
PrintTimings()
{
#ifndef RELEASE
    if( !alreadyTimed )
	throw std::logic_error("You have not yet run ReducedFIO.");
#endif
    std::cout << "ReducedFIO timings:\n"
	      << "--------------------------------------------\n"
              << "InitializeWeights:     "
              << initializeWeightsTimer.TotalTime() << " seconds.\n"
              << "SourceWeightRecursion: "
              << sourceWeightRecursionTimer.TotalTime() << " seconds.\n"
              << "SwitchToTargetInterp:  "
              << switchToTargetInterpTimer.TotalTime() << " seconds.\n"
              << "TargetWeightRecursion: "
              << targetWeightRecursionTimer.TotalTime() << " seconds.\n"
              << "SumScatter:            "
              << sumScatterTimer.TotalTime() << " seconds.\n"
              << "Total: " << timer.TotalTime() << " seconds.\n" << std::endl;
}

} // rfio
} // bfio
#endif

#include "bfio/rfio/context.hpp"
#include "bfio/rfio/potential_field.hpp"

#include "bfio/rfio/initialize_weights.hpp"
#include "bfio/rfio/source_weight_recursion.hpp"
#include "bfio/rfio/switch_to_target_interp.hpp"
#include "bfio/rfio/target_weight_recursion.hpp"

namespace bfio {
namespace rfio {

template<typename R,size_t d,size_t q>
std::unique_ptr<const rfio::PotentialField<R,d,q>>
transform
( const rfio::Context<R,d,q>& context,
  const Plan<d>& plan,
  const Amplitude<R,d>& amplitude,
  const Phase<R,d>& phase,
  const Box<R,d>& sourceBox,
  const Box<R,d>& targetBox,
  const vector<Source<R,d>>& mySources )
{
#ifdef TIMING
    rfio::ResetTimers();
    rfio::timer.Start();
#endif
    typedef complex<R> C;
    const size_t q_to_d = Pow<q,d>::val;

    // Extract our communicator and its size
    MPI_Comm comm = plan.GetComm();
    int rank, numProcesses;
    MPI_Comm_rank( comm, &rank );
    MPI_Comm_size( comm, &numProcesses ); 

    // Get the problem-specific parameters
    const size_t N = plan.GetN();
    const size_t log2N = Log2( N );
    const array<size_t,d>& myInitialSourceBoxCoords = 
        plan.GetMyInitialSourceBoxCoords();
    const array<size_t,d>& log2InitialSourceBoxesPerDim = 
        plan.GetLog2InitialSourceBoxesPerDim();
    array<size_t,d> mySourceBoxCoords = myInitialSourceBoxCoords;
    array<size_t,d> log2SourceBoxesPerDim = log2InitialSourceBoxesPerDim;
    Box<R,d> mySourceBox;
    for( size_t j=0; j<d; ++j )
    {
        mySourceBox.widths[j] = 
            sourceBox.widths[j] / (1u<<log2SourceBoxesPerDim[j]);
        mySourceBox.offsets[j] = 
            sourceBox.offsets[j] + mySourceBox.widths[j]*mySourceBoxCoords[j];
    }

    array<size_t,d> myTargetBoxCoords, log2TargetBoxesPerDim;
    myTargetBoxCoords.fill(0);
    log2TargetBoxesPerDim.fill(0);
    Box<R,d> myTargetBox;
    myTargetBox = targetBox;

    const size_t bootstrapSkip = plan.GetBootstrapSkip();

    // Compute the number of source and target boxes that our process is 
    // responsible for initializing weights in
    size_t log2WeightGridSize = 0;
    size_t log2LocalSourceBoxes = 0;
    size_t log2LocalTargetBoxes = 0;
    array<size_t,d> log2LocalSourceBoxesPerDim,
                    log2LocalTargetBoxesPerDim;
    log2LocalTargetBoxesPerDim.fill(0);
    for( size_t j=0; j<d; ++j )
    {
        if( log2N-log2SourceBoxesPerDim[j] >= bootstrapSkip )
            log2LocalSourceBoxesPerDim[j] = 
                (log2N-log2SourceBoxesPerDim[j]) - bootstrapSkip;
        else
            log2LocalSourceBoxesPerDim[j] = 0;
        log2LocalTargetBoxesPerDim[j] = bootstrapSkip;
        log2LocalSourceBoxes += log2LocalSourceBoxesPerDim[j];
        log2LocalTargetBoxes += log2LocalTargetBoxesPerDim[j];
        log2WeightGridSize += log2N-log2SourceBoxesPerDim[j];
    }

    // Initialize the weights using Lagrangian interpolation on the 
    // smooth component of the kernel.
    WeightGridList<R,d,q> weightGridList( 1u<<log2WeightGridSize );
#ifdef TIMING
    rfio::initializeWeightsTimer.Start();
#endif
    rfio::InitializeWeights
    ( context, plan, phase, sourceBox, targetBox, mySourceBox, 
      log2LocalSourceBoxes, log2LocalSourceBoxesPerDim, mySources, 
      weightGridList );
#ifdef TIMING
    rfio::initializeWeightsTimer.Stop();
#endif

    // Now cut the target domain if necessary
    for( size_t j=0; j<d; ++j )
    {
        if( log2LocalSourceBoxesPerDim[j] == 0 )
        {
            log2LocalTargetBoxesPerDim[j] -= 
                bootstrapSkip - (log2N-log2SourceBoxesPerDim[j]);
            log2LocalTargetBoxes -=
                bootstrapSkip - (log2N-log2SourceBoxesPerDim[j]);
        }
    }

    // Start the main recursion loop
    if( bootstrapSkip == log2N/2 )
    {
#ifdef TIMING
	rfio::switchToTargetInterpTimer.Start();
#endif
        rfio::SwitchToTargetInterp
        ( context, plan, amplitude, phase, sourceBox, targetBox, mySourceBox, 
          myTargetBox, log2LocalSourceBoxes, log2LocalTargetBoxes,
          log2LocalSourceBoxesPerDim, log2LocalTargetBoxesPerDim,
          weightGridList );
#ifdef TIMING
	rfio::switchToTargetInterpTimer.Stop();
#endif
    }
    for( size_t level=bootstrapSkip+1; level<=log2N; ++level )
    {
        // Compute the width of the nodes at this level
        array<R,d> wA;
        array<R,d> wB;
        for( size_t j=0; j<d; ++j )
        {
            wA[j] = targetBox.widths[j] / (1<<level);
            wB[j] = sourceBox.widths[j] / (1<<(log2N-level));
        }

        if( log2LocalSourceBoxes >= d )
        {
            // Refine target domain and coursen the source domain
            for( size_t j=0; j<d; ++j )
            {
                --log2LocalSourceBoxesPerDim[j];
                ++log2LocalTargetBoxesPerDim[j];
            }
            log2LocalSourceBoxes -= d;
            log2LocalTargetBoxes += d;

            // Loop over boxes in target domain. 
            ConstrainedHTreeWalker<d> AWalker( log2LocalTargetBoxesPerDim );
            WeightGridList<R,d,q> oldWeightGridList( weightGridList );
            for( size_t targetIndex=0; 
                 targetIndex<(1u<<log2LocalTargetBoxes); 
                 ++targetIndex, AWalker.Walk() )
            {
                const array<size_t,d> A = AWalker.State();

                // Compute coordinates and center of this target box
                array<R,d> x0A;
                for( size_t j=0; j<d; ++j )
                    x0A[j] = myTargetBox.offsets[j] + (A[j]+0.5)*wA[j];

                // Loop over the B boxes in source domain
                ConstrainedHTreeWalker<d> BWalker( log2LocalSourceBoxesPerDim );
                for( size_t sourceIndex=0; 
                     sourceIndex<(1u<<log2LocalSourceBoxes); 
                     ++sourceIndex, BWalker.Walk() )
                {
                    const array<size_t,d> B = BWalker.State();

                    // Compute coordinates and center of this source box
                    array<R,d> p0B;
                    for( size_t j=0; j<d; ++j )
                        p0B[j] = mySourceBox.offsets[j] + (B[j]+0.5)*wB[j];

                    // We are storing the interaction pairs source-major
                    const size_t interactionIndex = 
                        sourceIndex + (targetIndex<<log2LocalSourceBoxes);

                    // Grab the interaction offset for the parent of target box 
                    // i interacting with the children of source box k
                    const size_t parentInteractionOffset = 
                        ((targetIndex>>d)<<(log2LocalSourceBoxes+d)) + 
                        (sourceIndex<<d);

                    if( level <= log2N/2 )
                    {
#ifdef TIMING
			rfio::sourceWeightRecursionTimer.Start();
#endif
                        rfio::SourceWeightRecursion
                        ( context, plan, phase, level, x0A, p0B, wB, 
                          parentInteractionOffset, oldWeightGridList,
                          weightGridList[interactionIndex] );
#ifdef TIMING
			rfio::sourceWeightRecursionTimer.Stop();
#endif
                    }
                    else
                    {
                        array<R,d> x0Ap;
                        array<size_t,d> globalA;
                        size_t ARelativeToAp = 0;
                        for( size_t j=0; j<d; ++j )
                        {
                            globalA[j] = 
                                (myTargetBoxCoords[j]<<
                                 log2LocalTargetBoxesPerDim[j])+A[j];
                            x0Ap[j] = targetBox.offsets[j] + 
                                      (globalA[j]|1)*wA[j];
                            ARelativeToAp |= (globalA[j]&1)<<j;
                        }
#ifdef TIMING
			rfio::targetWeightRecursionTimer.Start();
#endif
                        rfio::TargetWeightRecursion
                        ( context, plan, phase, level,
                          ARelativeToAp, x0A, x0Ap, p0B, wA, wB,
                          parentInteractionOffset, oldWeightGridList, 
                          weightGridList[interactionIndex] );
#ifdef TIMING
			rfio::targetWeightRecursionTimer.Stop();
#endif
                    }
                }
            }
        }
        else 
        {
            const size_t log2NumMergingProcesses = d-log2LocalSourceBoxes;
            const size_t numMergingProcesses = 1u<<log2NumMergingProcesses;

            log2LocalSourceBoxes = 0; 
            for( size_t j=0; j<d; ++j )
                log2LocalSourceBoxesPerDim[j] = 0;

            // Fully refine target domain and coarsen source domain.
            // We partition the target domain after the SumScatter.
            const vector<size_t>& sourceDimsToMerge = 
                plan.GetSourceDimsToMerge( level );
            for( size_t i=0; i<log2NumMergingProcesses; ++i )
            {
                const size_t j = sourceDimsToMerge[i];
                if( mySourceBoxCoords[j] & 1 )
                    mySourceBox.offsets[j] -= mySourceBox.widths[j];
                mySourceBoxCoords[j] >>= 1;
                mySourceBox.widths[j] *= 2;
            }
            for( size_t j=0; j<d; ++j )
            {
                ++log2LocalTargetBoxesPerDim[j];
                ++log2LocalTargetBoxes;
            }

            // Compute the coordinates and center of this source box
            array<R,d> p0B;
            for( size_t j=0; j<d; ++j )
                p0B[j] = mySourceBox.offsets[j] + 0.5*wB[j];

            // Form the partial weights by looping over the boxes in the  
            // target domain.
            ConstrainedHTreeWalker<d> AWalker( log2LocalTargetBoxesPerDim );
            WeightGridList<R,d,q> partialWeightGridList
            ( 1<<log2LocalTargetBoxes );
            for( size_t targetIndex=0; 
                 targetIndex<(1u<<log2LocalTargetBoxes); 
                 ++targetIndex, AWalker.Walk() )
            {
                const array<size_t,d> A = AWalker.State();

                // Compute coordinates and center of this target box
                array<R,d> x0A;
                for( size_t j=0; j<d; ++j )
                    x0A[j] = myTargetBox.offsets[j] + (A[j]+0.5)*wA[j];

                // Compute the interaction offset of A's parent interacting 
                // with the remaining local source boxes
                const size_t parentInteractionOffset = 
                    ((targetIndex>>d)<<(d-log2NumMergingProcesses));
                if( level <= log2N/2 )
                {
#ifdef TIMING
		    rfio::sourceWeightRecursionTimer.Start();
#endif
                    rfio::SourceWeightRecursion
                    ( context, plan, phase, level, x0A, p0B, wB,
                      parentInteractionOffset, weightGridList,
                      partialWeightGridList[targetIndex] );
#ifdef TIMING
		    rfio::sourceWeightRecursionTimer.Stop();
#endif
                }
                else
                {
                    array<R,d> x0Ap;
                    array<size_t,d> globalA;
                    size_t ARelativeToAp = 0;
                    for( size_t j=0; j<d; ++j )
                    {
                        globalA[j] = 
                            (myTargetBoxCoords[j]<<
                             log2LocalTargetBoxesPerDim[j])+A[j];
                        x0Ap[j] = targetBox.offsets[j] + (globalA[j]|1)*wA[j];
                        ARelativeToAp |= (globalA[j]&1)<<j;
                    }
#ifdef TIMING
		    rfio::targetWeightRecursionTimer.Start();
#endif
                    rfio::TargetWeightRecursion
                    ( context, plan, phase, level,
                      ARelativeToAp, x0A, x0Ap, p0B, wA, wB,
                      parentInteractionOffset, weightGridList, 
                      partialWeightGridList[targetIndex] );
#ifdef TIMING
		    rfio::targetWeightRecursionTimer.Stop();
#endif
                }
            }

            // Scatter the summation of the weights
#ifdef TIMING
            rfio::sumScatterTimer.Start();
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
            rfio::sumScatterTimer.Stop();
#endif

            const vector<size_t>& targetDimsToCut = 
                plan.GetTargetDimsToCut( level );
            const vector<bool>& rightSideOfCut = 
                plan.GetRightSideOfCut( level );
            for( size_t i=0; i<log2NumMergingProcesses; ++i )
            {
                const size_t j = targetDimsToCut[i];
                myTargetBox.widths[j] *= 0.5;
                myTargetBoxCoords[j] *= 2;
                if( rightSideOfCut[i] )
                {
                    myTargetBoxCoords[j] |= 1;
                    myTargetBox.offsets[j] += myTargetBox.widths[j];
                }
                --log2LocalTargetBoxesPerDim[j];
                --log2LocalTargetBoxes;
            }
        }
        if( level==log2N/2 )
        {
#ifdef TIMING
	    rfio::switchToTargetInterpTimer.Start();
#endif
            rfio::SwitchToTargetInterp
            ( context, plan, amplitude, phase, sourceBox, targetBox, 
              mySourceBox, myTargetBox, log2LocalSourceBoxes, 
              log2LocalTargetBoxes, log2LocalSourceBoxesPerDim, 
              log2LocalTargetBoxesPerDim, weightGridList );
#ifdef TIMING
	    rfio::switchToTargetInterpTimer.Stop();
#endif
        }
    }

    // Construct the FIO PotentialField
    std::unique_ptr<const rfio::PotentialField<R,d,q>> 
    potentialField( 
        new rfio::PotentialField<R,d,q>
        ( context, amplitude, phase, sourceBox, myTargetBox, 
          myTargetBoxCoords, log2LocalTargetBoxesPerDim, weightGridList )
    );

#ifdef TIMING
    rfio::timer.Stop();
    rfio::alreadyTimed = true;
#endif
    return potentialField;
}

} // rfio

template<typename R,size_t d,size_t q>
std::unique_ptr< const rfio::PotentialField<R,d,q> >
ReducedFIO
( const rfio::Context<R,d,q>& context,
  const Plan<d>& plan,
  const Amplitude<R,d>& amplitude,
  const Phase<R,d>& phase,
  const Box<R,d>& sourceBox,
  const Box<R,d>& targetBox,
  const vector<Source<R,d>>& mySources )
{
    return rfio::transform
    ( context, plan, amplitude, phase, sourceBox, targetBox, mySources );
}

template<typename R,size_t d,size_t q>
std::unique_ptr< const rfio::PotentialField<R,d,q> >
ReducedFIO
( const rfio::Context<R,d,q>& context,
  const Plan<d>& plan,
  const Phase<R,d>& phase,
  const Box<R,d>& sourceBox,
  const Box<R,d>& targetBox,
  const vector<Source<R,d>>& mySources )
{
    UnitAmplitude<R,d> unitAmp;
    auto u = rfio::transform
    ( context, plan, unitAmp, phase, sourceBox, targetBox, mySources );
    return u;
}

} // bfio

#endif // ifndef BFIO_RFIO_HPP
