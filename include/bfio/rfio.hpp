/*
   ButterflyFIO: a distributed-memory fast algorithm for applying FIOs.
   Copyright (C) 2010-2011 Jack Poulson <jack.poulson@gmail.com>
 
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
#ifndef BFIO_RFIO_HPP
#define BFIO_RFIO_HPP 1

#include <iostream>
#include <memory>
#include <stdexcept>

#include "bfio/structures.hpp"
#include "bfio/tools.hpp"

#ifdef TIMING
namespace bfio {
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

template<typename R,std::size_t d,std::size_t q>
std::auto_ptr< const rfio::PotentialField<R,d,q> >
transform
( const rfio::Context<R,d,q>& context,
  const Plan<d>& plan,
  const Amplitude<R,d>& amplitude,
  const Phase<R,d>& phase,
  const Box<R,d>& sourceBox,
  const Box<R,d>& targetBox,
  const std::vector< Source<R,d> >& mySources )
{
#ifdef TIMING
    rfio::ResetTimers();
    rfio::timer.Start();
#endif
    typedef std::complex<R> C;
    const std::size_t q_to_d = Pow<q,d>::val;

    // Extract our communicator and its size
    MPI_Comm comm = plan.GetComm();
    int rank, numProcesses;
    MPI_Comm_rank( comm, &rank );
    MPI_Comm_size( comm, &numProcesses ); 

    // Get the problem-specific parameters
    const std::size_t N = plan.GetN();
    const std::size_t log2N = Log2( N );
    const Array<std::size_t,d>& myInitialSourceBoxCoords = 
        plan.GetMyInitialSourceBoxCoords();
    const Array<std::size_t,d>& log2InitialSourceBoxesPerDim = 
        plan.GetLog2InitialSourceBoxesPerDim();
    Array<std::size_t,d> mySourceBoxCoords = myInitialSourceBoxCoords;
    Array<std::size_t,d> log2SourceBoxesPerDim = log2InitialSourceBoxesPerDim;
    Box<R,d> mySourceBox;
    for( std::size_t j=0; j<d; ++j )
    {
        mySourceBox.widths[j] = 
            sourceBox.widths[j] / (1u<<log2SourceBoxesPerDim[j]);
        mySourceBox.offsets[j] = 
            sourceBox.offsets[j] + mySourceBox.widths[j]*mySourceBoxCoords[j];
    }

    Array<std::size_t,d> myTargetBoxCoords(0);
    Array<std::size_t,d> log2TargetBoxesPerDim(0);
    Box<R,d> myTargetBox;
    myTargetBox = targetBox;

    const std::size_t bootstrapSkip = plan.GetBootstrapSkip();

    // Compute the number of source and target boxes that our process is 
    // responsible for initializing weights in
    std::size_t log2WeightGridSize = 0;
    std::size_t log2LocalSourceBoxes = 0;
    std::size_t log2LocalTargetBoxes = 0;
    Array<std::size_t,d> log2LocalSourceBoxesPerDim;
    Array<std::size_t,d> log2LocalTargetBoxesPerDim(0);
    for( std::size_t j=0; j<d; ++j )
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
    for( std::size_t j=0; j<d; ++j )
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
    for( std::size_t level=bootstrapSkip+1; level<=log2N; ++level )
    {
        // Compute the width of the nodes at this level
        Array<R,d> wA;
        Array<R,d> wB;
        for( std::size_t j=0; j<d; ++j )
        {
            wA[j] = targetBox.widths[j] / (1<<level);
            wB[j] = sourceBox.widths[j] / (1<<(log2N-level));
        }

        if( log2LocalSourceBoxes >= d )
        {
            // Refine target domain and coursen the source domain
            for( std::size_t j=0; j<d; ++j )
            {
                --log2LocalSourceBoxesPerDim[j];
                ++log2LocalTargetBoxesPerDim[j];
            }
            log2LocalSourceBoxes -= d;
            log2LocalTargetBoxes += d;

            // Loop over boxes in target domain. 
            ConstrainedHTreeWalker<d> AWalker( log2LocalTargetBoxesPerDim );
            WeightGridList<R,d,q> oldWeightGridList( weightGridList );
            for( std::size_t targetIndex=0; 
                 targetIndex<(1u<<log2LocalTargetBoxes); 
                 ++targetIndex, AWalker.Walk() )
            {
                const Array<std::size_t,d> A = AWalker.State();

                // Compute coordinates and center of this target box
                Array<R,d> x0A;
                for( std::size_t j=0; j<d; ++j )
                    x0A[j] = myTargetBox.offsets[j] + (A[j]+0.5)*wA[j];

                // Loop over the B boxes in source domain
                ConstrainedHTreeWalker<d> BWalker( log2LocalSourceBoxesPerDim );
                for( std::size_t sourceIndex=0; 
                     sourceIndex<(1u<<log2LocalSourceBoxes); 
                     ++sourceIndex, BWalker.Walk() )
                {
                    const Array<std::size_t,d> B = BWalker.State();

                    // Compute coordinates and center of this source box
                    Array<R,d> p0B;
                    for( std::size_t j=0; j<d; ++j )
                        p0B[j] = mySourceBox.offsets[j] + (B[j]+0.5)*wB[j];

                    // We are storing the interaction pairs source-major
                    const std::size_t interactionIndex = 
                        sourceIndex + (targetIndex<<log2LocalSourceBoxes);

                    // Grab the interaction offset for the parent of target box 
                    // i interacting with the children of source box k
                    const std::size_t parentInteractionOffset = 
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
                        Array<R,d> x0Ap;
                        Array<std::size_t,d> globalA;
                        std::size_t ARelativeToAp = 0;
                        for( std::size_t j=0; j<d; ++j )
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
            const std::size_t log2NumMergingProcesses = d-log2LocalSourceBoxes;
            const std::size_t numMergingProcesses = 1u<<log2NumMergingProcesses;

            log2LocalSourceBoxes = 0; 
            for( std::size_t j=0; j<d; ++j )
                log2LocalSourceBoxesPerDim[j] = 0;

            // Fully refine target domain and coarsen source domain.
            // We partition the target domain after the SumScatter.
            const std::vector<std::size_t>& sourceDimsToMerge = 
                plan.GetSourceDimsToMerge( level );
            for( std::size_t i=0; i<log2NumMergingProcesses; ++i )
            {
                const std::size_t j = sourceDimsToMerge[i];
                if( mySourceBoxCoords[j] & 1 )
                    mySourceBox.offsets[j] -= mySourceBox.widths[j];
                mySourceBoxCoords[j] >>= 1;
                mySourceBox.widths[j] *= 2;
            }
            for( std::size_t j=0; j<d; ++j )
            {
                ++log2LocalTargetBoxesPerDim[j];
                ++log2LocalTargetBoxes;
            }

            // Compute the coordinates and center of this source box
            Array<R,d> p0B;
            for( std::size_t j=0; j<d; ++j )
                p0B[j] = mySourceBox.offsets[j] + 0.5*wB[j];

            // Form the partial weights by looping over the boxes in the  
            // target domain.
            ConstrainedHTreeWalker<d> AWalker( log2LocalTargetBoxesPerDim );
            WeightGridList<R,d,q> partialWeightGridList
            ( 1<<log2LocalTargetBoxes );
            for( std::size_t targetIndex=0; 
                 targetIndex<(1u<<log2LocalTargetBoxes); 
                 ++targetIndex, AWalker.Walk() )
            {
                const Array<std::size_t,d> A = AWalker.State();

                // Compute coordinates and center of this target box
                Array<R,d> x0A;
                for( std::size_t j=0; j<d; ++j )
                    x0A[j] = myTargetBox.offsets[j] + (A[j]+0.5)*wA[j];

                // Compute the interaction offset of A's parent interacting 
                // with the remaining local source boxes
                const std::size_t parentInteractionOffset = 
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
                    Array<R,d> x0Ap;
                    Array<std::size_t,d> globalA;
                    std::size_t ARelativeToAp = 0;
                    for( std::size_t j=0; j<d; ++j )
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
            std::vector<int> recvCounts( numMergingProcesses );
            for( std::size_t j=0; j<numMergingProcesses; ++j )
                recvCounts[j] = 2*weightGridList.Length()*q_to_d;
            // Currently two types of planned communication are supported, as 
            // they are the only required types for transforming and inverting 
            // the transform:
            //  1) partitions of dimensions 0 -> c
            //  2) partitions of dimensions c -> d-1
            // Both 1 and 2 include partitioning 0 -> d-1, but, in general, 
            // the second category never requires packing.
            const std::size_t log2SubclusterSize = 
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
                const std::size_t log2NumSubclusters = 
                    log2NumMergingProcesses-log2SubclusterSize;
                const std::size_t numSubclusters = 1u<<log2NumSubclusters;
                const std::size_t subclusterSize = 1u<<log2SubclusterSize;

                const std::size_t recvSize = recvCounts[0];
                const std::size_t sendSize = recvSize*numMergingProcesses;
                const std::size_t numChunksPerProcess = subclusterSize;
                const std::size_t chunkSize = recvSize / numChunksPerProcess;
                const R* partialBuffer = partialWeightGridList.Buffer();
                std::vector<R> sendBuffer( sendSize );
                for( std::size_t sc=0; sc<numSubclusters; ++sc )
                {
                    R* subclusterSendBuffer = 
                        &sendBuffer[sc*subclusterSize*recvSize];
                    const R* subclusterPartialBuffer = 
                        &partialBuffer[sc*subclusterSize*recvSize];
                    for( std::size_t p=0; p<subclusterSize; ++p )
                    {
                        R* processSend = &subclusterSendBuffer[p*recvSize];
                        for( std::size_t c=0; c<numChunksPerProcess; ++c )
                        {
                            std::memcpy 
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

            const std::vector<std::size_t>& targetDimsToCut = 
                plan.GetTargetDimsToCut( level );
            const std::vector<bool>& rightSideOfCut = 
                plan.GetRightSideOfCut( level );
            for( std::size_t i=0; i<log2NumMergingProcesses; ++i )
            {
                const std::size_t j = targetDimsToCut[i];
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
    std::auto_ptr< const rfio::PotentialField<R,d,q> > 
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

template<typename R,std::size_t d,std::size_t q>
std::auto_ptr< const rfio::PotentialField<R,d,q> >
ReducedFIO
( const rfio::Context<R,d,q>& context,
  const Plan<d>& plan,
  const Amplitude<R,d>& amplitude,
  const Phase<R,d>& phase,
  const Box<R,d>& sourceBox,
  const Box<R,d>& targetBox,
  const std::vector< Source<R,d> >& mySources )
{
    return rfio::transform
    ( context, plan, amplitude, phase, sourceBox, targetBox, mySources );
}

template<typename R,std::size_t d,std::size_t q>
std::auto_ptr< const rfio::PotentialField<R,d,q> >
ReducedFIO
( const rfio::Context<R,d,q>& context,
  const Plan<d>& plan,
  const Phase<R,d>& phase,
  const Box<R,d>& sourceBox,
  const Box<R,d>& targetBox,
  const std::vector< Source<R,d> >& mySources )
{
    UnitAmplitude<R,d> unitAmp;
    std::auto_ptr< const rfio::PotentialField<R,d,q> > u = 
    rfio::transform
    ( context, plan, unitAmp, phase, sourceBox, targetBox, mySources );
    return u;
}

} // bfio

#endif // BFIO_RFIO_HPP

