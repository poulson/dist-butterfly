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
#ifndef BFIO_INTERPOLATIVE_NUFT_HPP
#define BFIO_INTERPOLATIVE_NUFT_HPP 1

#include <iostream>
#include <memory>
#include <stdexcept>

#include "bfio/structures.hpp"
#include "bfio/tools.hpp"

#ifdef TIMING
namespace bfio {
namespace interpolative_nuft {

static bool alreadyTimed = false;

static bfio::Timer timer;
static bfio::Timer initializeCheckPotentialsTimer;
static bfio::Timer formCheckPotentialsTimer;
static bfio::Timer formEquivalentSourcesTimer;
static bfio::Timer sumScatterTimer;

static void
ResetTimers()
{
    timer.Reset();
    initializeCheckPotentialsTimer.Reset();
    formCheckPotentialsTimer.Reset();
    formEquivalentSourcesTimer.Reset();
    sumScatterTimer.Reset();
}

static void
PrintTimings()
{
#ifndef RELEASE
    if( !alreadyTimed )
        throw std::logic_error("You have not yet run InterpolativeNUFT.");
#endif
    std::cout << "InterpolativeNUFT timings:\n"
              << "------------------------------------------\n"
              << "InitializeCheckPotentials: "
              << initializeCheckPotentialsTimer.TotalTime() << " seconds.\n"
              << "FormCheckPotentials:       "
              << formCheckPotentialsTimer.TotalTime() << " seconds.\n"
              << "FormEquivalentSources:     "
              << formEquivalentSourcesTimer.TotalTime() << " seconds.\n"
              << "SumScatter:                "
              << sumScatterTimer.TotalTime() << " seconds.\n"
              << "Total: " << timer.TotalTime() << " seconds.\n" << std::endl;
}

} // interpolative_nuft
} // bfio
#endif

#include "bfio/interpolative_nuft/context.hpp"
#include "bfio/interpolative_nuft/form_equivalent_sources.hpp"
#include "bfio/interpolative_nuft/form_check_potentials.hpp"
#include "bfio/interpolative_nuft/initialize_check_potentials.hpp"
#include "bfio/interpolative_nuft/potential_field.hpp"

namespace bfio {

template<typename R,std::size_t d,std::size_t q>
std::auto_ptr< const interpolative_nuft::PotentialField<R,d,q> >
InterpolativeNUFT
( const interpolative_nuft::Context<R,d,q>& context,
  const Plan<d>& plan,
  const Box<R,d>& sourceBox,
  const Box<R,d>& targetBox,
  const std::vector< Source<R,d> >& mySources )
{
#ifdef TIMING
    interpolative_nuft::ResetTimers();
    interpolative_nuft::timer.Start();
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

    // Compute the number of leaf-level boxes in the source domain that 
    // our process is responsible for initializing the weights in. 
    std::size_t log2LocalSourceBoxes = 0;
    std::size_t log2LocalTargetBoxes = 0;
    Array<std::size_t,d> log2LocalSourceBoxesPerDim;
    Array<std::size_t,d> log2LocalTargetBoxesPerDim(0);
    for( std::size_t j=0; j<d; ++j )
    {
        log2LocalSourceBoxesPerDim[j] = log2N-log2SourceBoxesPerDim[j];
        log2LocalSourceBoxes += log2LocalSourceBoxesPerDim[j];
    }

    WeightGridList<R,d,q> weightGridList( 1<<log2LocalSourceBoxes );
#ifdef TIMING
    interpolative_nuft::initializeCheckPotentialsTimer.Start();
#endif
    interpolative_nuft::InitializeCheckPotentials
    ( context, plan, sourceBox, targetBox, mySourceBox, 
      log2LocalSourceBoxes, log2LocalSourceBoxesPerDim, mySources, 
      weightGridList );
#ifdef TIMING
    interpolative_nuft::initializeCheckPotentialsTimer.Stop();
    interpolative_nuft::formEquivalentSourcesTimer.Start();
#endif
    interpolative_nuft::FormEquivalentSources
    ( context, plan, mySourceBox, myTargetBox,
      log2LocalSourceBoxes, log2LocalTargetBoxes, 
      log2LocalSourceBoxesPerDim, log2LocalTargetBoxesPerDim, 
      weightGridList );
#ifdef TIMING
    interpolative_nuft::formEquivalentSourcesTimer.Stop();
#endif

    // Start the main recursion loop
    for( std::size_t level=1; level<=log2N; ++level )
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
            std::vector<R> prescalingArguments( q );
            Array<std::vector<R>,d> realPrescalings;
            Array<std::vector<R>,d> imagPrescalings;
            for( std::size_t j=0; j<d; ++j )
            {
                realPrescalings[j].resize(q);
                imagPrescalings[j].resize(q);
            }
            const std::vector<R>& chebyshevNodes = context.GetChebyshevNodes();
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

                // Store the prescaling factors for forming the check potentials
                for( std::size_t j=0; j<d; ++j )
                {
                    for( std::size_t t=0; t<q; ++t )
                        prescalingArguments[t] = 
                            TwoPi*x0A[j]*chebyshevNodes[t]*wB[j]/2;
                    SinCosBatch
                    ( prescalingArguments, 
                      imagPrescalings[j], realPrescalings[j] );
                }

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

#ifdef TIMING
		    interpolative_nuft::formCheckPotentialsTimer.Start();
#endif
                    interpolative_nuft::FormCheckPotentials
                    ( context, plan, level, realPrescalings, imagPrescalings,
                      x0A, p0B, wA, wB, parentInteractionOffset,
                      oldWeightGridList, weightGridList[interactionIndex] );
#ifdef TIMING
		    interpolative_nuft::formCheckPotentialsTimer.Stop();
#endif
                }
            }
#ifdef TIMING
	    interpolative_nuft::formEquivalentSourcesTimer.Start();
#endif
            interpolative_nuft::FormEquivalentSources
            ( context, plan, 
              mySourceBox, myTargetBox,
              log2LocalSourceBoxes, log2LocalTargetBoxes,
              log2LocalSourceBoxesPerDim, log2LocalTargetBoxesPerDim,
              weightGridList );
#ifdef TIMING
	    interpolative_nuft::formEquivalentSourcesTimer.Stop();
#endif
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
                p0B[j] = mySourceBox.offsets[j] + wB[j]/2;

            // Form the partial weights by looping over the boxes in the  
            // target domain.
            std::vector<R> prescalingArguments( q );
            Array<std::vector<R>,d> realPrescalings;
            Array<std::vector<R>,d> imagPrescalings;
            for( std::size_t j=0; j<d; ++j )
            {
                realPrescalings[j].resize(q);
                imagPrescalings[j].resize(q);
            }
            const std::vector<R>& chebyshevNodes = context.GetChebyshevNodes();
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

                // Store the prescaling factors for forming the check potentials
                for( std::size_t j=0; j<d; ++j )
                {
                    for( std::size_t t=0; t<q; ++t )
                        prescalingArguments[t] =
                            TwoPi*x0A[j]*chebyshevNodes[t]*wB[j]/2;
                    SinCosBatch
                    ( prescalingArguments, 
                      imagPrescalings[j], realPrescalings[j] );
                }

                // Compute the interaction offset of A's parent interacting 
                // with the remaining local source boxes
                const std::size_t parentInteractionOffset = 
                    ((targetIndex>>d)<<(d-log2NumMergingProcesses));

#ifdef TIMING
		interpolative_nuft::formCheckPotentialsTimer.Start();
#endif
                interpolative_nuft::FormCheckPotentials
                ( context, plan, level, realPrescalings, imagPrescalings,
                  x0A, p0B, wA, wB, parentInteractionOffset,
                  weightGridList, partialWeightGridList[targetIndex] );
#ifdef TIMING
		interpolative_nuft::formCheckPotentialsTimer.Stop();
#endif
            }

            // Scatter the summation of the weights
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
#ifdef TIMING
		interpolative_nuft::sumScatterTimer.Start();
#endif
                SumScatter    
                ( partialWeightGridList.Buffer(), weightGridList.Buffer(),
                  &recvCounts[0], clusterComm );
#ifdef TIMING
		interpolative_nuft::sumScatterTimer.Stop();
#endif
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
#ifdef TIMING
		interpolative_nuft::sumScatterTimer.Start();
#endif
                SumScatter
                ( &sendBuffer[0], weightGridList.Buffer(), 
                  &recvCounts[0], clusterComm );
#ifdef TIMING
		interpolative_nuft::sumScatterTimer.Stop();
#endif
            }

            // Adjust our local target box
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
            
            // Backtransform all of the potentials into equivalent sources
#ifdef TIMING
	    interpolative_nuft::formEquivalentSourcesTimer.Start();
#endif
            interpolative_nuft::FormEquivalentSources
            ( context, plan, mySourceBox, myTargetBox,
              log2LocalSourceBoxes, log2LocalTargetBoxes,
              log2LocalSourceBoxesPerDim, log2LocalTargetBoxesPerDim,
              weightGridList );
#ifdef TIMING
	    interpolative_nuft::formEquivalentSourcesTimer.Stop();
#endif
        }
    }

    // Construct the PotentialField
    std::auto_ptr< const interpolative_nuft::PotentialField<R,d,q> > 
        potentialField(
            new interpolative_nuft::PotentialField<R,d,q>
                ( context, sourceBox, myTargetBox, 
                  log2LocalTargetBoxesPerDim, weightGridList )
        );

#ifdef TIMING
    interpolative_nuft::timer.Stop();
    interpolative_nuft::alreadyTimed = true;
#endif
    return potentialField;
}

} // bfio

#endif // BFIO_INTERPOLATIVE_NUFT_HPP

