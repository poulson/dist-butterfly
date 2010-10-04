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
#ifndef BFIO_GENERAL_FIO_HPP
#define BFIO_GENERAL_FIO_HPP 1

#include <iostream>
#include <memory>
#include <stdexcept>

#include "bfio/structures.hpp"
#include "bfio/tools.hpp"

#include "bfio/general_fio/context.hpp"
#include "bfio/general_fio/potential_field.hpp"

#include "bfio/general_fio/initialize_weights.hpp"
#include "bfio/general_fio/source_weight_recursion.hpp"
#include "bfio/general_fio/switch_to_target_interp.hpp"
#include "bfio/general_fio/target_weight_recursion.hpp"

namespace bfio {
namespace general_fio {

namespace {
inline void 
ReportWithFlush( int rank, const std::string& msg )
{
#ifdef TRACE
    std::cout << msg;
    std::cout.flush();
#endif
}
inline void
Report( int rank, const std::string& msg )
{
#ifdef TRACE
    std::cout << msg << std::endl;
#endif
}
}

template<typename R,std::size_t d,std::size_t q>
std::auto_ptr< const general_fio::PotentialField<R,d,q> >
transform
( const general_fio::Context<R,d,q>& context,
  const Plan<d>& plan,
  const AmplitudeFunctor<R,d>& Amp,
  const PhaseFunctor<R,d>& Phi,
  const Box<R,d>& sourceBox,
  const Box<R,d>& targetBox,
  const std::vector< Source<R,d> >& mySources )
{
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

    // Initialize the weights using Lagrangian interpolation on the 
    // smooth component of the kernel.
    ReportWithFlush( rank, "  Initializing weights..." );
    WeightGridList<R,d,q> weightGridList( 1<<log2LocalSourceBoxes );
    general_fio::InitializeWeights<R,d,q>
    ( context, plan, Phi, sourceBox, targetBox, mySourceBox, 
      log2LocalSourceBoxes, log2LocalSourceBoxesPerDim, mySources, 
      weightGridList );
    Report( rank, "done." );

    // Start the main recursion loop
    if( log2N == 0 || log2N == 1 )
    {
        ReportWithFlush( rank, "  Switching to target interpolation..." );
        general_fio::SwitchToTargetInterp<R,d,q>
        ( context, plan, Amp, Phi, sourceBox, targetBox, mySourceBox, 
          myTargetBox, log2LocalSourceBoxes, log2LocalTargetBoxes,
          log2LocalSourceBoxesPerDim, log2LocalTargetBoxesPerDim,
          weightGridList );
        Report( rank, "done." );
    }
    std::size_t numCommunications = 0;
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
                        ReportWithFlush( rank, "  Source weight recursion..." );
                        general_fio::SourceWeightRecursion<R,d,q>
                        ( context, plan, Phi, 0, 0, x0A, p0B, wB, 
                          parentInteractionOffset, oldWeightGridList,
                          weightGridList[interactionIndex] );
                        Report( rank, "done." );
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
                        ReportWithFlush( rank, "  Target weight recursion..." );
                        general_fio::TargetWeightRecursion<R,d,q>
                        ( context, plan, Phi, 0, 0, 
                          ARelativeToAp, x0A, x0Ap, p0B, wA, wB,
                          parentInteractionOffset, oldWeightGridList, 
                          weightGridList[interactionIndex] );
                        Report( rank, "done." );
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
            const Array<bool,d>& sourceDimsToMerge = 
                plan.GetSourceDimsToMerge( numCommunications );
            for( std::size_t j=0; j<d; ++j )
            {
                ++log2LocalTargetBoxesPerDim[j];
                ++log2LocalTargetBoxes;
                if( sourceDimsToMerge[j] )
                {
                    if( mySourceBoxCoords[j] & 1 )
                    {
                        mySourceBox.offsets[j] -= sourceBox.offsets[j];
                        mySourceBox.offsets[j] *=
                            static_cast<R>(mySourceBoxCoords[j]-1) / 
                            mySourceBoxCoords[j];
                        mySourceBox.offsets[j] += sourceBox.offsets[j];
                    }
                    mySourceBoxCoords[j] >>= 1;
                    mySourceBox.widths[j] *= 2;
                }
            }

            // Compute the coordinates and center of this source box
            Array<R,d> p0B;
            for( std::size_t j=0; j<d; ++j )
                p0B[j] = mySourceBox.offsets[j] + 0.5*wB[j];

            // Grab the communicator for this step as well as our rank in it
            MPI_Comm clusterComm = plan.GetSubcommunicator( numCommunications );
            int clusterRank;
            MPI_Comm_rank( clusterComm, &clusterRank );

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
                    ReportWithFlush
                    ( rank, "  Parallel source weight recursion..." );
                    general_fio::SourceWeightRecursion<R,d,q>
                    ( context, plan, Phi, log2NumMergingProcesses, clusterRank, 
                      x0A, p0B, wB, parentInteractionOffset,
                      weightGridList, partialWeightGridList[targetIndex] );
                    Report( rank, "done." );
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
                    ReportWithFlush
                    ( rank, "Parallel target weight recursion..." );
                    general_fio::TargetWeightRecursion<R,d,q>
                    ( context, plan, Phi, log2NumMergingProcesses, clusterRank,
                      ARelativeToAp, x0A, x0Ap, p0B, wA, wB,
                      parentInteractionOffset, weightGridList, 
                      partialWeightGridList[targetIndex] );
                    Report( rank, "done." );
                }
            }

            // Scatter the summation of the weights
            std::vector<int> recvCounts( numMergingProcesses );
            for( std::size_t j=0; j<numMergingProcesses; ++j )
                recvCounts[j] = 2*weightGridList.Length()*q_to_d;
            ReportWithFlush( rank, "  SumScatter..." );
            // Currently two types of planned communication are supported, as 
            // they are the only required types for transforming and inverting 
            // the transform:
            //  1) partitions of dimensions 0 -> c
            //  2) partitions of dimensions c -> d-1
            // Both 1 and 2 include partitioning 0 -> d-1, but, in general, 
            // the second category never requires packing.
            const Array<bool,d>& targetDimsToCut = 
                plan.GetTargetDimsToCut( numCommunications );
            bool setFirstPartDim = false;
            bool finalizedLastPartDim = false;
            std::size_t firstPartDim;
            std::size_t lastPartDim = 0;
            for( std::size_t j=0; j<d; ++j )
            {
                if( targetDimsToCut[j] && finalizedLastPartDim )
                    std::runtime_error("Invalid communication in plan.");
                if( !targetDimsToCut[j] && setFirstPartDim )
                {
                    lastPartDim = j-1;
                    finalizedLastPartDim = true;
                }
                if( targetDimsToCut[j] && !setFirstPartDim )
                {
                    firstPartDim = j;
                    setFirstPartDim = true;
                }
            }
            if( !finalizedLastPartDim )
                lastPartDim = d-1;
            if( lastPartDim == d-1 )
            {
                // We must have partition dims of the form c -> d-1
                SumScatter
                ( partialWeightGridList.Buffer(), weightGridList.Buffer(),
                  &recvCounts[0], clusterComm );
            }
            else
            {
                // We must have partition dims of the form 0 -> c < d-1.
                // Thus we must copy 2^{d-log2NumMergingProcesses} chunks for 
                // each of the 2^log2NumMergingProcesses processes.
                const R* partialBuffer = partialWeightGridList.Buffer();
                const std::size_t recvSize = recvCounts[0];
                const std::size_t sendSize = recvSize*numMergingProcesses;
                const std::size_t numChunksPerProcess = 
                    1u<<(d-log2NumMergingProcesses);
                const std::size_t chunkSize = recvSize / numChunksPerProcess;
                std::vector<R> sendBuffer( sendSize );
                for( std::size_t p=0; p<numMergingProcesses; ++p )
                {
                    R* sendOffset = &sendBuffer[p*recvSize];
                    for( std::size_t j=0; j<numChunksPerProcess; ++j )
                    {
                        std::memcpy
                        ( &sendOffset[j*chunkSize], 
                          &partialBuffer[(p+j*numMergingProcesses)*chunkSize],
                          chunkSize*sizeof(R) );
                    }
                }
                SumScatter
                ( &sendBuffer[0], weightGridList.Buffer(), 
                  &recvCounts[0], clusterComm );
            }
            Report( rank, "done." );

            const Array<bool,d>& rightSideOfCut = 
                plan.GetRightSideOfCut( numCommunications );
            for( std::size_t j=0; j<d; ++j )
            {
                if( targetDimsToCut[j] )
                {
                    myTargetBox.widths[j] *= 0.5;
                    myTargetBoxCoords[j] *= 2;
                    if( rightSideOfCut[j] )
                    {
                        myTargetBoxCoords[j] |= 1;
                        myTargetBox.offsets[j] += myTargetBox.widths[j];
                    }
                    --log2LocalTargetBoxesPerDim[j];
                    --log2LocalTargetBoxes;
                }
            }

            ++numCommunications;
        }
        if( level==log2N/2 )
        {
            ReportWithFlush( rank, "  Switching to target interpolation..." );
            general_fio::SwitchToTargetInterp<R,d,q>
            ( context, plan, Amp, Phi, sourceBox, targetBox, mySourceBox, 
              myTargetBox, log2LocalSourceBoxes, log2LocalTargetBoxes,
              log2LocalSourceBoxesPerDim, log2LocalTargetBoxesPerDim,
              weightGridList );
            Report( rank, "done." );
        }
    }

    // Construct the general FIO PotentialField
    ReportWithFlush( rank, "  Constructing PotentialField..." );
    std::auto_ptr< const general_fio::PotentialField<R,d,q> > potentialField( 
        new general_fio::PotentialField<R,d,q>
            ( context, Phi, sourceBox, myTargetBox, log2LocalTargetBoxesPerDim,
              weightGridList )
    );
    Report( rank, "done." );

    return potentialField;
}

} // general_fio

template<typename R,std::size_t d,std::size_t q>
std::auto_ptr< const general_fio::PotentialField<R,d,q> >
GeneralFIO
( const general_fio::Context<R,d,q>& context,
  const Plan<d>& plan,
  const AmplitudeFunctor<R,d>& Amp,
  const PhaseFunctor<R,d>& Phi,
  const Box<R,d>& sourceBox,
  const Box<R,d>& targetBox,
  const std::vector< Source<R,d> >& mySources )
{
    return general_fio::transform
    ( context, plan, Amp, Phi, sourceBox, targetBox, mySources );
}

template<typename R,std::size_t d,std::size_t q>
std::auto_ptr< const general_fio::PotentialField<R,d,q> >
GeneralFIO
( const general_fio::Context<R,d,q>& context,
  const Plan<d>& plan,
  const PhaseFunctor<R,d>& Phi,
  const Box<R,d>& sourceBox,
  const Box<R,d>& targetBox,
  const std::vector< Source<R,d> >& mySources )
{
    return general_fio::transform
    ( context, plan, UnitAmplitude<R,d>(), Phi, sourceBox, targetBox, 
      mySources );
}

} // bfio

#endif // BFIO_GENERAL_FIO_HPP

