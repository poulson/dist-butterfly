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
static bfio::Timer sWeightRecursionTimer;
static bfio::Timer sWeightRecursionTimer1;
static bfio::Timer sWeightRecursionTimer1Phase;
static bfio::Timer sWeightRecursionTimer1SinCos;
static bfio::Timer sWeightRecursionTimer2;
static bfio::Timer sWeightRecursionTimer3;

static bfio::Timer tWeightRecursionTimer;
static bfio::Timer tWeightRecursionTimer1;
static bfio::Timer tWeightRecursionTimer1Phase;
static bfio::Timer tWeightRecursionTimer1SinCos;
static bfio::Timer tWeightRecursionTimer2;
static bfio::Timer tWeightRecursionTimer3;

static bfio::Timer switchToTargetInterpTimer;
static bfio::Timer sumScatterTimer;

static inline void
ResetTimers()
{
    timer.Reset();
    initializeWeightsTimer.Reset();
    sWeightRecursionTimer.Reset();
    sWeightRecursionTimer1.Reset();
    sWeightRecursionTimer1Phase.Reset();
    sWeightRecursionTimer1SinCos.Reset();
    sWeightRecursionTimer2.Reset();
    sWeightRecursionTimer3.Reset();
    tWeightRecursionTimer.Reset();
    tWeightRecursionTimer1.Reset();
    tWeightRecursionTimer1Phase.Reset();
    tWeightRecursionTimer1SinCos.Reset();
    tWeightRecursionTimer2.Reset();
    tWeightRecursionTimer3.Reset();
    switchToTargetInterpTimer.Reset();
    sumScatterTimer.Reset();
}

static inline void
PrintTimings()
{
#ifndef RELEASE
    if( !alreadyTimed )
	throw std::logic_error("You have not yet run RFIO.");
#endif
    std::cout << "RFIO timings:\n"
	      << "--------------------------------------------\n"
              << "InitializeWeights:     "
              << initializeWeightsTimer.Total() << " seconds.\n"
              << "SourceWeightRecursion: "
              << sWeightRecursionTimer.Total() << " seconds.\n"
              << "  Stage 1: "
              << sWeightRecursionTimer1.Total() << " seconds.\n"
              << "    Phase: "
              << sWeightRecursionTimer1Phase.Total() << " seconds.\n"
              << "    SinCos: "
              << sWeightRecursionTimer1SinCos.Total() << " seconds.\n"
              << "  Stage 2: "
              << sWeightRecursionTimer2.Total() << " seconds.\n"
              << "  Stage 3: "
              << sWeightRecursionTimer3.Total() << " seconds.\n"
              << "SwitchToTargetInterp:  "
              << switchToTargetInterpTimer.Total() << " seconds.\n"
              << "TargetWeightRecursion: "
              << tWeightRecursionTimer.Total() << " seconds.\n"
              << "  Stage 1: "
              << tWeightRecursionTimer1.Total() << " seconds.\n"
              << "    Phase: "
              << tWeightRecursionTimer1Phase.Total() << " seconds.\n"
              << "    SinCos: "
              << tWeightRecursionTimer1SinCos.Total() << " seconds.\n"
              << "  Stage 2: "
              << tWeightRecursionTimer2.Total() << " seconds.\n"
              << "  Stage 3: "
              << tWeightRecursionTimer3.Total() << " seconds.\n"
              << "SumScatter:            "
              << sumScatterTimer.Total() << " seconds.\n"
              << "Total: " << timer.Total() << " seconds.\n" << std::endl;
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
  const Box<R,d>& sBox,
  const Box<R,d>& tBox,
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
    const size_t bootstrap = plan.GetBootstrapSkip();

    // Compute the number of source and target boxes that our process is 
    // responsible for initializing weights in
    size_t log2WeightGridSize = 0;
    size_t log2LocalSBoxes = 0;
    size_t log2LocalTBoxes = 0;
    array<size_t,d> log2LocalSBoxesPerDim, log2LocalTBoxesPerDim;
    log2LocalTBoxesPerDim.fill(0);
    for( size_t j=0; j<d; ++j )
    {
        if( log2N-log2SBoxesPerDim[j] >= bootstrap )
            log2LocalSBoxesPerDim[j]= (log2N-log2SBoxesPerDim[j]) - bootstrap;
        else
            log2LocalSBoxesPerDim[j] = 0;
        log2LocalTBoxesPerDim[j] = bootstrap;
        log2LocalSBoxes += log2LocalSBoxesPerDim[j];
        log2LocalTBoxes += log2LocalTBoxesPerDim[j];
        log2WeightGridSize += log2N-log2SBoxesPerDim[j];
    }

    // Initialize the weights using Lagrangian interpolation on the 
    // smooth component of the kernel.
    WeightGridList<R,d,q> weightGridList( 1u<<log2WeightGridSize );
#ifdef TIMING
    rfio::initializeWeightsTimer.Start();
#endif
    rfio::InitializeWeights
    ( context, plan, phase, sBox, tBox, mySBox, 
      log2LocalSBoxes, log2LocalSBoxesPerDim, mySources, weightGridList );
#ifdef TIMING
    rfio::initializeWeightsTimer.Stop();
#endif

    // Now cut the target domain if necessary
    for( size_t j=0; j<d; ++j )
    {
        if( log2LocalSBoxesPerDim[j] == 0 )
        {
            log2LocalTBoxesPerDim[j] -= bootstrap - (log2N-log2SBoxesPerDim[j]);
            log2LocalTBoxes -= bootstrap - (log2N-log2SBoxesPerDim[j]);
        }
    }

    // Start the main recursion loop
    if( bootstrap == log2N/2 )
    {
#ifdef TIMING
	rfio::switchToTargetInterpTimer.Start();
#endif
        rfio::SwitchToTargetInterp
        ( context, plan, amplitude, phase, sBox, tBox, mySBox, myTBox, 
          log2LocalSBoxes, log2LocalTBoxes,
          log2LocalSBoxesPerDim, log2LocalTBoxesPerDim, weightGridList );
#ifdef TIMING
	rfio::switchToTargetInterpTimer.Stop();
#endif
    }
    for( size_t level=bootstrap+1; level<=log2N; ++level )
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

                    if( level <= log2N/2 )
                    {
#ifdef TIMING
			rfio::sWeightRecursionTimer.Start();
#endif
                        rfio::SourceWeightRecursion
                        ( context, plan, phase, level, x0A, p0B, wB, 
                          parentIOffset, oldWeightGridList,
                          weightGridList[iIndex] );
#ifdef TIMING
			rfio::sWeightRecursionTimer.Stop();
#endif
                    }
                    else
                    {
                        array<R,d> x0Ap;
                        array<size_t,d> globalA;
                        size_t ARelativeToAp = 0;
                        for( size_t j=0; j<d; ++j )
                        {
                            globalA[j] = A[j]+
                                (myTBoxCoords[j]<<log2LocalTBoxesPerDim[j]);
                            x0Ap[j] = tBox.offsets[j] + (globalA[j]|1)*wA[j];
                            ARelativeToAp |= (globalA[j]&1)<<j;
                        }
#ifdef TIMING
			rfio::tWeightRecursionTimer.Start();
#endif
                        rfio::TargetWeightRecursion
                        ( context, plan, phase, level,
                          ARelativeToAp, x0A, x0Ap, p0B, wA, wB,
                          parentIOffset, oldWeightGridList, 
                          weightGridList[iIndex] );
#ifdef TIMING
			rfio::tWeightRecursionTimer.Stop();
#endif
                    }
                }
            }
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
                p0B[j] = mySBox.offsets[j] + 0.5*wB[j];

            // Form the partial weights by looping over the boxes in the  
            // target domain.
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

                // Compute the interaction offset of A's parent interacting 
                // with the remaining local source boxes
                const size_t parentIOffset = 
                    ((tIndex>>d)<<(d-log2NumMergingProcesses));
                if( level <= log2N/2 )
                {
#ifdef TIMING
		    rfio::sWeightRecursionTimer.Start();
#endif
                    rfio::SourceWeightRecursion
                    ( context, plan, phase, level, x0A, p0B, wB,
                      parentIOffset, weightGridList,
                      partialWeightGridList[tIndex] );
#ifdef TIMING
		    rfio::sWeightRecursionTimer.Stop();
#endif
                }
                else
                {
                    array<R,d> x0Ap;
                    array<size_t,d> globalA;
                    size_t ARelativeToAp = 0;
                    for( size_t j=0; j<d; ++j )
                    {
                        globalA[j] = A[j] +
                            (myTBoxCoords[j]<<log2LocalTBoxesPerDim[j]);
                        x0Ap[j] = tBox.offsets[j] + (globalA[j]|1)*wA[j];
                        ARelativeToAp |= (globalA[j]&1)<<j;
                    }
#ifdef TIMING
		    rfio::tWeightRecursionTimer.Start();
#endif
                    rfio::TargetWeightRecursion
                    ( context, plan, phase, level,
                      ARelativeToAp, x0A, x0Ap, p0B, wA, wB,
                      parentIOffset, weightGridList, 
                      partialWeightGridList[tIndex] );
#ifdef TIMING
		    rfio::tWeightRecursionTimer.Stop();
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

            const vector<size_t>& tDimsToCut = plan.GetTargetDimsToCut( level );
            const vector<bool>& rightSideOfCut=plan.GetRightSideOfCut( level );
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
        }
        if( level==log2N/2 )
        {
#ifdef TIMING
	    rfio::switchToTargetInterpTimer.Start();
#endif
            rfio::SwitchToTargetInterp
            ( context, plan, amplitude, phase, sBox, tBox, mySBox, myTBox,
              log2LocalSBoxes, log2LocalTBoxes, 
              log2LocalSBoxesPerDim, log2LocalTBoxesPerDim, weightGridList );
#ifdef TIMING
	    rfio::switchToTargetInterpTimer.Stop();
#endif
        }
    }

    // Construct the FIO PotentialField
    std::unique_ptr<const rfio::PotentialField<R,d,q>> 
    potentialField( 
        new rfio::PotentialField<R,d,q>
        ( context, amplitude, phase, sBox, myTBox, myTBoxCoords, 
          log2LocalTBoxesPerDim, weightGridList )
    );

#ifdef TIMING
    rfio::timer.Stop();
    rfio::alreadyTimed = true;
#endif
    return potentialField;
}

} // rfio

template<typename R,size_t d,size_t q>
std::unique_ptr<const rfio::PotentialField<R,d,q>>
RFIO
( const rfio::Context<R,d,q>& context,
  const Plan<d>& plan,
  const Amplitude<R,d>& amplitude,
  const Phase<R,d>& phase,
  const Box<R,d>& sBox,
  const Box<R,d>& tBox,
  const vector<Source<R,d>>& mySources )
{
    return rfio::transform
    ( context, plan, amplitude, phase, sBox, tBox, mySources );
}

template<typename R,size_t d,size_t q>
std::unique_ptr<const rfio::PotentialField<R,d,q>>
RFIO
( const rfio::Context<R,d,q>& context,
  const Plan<d>& plan,
  const Phase<R,d>& phase,
  const Box<R,d>& sBox,
  const Box<R,d>& tBox,
  const vector<Source<R,d>>& mySources )
{
    UnitAmplitude<R,d> unitAmp;
    auto u = rfio::transform
    ( context, plan, unitAmp, phase, sBox, tBox, mySources );
    return u;
}

} // bfio

#endif // ifndef BFIO_RFIO_HPP
