/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of DistButterfly and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef DBF_LNUFT_HPP
#define DBF_LNUFT_HPP

#include <memory>

#include "dist-butterfly/structures.hpp"
#include "dist-butterfly/tools.hpp"

#ifdef TIMING
namespace dbf {

using std::array;
using std::memcpy;
using std::size_t;
using std::vector;

namespace lnuft {

static bool alreadyTimed = false;

static Timer timer;
static Timer initializeWeightsTimer;
static Timer M2MTimer;
static Timer L2LTimer;
static Timer M2LTimer;
static Timer sumScatterTimer;

static inline void 
ResetTimers()
{
    timer.Reset();
    initializeWeightsTimer.Reset();
    M2MTimer.Reset();
    M2LTimer.Reset();
    L2LTimer.Reset();
    sumScatterTimer.Reset();
}

static inline void 
PrintTimings()
{
#ifndef RELEASE
    if( !alreadyTimed )
        throw std::logic_error("You have not yet run LNUFT.");
#endif
    std::cout << "LNUFT timings:\n"
              << "------------------------------------------\n" 
              << "InitializeWeights:     " 
              << initializeWeightsTimer.Total() << " seconds.\n"
              << "M2M: " << M2MTimer.Total() << " seconds.\n"
              << "M2L:  " << M2LTimer.Total() << " seconds.\n"
              << "L2L: " << L2LTimer.Total() << " seconds.\n"
              << "SumScatter:            "
              << sumScatterTimer.Total() << " seconds.\n"
              << "Total: " << timer.Total() << " seconds.\n" << std::endl;
}

} // lnuft
} // dbf
#endif

#include "dist-butterfly/lnuft/context.hpp"
#include "dist-butterfly/lnuft/ft_phases.hpp"
#include "dist-butterfly/lnuft/potential_field.hpp"

#include "dist-butterfly/butterfly/initialize_weights.hpp"
#include "dist-butterfly/butterfly/M2M.hpp"
#include "dist-butterfly/butterfly/L2L.hpp"

#include "dist-butterfly/lnuft/M2L.hpp"

namespace dbf {

template<typename R,size_t d,size_t q>
std::unique_ptr<const lnuft::PotentialField<R,d,q>>
LNUFT
( const lnuft::Context<R,d,q>& nuftContext,
  const Plan<d>& plan,
  const Box<R,d>& sBox,
  const Box<R,d>& tBox,
  const vector<Source<R,d>>& mySources )
{
#ifdef TIMING
    lnuft::ResetTimers();
    lnuft::timer.Start();
#endif
    typedef complex<R> C;
    const size_t q_to_d = Pow<q,d>::val;
    const bfly::Context<R,d,q>& bflyContext = nuftContext.GetButterflyContext();

    // We will choose the phase function based on the context's direction.
    // We could potentially have the plan direction be different
    // (e.g., Forward direction with Adjoint FT phase function)
    const Direction direction = nuftContext.GetDirection();
    const lnuft::ForwardFTPhase<R,d> forwardPhase;
    const lnuft::AdjointFTPhase<R,d> adjointPhase;
    const lnuft::FTPhase<R,d>& phase = 
        ( direction==FORWARD ? 
          (const lnuft::FTPhase<R,d>&)forwardPhase :
          (const lnuft::FTPhase<R,d>&)adjointPhase );

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
    size_t log2LocalSBoxes=0,
           log2LocalTBoxes=0;
    array<size_t,d> log2LocalSBoxesPerDim, log2LocalTBoxesPerDim;
    log2LocalTBoxesPerDim.fill(0);
    for( size_t j=0; j<d; ++j )
    {
        log2LocalSBoxesPerDim[j] = log2N-log2SBoxesPerDim[j];
        log2LocalSBoxes += log2LocalSBoxesPerDim[j];
    }

    // Initialize the weights using Lagrangian interpolation on the 
    // smooth component of the kernel.
    WeightGridList<R,d,q> weightGridList( 1<<log2LocalSBoxes );
#ifdef TIMING
    lnuft::initializeWeightsTimer.Start();
#endif
    bfly::InitializeWeights
    ( bflyContext, plan, phase, sBox, tBox, mySBox, 
      log2LocalSBoxes, log2LocalSBoxesPerDim, mySources, weightGridList );
#ifdef TIMING
    lnuft::initializeWeightsTimer.Stop();
#endif

    // Start the main recursion loop
    if( log2N == 0 || log2N == 1 )
    {
#ifdef TIMING
        lnuft::M2LTimer.Start();
#endif
        lnuft::M2L
        ( nuftContext, plan, sBox, tBox, mySBox, myTBox,
          log2LocalSBoxes, log2LocalTBoxes,
          log2LocalSBoxesPerDim, log2LocalTBoxesPerDim, weightGridList );
#ifdef TIMING
        lnuft::M2LTimer.Stop();
#endif
    }
    for( size_t level=1; level<=log2N; ++level )
    {
        // Compute the width of the nodes at this level
        array<R,d> wA, wB;
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
                    x0A[j] = myTBox.offsets[j] + (A[j]+R(1)/R(2))*wA[j];

                // Loop over the B boxes in source domain
                ConstrainedHTreeWalker<d> BWalker( log2LocalSBoxesPerDim );
                for( size_t sIndex=0; 
                     sIndex<(1u<<log2LocalSBoxes); ++sIndex, BWalker.Walk() )
                {
                    const array<size_t,d> B = BWalker.State();

                    // Compute coordinates and center of this source box
                    array<R,d> p0B;
                    for( size_t j=0; j<d; ++j )
                        p0B[j] = mySBox.offsets[j] + (B[j]+R(1)/R(2))*wB[j];

                    // We are storing the interaction pairs source-major
                    const size_t iIndex = sIndex + (tIndex<<log2LocalSBoxes);

                    // Grab the interaction offset for the parent of target box 
                    // i interacting with the children of source box k
                    const size_t parentIOffset = 
                        ((tIndex>>d)<<(log2LocalSBoxes+d)) + (sIndex<<d);

                    if( level <= log2N/2 )
                    {
#ifdef TIMING
                        lnuft::M2MTimer.Start();
#endif
                        bfly::M2M
                        ( bflyContext, plan, phase, level, x0A, p0B, wB,
                          parentIOffset, oldWeightGridList,
                          weightGridList[iIndex] );
#ifdef TIMING
                        lnuft::M2MTimer.Stop();
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
                        lnuft::L2LTimer.Start();
#endif
                        bfly::L2L
                        ( bflyContext, plan, phase, level,
                          ARelativeToAp, x0A, x0Ap, p0B, wA, wB,
                          parentIOffset, oldWeightGridList, 
                          weightGridList[iIndex] );
#ifdef TIMING
                        lnuft::L2LTimer.Stop();
#endif
                    }
                }
            }
        }
        else 
        {
            const size_t log2NumMerging = d-log2LocalSBoxes;

            log2LocalSBoxes = 0; 
            for( size_t j=0; j<d; ++j )
                log2LocalSBoxesPerDim[j] = 0;

            // Fully refine target domain and coarsen source domain.
            // We partition the target domain after the SumScatter.
            const vector<size_t>& sDimsToMerge = 
                plan.GetSourceDimsToMerge( level );
            for( size_t i=0; i<log2NumMerging; ++i )
            {
                const size_t j = sDimsToMerge[i];
                if( mySBoxCoords[j] & 1 )
                    mySBox.offsets[j] -= mySBox.widths[j];
                mySBoxCoords[j] /= 2;
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
            ConstrainedHTreeWalker<d> AWalker( log2LocalTBoxesPerDim );
            WeightGridList<R,d,q> partialWeightGridList
            ( 1<<log2LocalTBoxes );
            for( size_t tIndex=0; 
                 tIndex<(1u<<log2LocalTBoxes); ++tIndex, AWalker.Walk() )
            {
                const array<size_t,d> A = AWalker.State();

                // Compute coordinates and center of this target box
                array<R,d> x0A;
                for( size_t j=0; j<d; ++j )
                    x0A[j] = myTBox.offsets[j] + (A[j]+R(1)/R(2))*wA[j];

                // Compute the interaction offset of A's parent interacting 
                // with the remaining local source boxes
                const size_t parentIOffset = ((tIndex>>d)<<(d-log2NumMerging));
                if( level <= log2N/2 )
                {
#ifdef TIMING
                    lnuft::M2MTimer.Start();
#endif
                    bfly::M2M
                    ( bflyContext, plan, phase, level, x0A, p0B, wB,
                      parentIOffset, weightGridList,
                      partialWeightGridList[tIndex] );
#ifdef TIMING
                    lnuft::M2MTimer.Stop();
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
                    lnuft::L2LTimer.Start();
#endif
                    bfly::L2L
                    ( bflyContext, plan, phase, level,
                      ARelativeToAp, x0A, x0Ap, p0B, wA, wB,
                      parentIOffset, weightGridList, 
                      partialWeightGridList[tIndex] );
#ifdef TIMING
                    lnuft::L2LTimer.Stop();
#endif
                }
            }

            // Scatter the summation of the weights
#ifdef TIMING
            lnuft::sumScatterTimer.Start();
#endif
            const size_t recvSize = 2*weightGridList.Length()*q_to_d;
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
                  recvSize, clusterComm );
            }
            else
            {
                const size_t log2NumSubclusters = 
                    log2NumMerging-log2SubclusterSize;
                const size_t numSubclusters = 1u<<log2NumSubclusters;
                const size_t subclusterSize = 1u<<log2SubclusterSize;

                const size_t numChunksPerProcess = subclusterSize;
                const size_t chunkSize = recvSize / numChunksPerProcess;
                const R* partialBuffer = partialWeightGridList.Buffer();
                vector<R> sendBuffer( recvSize<<log2NumMerging );
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
                  recvSize, clusterComm );
            }
#ifdef TIMING
            lnuft::sumScatterTimer.Stop();
#endif

            const vector<size_t>& tDimsToCut = plan.GetTargetDimsToCut( level );
            const vector<bool>& rightSideOfCut=plan.GetRightSideOfCut( level );
            for( size_t i=0; i<log2NumMerging; ++i )
            {
                const size_t j = tDimsToCut[i];
                myTBox.widths[j] /= 2;
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
            lnuft::M2LTimer.Start();
#endif
            lnuft::M2L
            ( nuftContext, plan, sBox, tBox, mySBox, myTBox,
              log2LocalSBoxes, log2LocalTBoxes,
              log2LocalSBoxesPerDim, log2LocalTBoxesPerDim, weightGridList );
#ifdef TIMING
            lnuft::M2LTimer.Stop();
#endif
        }
    }

    // Construct the PotentialField
    std::unique_ptr<const lnuft::PotentialField<R,d,q>> 
    potentialField( 
        new lnuft::PotentialField<R,d,q>
            ( nuftContext, sBox, myTBox, myTBoxCoords,
              log2LocalTBoxesPerDim, weightGridList )
    );

#ifdef TIMING
    lnuft::timer.Stop();
    lnuft::alreadyTimed = true;
#endif
    return potentialField;
}

} // dbf

#endif // ifndef DBF_LNUFT_HPP
