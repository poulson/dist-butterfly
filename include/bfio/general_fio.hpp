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
#ifdef TRACE
    if( rank == 0 )
    {
        std::cout << "  Initializing weights...";
        std::cout.flush();
    }
#endif
    WeightGridList<R,d,q> weightGridList( 1<<log2LocalSourceBoxes );
    general_fio::InitializeWeights<R,d,q>
    ( context, plan, Phi, sourceBox, targetBox, mySourceBox, 
      log2LocalSourceBoxes, log2LocalSourceBoxesPerDim, mySources, 
      weightGridList );
#ifdef TRACE
    if( rank == 0 )
        std::cout << "done." << std::endl;
#endif

    // Start the main recursion loop
    if( log2N == 0 || log2N == 1 )
    {
#ifdef TRACE
        if( rank == 0 )
        {
            std::cout << "  Switching to target interpolation...";
            std::cout.flush();
        }
#endif
        general_fio::SwitchToTargetInterp<R,d,q>
        ( context, plan, Amp, Phi, sourceBox, targetBox, mySourceBox, 
          myTargetBox, log2LocalSourceBoxes, log2LocalTargetBoxes,
          log2LocalSourceBoxesPerDim, log2LocalTargetBoxesPerDim,
          weightGridList );
#ifdef TRACE
        if( rank == 0 )
            std::cout << "done." << std::endl;
#endif
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
            for( std::size_t i=0; 
                 i<(1u<<log2LocalTargetBoxes); 
                 ++i, AWalker.Walk() )
            {
                const Array<std::size_t,d> A = AWalker.State();

                // Compute coordinates and center of this target box
                Array<R,d> x0A;
                for( std::size_t j=0; j<d; ++j )
                    x0A[j] = myTargetBox.offsets[j] + (A[j]+0.5)*wA[j];

                // Loop over the B boxes in source domain
                ConstrainedHTreeWalker<d> BWalker( log2LocalSourceBoxesPerDim );
                for( std::size_t k=0; 
                     k<(1u<<log2LocalSourceBoxes); 
                     ++k, BWalker.Walk() )
                {
                    const Array<std::size_t,d> B = BWalker.State();

                    // Compute coordinates and center of this source box
                    Array<R,d> p0B;
                    for( std::size_t j=0; j<d; ++j )
                        p0B[j] = mySourceBox.offsets[j] + (B[j]+0.5)*wB[j];

                    const std::size_t key = k + (i<<log2LocalSourceBoxes);
                    const std::size_t parentOffset = 
                        ((i>>d)<<(log2LocalSourceBoxes+d)) + (k<<d);
                    if( level <= log2N/2 )
                    {
#ifdef TRACE
                        if( rank == 0 )
                        {
                            std::cout << "  Source weight recursion...";
                            std::cout.flush();
                        }
#endif
                        general_fio::SourceWeightRecursion<R,d,q>
                        ( context, Phi, 0, 0, x0A, p0B, wB, parentOffset,
                          oldWeightGridList, weightGridList[key] );
#ifdef TRACE
                        if( rank == 0 )
                            std::cout << "done." << std::endl;
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
#ifdef TRACE
                        if( rank == 0 )
                        {
                            std::cout << "  Target weight recursion...";
                            std::cout.flush();
                        }
#endif
                        general_fio::TargetWeightRecursion<R,d,q>
                        ( context, plan, Phi, 0, 0, 
                          ARelativeToAp, x0A, x0Ap, p0B, wA, wB,
                          parentOffset, oldWeightGridList, 
                          weightGridList[key] );
#ifdef TRACE
                        if( rank == 0 )
                            std::cout << "done." << std::endl;
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
            for( std::size_t i=0; 
                 i<(1u<<log2LocalTargetBoxes); 
                 ++i, AWalker.Walk() )
            {
                const Array<std::size_t,d> A = AWalker.State();

                // Compute coordinates and center of this target box
                Array<R,d> x0A;
                for( std::size_t j=0; j<d; ++j )
                    x0A[j] = myTargetBox.offsets[j] + (A[j]+0.5)*wA[j];

                const std::size_t parentOffset = 
                    ((i>>d)<<(d-log2NumMergingProcesses));
                if( level <= log2N/2 )
                {
#ifdef TRACE
                    if( rank == 0 )
                    {
                        std::cout << "  Parallel source weight recursion...";
                        std::cout.flush();
                    }
#endif
                    general_fio::SourceWeightRecursion<R,d,q>
                    ( context, Phi, log2NumMergingProcesses, clusterRank, 
                      x0A, p0B, wB, parentOffset,
                      weightGridList, partialWeightGridList[i] );
#ifdef TRACE
                    if( rank == 0 )
                        std::cout << "done." << std::endl;
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
#ifdef TRACE
                    if( rank == 0 )
                    {
                        std::cout << "  Parallel target weight recursion...";
                        std::cout.flush();
                    }
#endif
                    general_fio::TargetWeightRecursion<R,d,q>
                    ( context, plan, Phi, log2NumMergingProcesses, clusterRank,
                      ARelativeToAp, x0A, x0Ap, p0B, wA, wB,
                      parentOffset, weightGridList, partialWeightGridList[i] );
#ifdef TRACE
                    if( rank == 0 )
                        std::cout << "done." << std::endl;
#endif
                }
            }

            // TODO: For the SpatialToFreq plan, we might have to pack for one
            //       of the iterations.
            //
            // Scatter the summation of the weights
            std::vector<int> recvCounts( numMergingProcesses );
            for( std::size_t j=0; j<numMergingProcesses; ++j )
                recvCounts[j] = 2*weightGridList.Length()*q_to_d;
#ifdef TRACE
            if( rank == 0 )
            {
                std::cout << "  SumScatter...";
                std::cout.flush();
            }
#endif
            SumScatter
            ( partialWeightGridList.Buffer(), weightGridList.Buffer(),
              &recvCounts[0], clusterComm );
#ifdef TRACE
            if( rank == 0 )
                std::cout << "done." << std::endl;
#endif

            const Array<bool,d>& targetDimsToCut = 
                plan.GetTargetDimsToCut( numCommunications );
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
#ifdef TRACE
            if( rank == 0 )
            {
                std::cout << "  Switching to target interpolation...";
                std::cout.flush();
            }
#endif
            general_fio::SwitchToTargetInterp<R,d,q>
            ( context, plan, Amp, Phi, sourceBox, targetBox, mySourceBox, 
              myTargetBox, log2LocalSourceBoxes, log2LocalTargetBoxes,
              log2LocalSourceBoxesPerDim, log2LocalTargetBoxesPerDim,
              weightGridList );
#ifdef TRACE
            if( rank == 0 )
                std::cout << "done." << std::endl;
#endif
        }
    }

    // Construct the general FIO PotentialField
#ifdef TRACE
    if( rank == 0 )
    {
        std::cout << "  Constructing PotentialField...";
        std::cout.flush();
    }
#endif
    std::auto_ptr< const general_fio::PotentialField<R,d,q> > potentialField( 
        new general_fio::PotentialField<R,d,q>
            ( context, Phi, sourceBox, myTargetBox, log2LocalTargetBoxesPerDim,
              weightGridList )
    );
#ifdef TRACE
    if( rank == 0 )
        std::cout << "done." << std::endl;
#endif

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

