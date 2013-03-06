/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of DistButterfly and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef DBF_BFLY_INITIALIZE_WEIGHTS_HPP
#define DBF_BFLY_INITIALIZE_WEIGHTS_HPP

#include "dist-butterfly/constants.hpp"

#include "dist-butterfly/structures/box.hpp"
#include "dist-butterfly/structures/constrained_htree_walker.hpp"
#include "dist-butterfly/structures/plan.hpp"
#include "dist-butterfly/structures/weight_grid_list.hpp"

#include "dist-butterfly/tools/flatten_constrained_htree_index.hpp"
#include "dist-butterfly/tools/mpi.hpp"
#include "dist-butterfly/tools/special_functions.hpp"

#include "dist-butterfly/functors/phase.hpp"

#include "dist-butterfly/butterfly/context.hpp"

namespace dbf {

using std::array;
using std::memset;
using std::size_t;
using std::vector;

namespace bfly {

template<typename R,size_t d,size_t q>
inline void
InitializeWeights
( const Context<R,d,q>& context,
  const Plan<d>& plan,
  const Phase<R,d>& phase,
  const Box<R,d>& sBox,
  const Box<R,d>& tBox,
  const Box<R,d>& mySBox,
  const size_t log2LocalSBoxes,
  const array<size_t,d>& log2LocalSBoxesPerDim,
  const vector<Source<R,d>>& mySources,
        WeightGridList<R,d,q>& weightGridList )
{
    const size_t N = plan.GetN();
    const size_t q_to_d = Pow<q,d>::val;

#ifdef TIMING
    Timer computeTimer;
    Timer setToPotentialTimer;
    Timer preprocessTimer;
    Timer lagrangeTimer;
    Timer axpyTimer;
#endif // TIMING

    MPI_Comm comm = plan.GetComm();
    int rank;
    MPI_Comm_rank( comm, &rank );

    const size_t bootstrap = plan.GetBootstrapSkip();
    MPI_Comm bootstrapComm = plan.GetBootstrapClusterComm();
    int numMergingProcesses;
    MPI_Comm_size( bootstrapComm, &numMergingProcesses );

    if( numMergingProcesses == 1 )
    {
        // Compute the source box widths
        array<R,d> wB;
        for( size_t j=0; j<d; ++j )
            wB[j] = sBox.widths[j] / (N>>bootstrap);

        // Compute the target box widths
        array<R,d> wA;
        for( size_t j=0; j<d; ++j )
            wA[j] = tBox.widths[j] / (1u<<bootstrap);

        // Compute the unscaled weights for each local box by looping over 
        // our sources and sorting them into the appropriate local box one 
        // at a time. We throw an error if a source is outside of our source
        // box.
        vector<R> phiResults, sinResults, cosResults;
        const size_t numSources = mySources.size();
        vector<array<R,d>> pPoints( numSources ), pRefPoints( numSources );
        vector<size_t> flattenedSBoxIndices( numSources );
        for( size_t s=0; s<numSources; ++s )
        {
            const array<R,d>& p = mySources[s].p;
            pPoints[s] = p;

            // Determine which local box we're in (if any)
            array<size_t,d> B;
            for( size_t j=0; j<d; ++j )
            {
                R leftBound = mySBox.offsets[j];
                R rightBound = leftBound + mySBox.widths[j];
                if( p[j] < leftBound || p[j] >= rightBound )
                {
                    std::ostringstream msg;
                    msg << "Source " << s << " was at " << p[j]
                        << " in dimension " << j 
                        << ", but our source box in this "
                        << "dim. is [" << leftBound << "," << rightBound 
                        << ").";
                    throw std::runtime_error( msg.str() );
                }

                // We must be in the box, so bitwise determine the coord. index
                B[j] = 0;
                for( size_t k=log2LocalSBoxesPerDim[j]; k>0; --k )
                {
                    const R middle = (rightBound+leftBound)/2.;
                    if( p[j] < middle )
                    {
                        // implicitly setting bit k-1 of B[j] to 0
                        rightBound = middle;
                    }
                    else
                    {
                        B[j] |= (1<<(k-1));
                        leftBound = middle;
                    }
                }
            }

            // Translate the local integer coordinates into the source center.
            array<R,d> p0;
            for( size_t j=0; j<d; ++j )
                p0[j] = mySBox.offsets[j] + (B[j]+R(1)/R(2))*wB[j];

            // In order to add this point's contribution to the unscaled weights
            // of B we will evaluate the Lagrangian polynomial on the reference 
            // grid, so we need to map p to it first.
            for( size_t j=0; j<d; ++j )
                pRefPoints[s][j] = (p[j]-p0[j])/wB[j];
    
            // Flatten the integer coordinates of B
            flattenedSBoxIndices[s] = 
                FlattenConstrainedHTreeIndex( B, log2LocalSBoxesPerDim );
        }

        // Set all of the weights to zero
        memset
        ( weightGridList.Buffer(), 0, 
          weightGridList.Length()*2*q_to_d*sizeof(R) );

#ifdef TIMING
        computeTimer.Start();
#endif // TIMING
        // Set all of the weights to the potentials in the target boxes. 
        // We take care to avoid redundant Lagrangian interpolation; it was 
        // previously the bottleneck.
#ifdef TIMING
        setToPotentialTimer.Start();
#endif // TIMING
        for( size_t t=0; t<q_to_d; ++t )
        {
            vector<R> lagrangeResults;
#ifdef TIMING
            lagrangeTimer.Start();
#endif // TIMING
            context.LagrangeBatch( t, pRefPoints, lagrangeResults );
#ifdef TIMING
            lagrangeTimer.Stop();
#endif // TIMING

            HTreeWalker<d> AWalker;
            for( size_t tIndex=0;
                 tIndex<(1u<<(d*bootstrap)); ++tIndex, AWalker.Walk() )
            {
                const array<size_t,d> A = AWalker.State();

                // Compute the center of the target box
                array<R,d> x0A;
                for( size_t j=0; j<d; ++j )
                    x0A[j] = tBox.offsets[j] + (A[j]+R(1)/R(2))*wA[j];

                const vector<array<R,d>> xPoint( 1, x0A );

#ifdef TIMING
                preprocessTimer.Start();
#endif // TIMING
                phase.BatchEvaluate( xPoint, pPoints, phiResults );
                SinCosBatch( phiResults, sinResults, cosResults );
#ifdef TIMING
                preprocessTimer.Stop();
#endif // TIMING

                {
                    vector<R> realBeta( numSources ), imagBeta( numSources );
                    R* RESTRICT realBetaBuffer = &realBeta[0];
                    R* RESTRICT imagBetaBuffer = &imagBeta[0];
                    const R* RESTRICT cosBuffer = &cosResults[0];
                    const R* RESTRICT sinBuffer = &sinResults[0];
                    for( size_t s=0; s<numSources; ++s )
                    {
                        const R realPhase = cosBuffer[s];
                        const R imagPhase = sinBuffer[s];
                        const R realMagnitude = real(mySources[s].magnitude);
                        const R imagMagnitude = imag(mySources[s].magnitude);
                        realBetaBuffer[s] = 
                            realPhase*realMagnitude-imagPhase*imagMagnitude;
                        imagBetaBuffer[s] = 
                            imagPhase*realMagnitude+realPhase*imagMagnitude;
                    }

#ifdef TIMING
                    axpyTimer.Start();
#endif // TIMING
                    const R* RESTRICT lagrangeBuffer = &lagrangeResults[0];
                    for( size_t s=0; s<numSources; ++s )
                    {
                        const size_t sIndex = flattenedSBoxIndices[s];
                        const size_t iIndex = 
                            sIndex + (tIndex<<log2LocalSBoxes);
                        weightGridList[iIndex].RealWeight(t) += 
                            realBetaBuffer[s]*lagrangeBuffer[s];
                        weightGridList[iIndex].ImagWeight(t) +=
                            imagBetaBuffer[s]*lagrangeBuffer[s];
                    }
#ifdef TIMING
                    axpyTimer.Stop();
#endif // TIMING
                }
            }
        }
#ifdef TIMING
        setToPotentialTimer.Stop();
#endif // TIMING

        HTreeWalker<d> AWalker;
        for( size_t tIndex=0;
             tIndex<(1u<<(d*bootstrap)); ++tIndex, AWalker.Walk() )
        {
            const array<size_t,d> A = AWalker.State();

            // Compute the center of the target box
            array<R,d> x0A;
            for( size_t j=0; j<d; ++j )
                x0A[j] = tBox.offsets[j] + (A[j]+R(1)/R(2))*wA[j];

            const vector<array<R,d>> xPoint( 1, x0A );

            // Loop over all of the boxes to compute the {p_t^B} and prefactors
            // for each delta weight {delta_t^AB}
            vector<array<R,d>> chebyshevPoints( q_to_d );
            const vector<array<R,d>>& chebyshevGrid = 
                context.GetChebyshevGrid();
            ConstrainedHTreeWalker<d> BWalker( log2LocalSBoxesPerDim );
            for( size_t sIndex=0; 
                 sIndex<(1u<<log2LocalSBoxes); ++sIndex, BWalker.Walk() ) 
            {
                const array<size_t,d> B = BWalker.State();

                // Translate the local coordinates into the source center 
                array<R,d> p0;
                for( size_t j=0; j<d; ++j )
                    p0[j] = mySBox.offsets[j] + (B[j]+R(1)/R(2))*wB[j];

                const size_t iIndex = sIndex + (tIndex<<log2LocalSBoxes);
                WeightGrid<R,d,q>& weightGrid = weightGridList[iIndex];
    
                // Compute the prefactors given this p0 and multiply it by 
                // the corresponding weights
                {
                    R* RESTRICT chebyshevPointsBuffer = &chebyshevPoints[0][0];
                    const R* RESTRICT p0Buffer = &p0[0];
                    const R* RESTRICT wBBuffer = &wB[0];
                    const R* RESTRICT chebyshevBuffer = &chebyshevGrid[0][0];
                    for( size_t t=0; t<q_to_d; ++t )
                        for( size_t j=0; j<d; ++j )
                            chebyshevPointsBuffer[t*d+j] = 
                                p0Buffer[j] + 
                                wBBuffer[j]*chebyshevBuffer[t*d+j];
                }
                phase.BatchEvaluate( xPoint, chebyshevPoints, phiResults );
                SinCosBatch( phiResults, sinResults, cosResults );
                {
                    R* RESTRICT realBuffer = weightGrid.RealBuffer();
                    R* RESTRICT imagBuffer = weightGrid.ImagBuffer();
                    const R* RESTRICT cosBuffer = &cosResults[0];
                    const R* RESTRICT sinBuffer = &sinResults[0];
                    for( size_t t=0; t<q_to_d; ++t )
                    {
                        const R realPhase = cosBuffer[t];
                        const R imagPhase = -sinBuffer[t];
                        const R realWeight = realBuffer[t];
                        const R imagWeight = imagBuffer[t];
                        realBuffer[t] = 
                            realPhase*realWeight - imagPhase*imagWeight;
                        imagBuffer[t] = 
                            imagPhase*realWeight + realPhase*imagWeight;
                    }
                }
            }
        }
#ifdef TIMING
        computeTimer.Stop();
#endif // TIMING
    }
    else
    {
        throw std::runtime_error("Parallel bootstrapping not yet supported.");
    }
}

} // bfly
} // dbf

#endif // ifndef DBF_BFLY_INITIALIZE_WEIGHTS_HPP 
