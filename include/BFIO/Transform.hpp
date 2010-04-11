/*
   Copyright 2010 Jack Poulson

   This file is part of ButterflyFIO.

   This program is free software: you can redistribute it and/or modify it under
   the terms of the GNU Lesser General Public License as published by the
   Free Software Foundation; either version 3 of the License, or 
   (at your option) any later version.

   This program is distributed in the hope that it will be useful, but 
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with this program. If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef BFIO_TRANSFORM_HPP
#define BFIO_TRANSFORM_HPP 1

#include <bitset>

#include "BFIO/Util.hpp"
#include "BFIO/Template.hpp"

namespace BFIO
{
    // Applies the butterfly algorithm for the Fourier integral operator 
    // defined by Psi. This allows one to call the function
    // with their own functor, Psi, with potentially no performance penalty. 
    // R is the datatype for representing a Real and d is the spatial and 
    // frequency dimension. q is the number of points in each dimension of the 
    // Chebyshev tensor-product grid (q^d points total).
    template<typename Psi,typename R,unsigned d,unsigned q>
    void
    Transform
    ( const unsigned N, 
      const std::vector< Source<R,d> >& mySources,
            std::vector< LRP<Psi,R,d,q> >& myLRPs,
            MPI_Comm comm                         )
    {
        using namespace std;
        typedef complex<R> C;

        int rank, S;
        MPI_Comm_rank( comm, &rank );
        MPI_Comm_size( comm, &S    ); 
        bitset<sizeof(int)*8> rankBits(rank); 

        // Assert that N and size are powers of 2
        if( ! IsPowerOfTwo(N) )
            throw "Must use a power of 2 problem size.";
        if( ! IsPowerOfTwo(S) ) 
            throw "Must use a power of 2 number of processes.";
        const unsigned L = Log2( N );
        const unsigned s = Log2( S );
        if( s > d*L )
            throw "Cannot use more than N^d processes.";

        // Determine the number of boxes in each dimension of the frequency
        // domain by applying the partitions cyclically over the d dimensions.
        // We can simultaneously compute the indices of our box.
        Array<unsigned,d> myFreqBox;
        Array<unsigned,d> mySpatialBox;
        Array<unsigned,d> log2FreqBoxesPerDim;
        Array<unsigned,d> log2SpatialBoxesPerDim;
        for( unsigned j=0; j<d; ++j )
        {
            myFreqBox[j] = 0;
            mySpatialBox[j] = 0;
            log2FreqBoxesPerDim[j] = 0;
            log2SpatialBoxesPerDim[j] = 0;
        }
        for( unsigned j=s; j>0; --j )
        {
            static unsigned nextDim = 0;
            // Double our current coordinate in the 'nextDim' dimension 
            // and then choose the left/right position based on the (j-1)'th
            // bit of our rank
            myFreqBox[nextDim] = (myFreqBox[nextDim]<<1)+rankBits[j-1];

            log2FreqBoxesPerDim[nextDim]++;
            nextDim = (nextDim+1) % d;
        }

        // Initialize the widths of the boxes in the spatial and frequency 
        // domains that our process is responsible for
        Array<R,d> myFreqBoxWidths;
        Array<R,d> mySpatialBoxWidths;
        for( unsigned j=0; j<d; ++j )
        {
            myFreqBoxWidths[j] = static_cast<R>(1) / 
                                 static_cast<R>(1<<log2FreqBoxesPerDim[j]);
            mySpatialBoxWidths[j] = static_cast<R>(1);
        }
        
        // Compute the offsets for our frequency and spatial box
        Array<R,d> myFreqBoxOffsets;
        Array<R,d> mySpatialBoxOffsets;
        for( unsigned j=0; j<d; ++j )
        {
            myFreqBoxOffsets[j] = myFreqBox[j]*myFreqBoxWidths[j];
            mySpatialBoxOffsets[j] = mySpatialBox[j]*mySpatialBoxWidths[j];
        }

        // Compute the number of 1/N width boxes in the frequency domain that 
        // our process is responsible for initializing the weights in. Also
        // initialize each box being responsible for all of the spatial domain.
        unsigned log2LocalFreqBoxes = 0;
        unsigned log2LocalSpatialBoxes = 0;
        Array<unsigned,d> log2LocalFreqBoxesPerDim;
        Array<unsigned,d> log2LocalSpatialBoxesPerDim;
        for( unsigned j=0; j<d; ++j )
        {
            log2LocalFreqBoxesPerDim[j] = L-log2FreqBoxesPerDim[j];
            log2LocalSpatialBoxesPerDim[j] = 0;
            log2LocalFreqBoxes += log2LocalFreqBoxesPerDim[j];
        }

        // Compute {zi} for the Chebyshev nodes of order q over [-1/2,+1/2]
        Array<R,q> chebyNodes;
        for( unsigned i=0; i<q; ++i )
            chebyNodes[i] = 0.5*cos(i*Pi/(q-1));

        // Compute the Chebyshev grid over [-1/2,+1/2]^d
        Array< Array<R,d>,Power<q,d>::value > chebyGrid;
        ComputeChebyGrid( chebyNodes, chebyGrid );

        // Initialize the weights using Lagrangian interpolation on the 
        // smooth component of the kernel.
        vector< Array<C,Power<q,d>::value> > weights(1<<log2LocalFreqBoxes);
        InitializeWeights<Psi,R,d,q>
        ( N, mySources, chebyNodes, myFreqBoxWidths, myFreqBox,
          log2LocalFreqBoxes, log2FreqBoxesPerDim, weights      );

        // First half of algorithm: frequency interpolation
        for( unsigned l=1; l<L/2; ++l )
        {
            // Compute the width of the nodes at level l
            const R wA = static_cast<R>(1) / static_cast<R>(1<<l);
            const R wB = static_cast<R>(1) / static_cast<R>(1<<(L-l));

            if( s <= d*(L-l) )
            {
                // Form the N^d/S = 2^(d*L) / 2^s = 2^(d*L-s) weights

                // Loop over A boxes in spatial domain. 'i' will represent the 
                // leaf number w.r.t. the tree implied by cyclically assigning
                // the spatial bisections across the d dimensions. Thus if we 
                // distribute the data cyclically in the _reverse_ order over 
                // the d dimensions, then the ReduceScatter will not require 
                // any packing or unpacking.
                vector< Array<C,Power<q,d>::value> > oldWeights = weights;
                for( unsigned i=0; i<(1<<log2LocalSpatialBoxes); ++i )
                {
                    // Compute the coordinates and center of this spatial box
                    Array<R,d> x0;
                    Array<unsigned,d> A;
                    for( unsigned j=0; j<d; ++j )
                    {
                        static unsigned log2LocalSpatialBoxesUpToDim = 0;
                        // A[j] = (i/localSpatialBoxesUpToDim) % 
                        //        localSpatialBoxesPerDim[j]
                        A[j] = (i>>log2LocalSpatialBoxesUpToDim) &
                               ((1<<log2LocalSpatialBoxesPerDim[j])-1);
                        x0[j] = myFreqBoxOffsets[j] + A[j]*wA + wA/2;
                    }

                    // Loop over the B boxes in frequency domain
                    for( unsigned k=0; k<(1<<log2LocalFreqBoxes); ++k )
                    {
                        // Compute the coordinates and center of this freq box
                        Array<R,d> p0;
                        Array<unsigned,d> B;
                        for( unsigned j=0; j<d; ++j )
                        {
                            static unsigned log2LocalFreqBoxesUpToDim = 0;
                            B[j] = (k>>log2LocalFreqBoxesUpToDim) &
                                   ((1<<log2LocalFreqBoxesPerDim[j])-1);
                            p0[j] = mySpatialBoxOffsets[j] + B[j]*wB + wB/2;
                        }

                        const unsigned pairKey = k+i*(1<<log2LocalFreqBoxes);
                        const unsigned parentPairOffset = 
                            (k<<d)+(i>>d)*(1<<(log2LocalFreqBoxes+d));
                        FreqWeightRecursion<Psi,R,d,q>
                        ( N, chebyNodes, chebyGrid, x0, p0, wB,
                          parentPairOffset, oldWeights, weights[pairKey] );
                    }
                }

                // Refine the spatial domain and coursen the frequency domain
                for( unsigned j=0; j<d; ++j )
                {
                    --log2LocalFreqBoxesPerDim[j];
                    ++log2LocalSpatialBoxesPerDim[j];
                }
                log2LocalFreqBoxes -= d;
                log2LocalSpatialBoxes += d;
            }
            else 
            {
                static vector< Array<C,Power<q,d>::value> > 
                      partialWeights(1<<(d+log2LocalFreqBoxes));
                // There are currently 2^(d*(L-l)) leaves. The frequency 
                // partitioning is implied by reading the rank bits right-to-
                // left, but the spatial partitioning is implied by reading the
                // rank bits left-to-right starting from bit s-1. The spatial 
                // partitioning among cores begins at the precise moment when 
                // trees begin mergining in the frequency domain: the lowest 
                // l such that s > d*(L-l), namely, l = L - floor( s/d ). The 
                // first merge is the only case where the team could potentially
                // differ from 2^d processes.
                unsigned log2Procs = ( l == L-(s/d) ? s-d*(L-l) : d ); 

                // We notice that our consistency in the cyclic bisection of 
                // the frequency domain means that if log2Procs=a, then 
                // we communicate with 1 other process in each of the first 
                // a of d dimensions. Getting these ranks is implicit in the
                // tree structure.
                
                // Form the partial weights. 

                // ReduceScatter over the necessary team into weights
                // MPI_Comm_group, MPI_Group_incl, MPI_Comm_group...
            }
        }

        // Switch to spatial interpolation
        {
            // Compute the width of the nodes at level l
            const unsigned l = L/2;
            const R wA = static_cast<R>(1) / static_cast<R>(1<<l);
            const R wB = static_cast<R>(1) / static_cast<R>(1<<(L-l));
            vector< Array<complex<R>,Power<q,d>::value> > oldWeights = weights;
            for( unsigned i=0; i<(1<<log2LocalSpatialBoxes); ++i )
            {
                // Compute the coordinates and center of this spatial box
                Array<R,d> x0;
                Array<unsigned,d> A;
                for( unsigned j=0; j<d; ++j )
                {
                    static unsigned log2LocalSpatialBoxesUpToDim = 0; 
                    // A[j] = (i/localSpatialBoxesUpToDim) % 
                    //        localSpatialBoxesPerDim[j]
                    A[j] = (i>>log2LocalSpatialBoxesUpToDim) &
                           ((1<<log2LocalSpatialBoxesPerDim[j])-1);
                    x0[j] = myFreqBoxOffsets[j] + A[j]*wA + wA/2;
                }

                static Array< Array<R,d>,Power<q,d>::value > xPoints;
                for( unsigned t=0; t<Power<q,d>::value; ++t )
                    for( unsigned j=0; j<d; ++j )
                        xPoints[t][j] = x0[j] + wA*chebyGrid[t][j];

                for( unsigned k=0; k<(1<<log2LocalFreqBoxes); ++k )
                {
                    // Compute the coordinates and center of this freq box
                    Array<R,d> p0;
                    Array<unsigned,d> B;
                    for( unsigned j=0; j<d; ++j )
                    {
                        static unsigned log2LocalFreqBoxesUpToDim = 0;
                        B[j] = (k>>log2LocalFreqBoxesUpToDim) &
                               ((1<<log2LocalFreqBoxesPerDim[j])-1);
                        p0[j] = mySpatialBoxOffsets[j] + B[j]*wB + wB/2;
                    }

                    static Array< Array<R,d>,Power<q,d>::value > pPoints;
                    for( unsigned t=0; t<Power<q,d>::value; ++t )
                        for( unsigned j=0; j<d; ++j )
                            pPoints[t][j] = p0[j] + wB*chebyGrid[t][j];

                    const unsigned key = k+i*(1<<log2LocalFreqBoxes);
                    for( unsigned t=0; t<Power<q,d>::value; ++t )
                    {
                        weights[key][t] = 0;
                        for( unsigned tp=0; tp<Power<q,d>::value; ++tp )
                        {
                            C a = C(0,2*Pi*N*Psi::Eval(xPoints[t],pPoints[s]));
                            weights[key][t] += exp(a)*oldWeights[key][tp];
                        }
                    }
                }
            }
        }

        // Second half of algorithm: spatial interpolation
        for( unsigned l=L/2; l<L; ++l )
        {
            if( s <= d*(L-l) )
            {
                // Form the weights
            }
            else
            {
                unsigned log2Procs = ( l == L-(s/d) ? s-d*(L-l) : d );

                // Form the partial weights

                // ReduceScatter over the necessary team into weights
            }
        }

        // Construct Low-Rank Potentials (LRPs) from weights
        {
            myLRPs.resize( 1<<(d*L-s) );
            // Fill in the LRPs
        }
    }
}

#endif /* BFIO_TRANSFORM_HPP */

