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

        // Determine the number of partitions in each dimension of the 
        // frequency domain by applying the partitions cyclically over the
        // d dimensions. We can simultaneously compute the indices of our 
        // box in each spatial dimension.
        Array<R,d> myFreqBoxWidths;
        Array<unsigned,d> myFreqBoxCoords;
        Array<unsigned,d> log2FreqParts;
        for( unsigned j=0; j<d; ++j )
            myFreqBoxCoords[j] = log2FreqParts[j] = 0;
        for( unsigned j=s; j>0; --j )
        {
            static unsigned nextPartDim = 0;
            // Double our current coordinate in the 'nextPartDim' dimension 
            // and then choose the left/right position based on the (j-1)'th
            // bit of our rank
            myFreqBoxCoords[nextPartDim] = 
                (myFreqBoxCoords[nextPartDim]<<1)+rankBits[j-1];

            log2FreqParts[nextPartDim]++;
            nextPartDim = (nextPartDim+1) % d;
        }
        for( unsigned j=0; j<d; ++j )
            myFreqBoxWidths[j] = 1. / static_cast<R>(1<<log2FreqParts[j]);

        // Compute the number of 1/N width boxes in the frequency domain that 
        // our process is responsible for initializing the weights in. Also
        // initialize each box being responsible for all of the spatial domain.
        unsigned freqBoxes = 1;
        unsigned spatialBoxes = 1;
        Array<unsigned,d> log2FreqBoxes;
        Array<unsigned,d> log2SpatialBoxes;
        for( unsigned j=0; j<d; ++j )
        {
            log2FreqBoxes[j] = L-log2FreqParts[j];
            log2SpatialBoxes[j] = 0;
            freqBoxes <<= log2FreqBoxes[j];
        }

        // Compute {zi} for the Chebyshev grid of order q over [-1/2,+1/2]
        Array<R,q> chebyGrid;
        for( unsigned i=0; i<q; ++i )
            chebyGrid[i] = 0.5*cos(i*Pi/(q-1));

        // Initialize the weights using Lagrangian interpolation on the 
        // smooth component of the kernel.
        vector< Array<C,Power<q,d>::value> > weights(freqBoxes);
        InitializeWeights<Psi,R,d,q>
        ( N, mySources, chebyGrid, myFreqBoxWidths,
          myFreqBoxCoords, freqBoxes, log2FreqBoxes, weights );

        // First half of algorithm: frequency interpolation
        vector< Array<C,Power<q,d>::value> > partialWeights((2<<d)*freqBoxes);
        for( unsigned l=1; l<L/2; ++l )
        {
            if( s <= d*(L-l) )
            {
                // Form the N^d/S = 2^(d*L) / 2^s = 2^(d*L-s) weights

                // Loop over A boxes in spatial domain
                for( unsigned i=0; i<spatialBoxes; ++i )
                {
                    // Compute the coordinates and center of this spatial box

                    // Loop over the B boxes in frequency domain
                    for( unsigned j=0; j<freqBoxes; ++j )
                    {
                        // Compute the coordinates and center of this freq box

                        // Sum over the frequency children
                        for( unsigned c=0; c<(1<<d); ++c )
                        {

                        }
                        // Multiply by the prefactor 
                        //weights[j+i*freqBoxes] *= 
                        //    exp( C(0,-2*Pi*N*Psi::Eval(x0,pt)) );
                    }
                }

                // Refine the spatial domain and coursen the frequency domain
                for( unsigned j=0; j<d; ++j )
                {
                    --log2FreqBoxes[j];
                    ++log2SpatialBoxes[j];
                }
                freqBoxes >>= d;
                spatialBoxes <<= d;
            }
            else 
            {
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
                // we communicate with 1 other process in the first a of d 
                // dimensions. Getting these ranks is implicit in the tree 
                // structure.
                
                // Form the partial weights. 

                // ReduceScatter over the necessary team into weights
                // MPI_Comm_group, MPI_Group_incl, MPI_Comm_group...
            }
        }

        // Switch to spatial interpolation

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

