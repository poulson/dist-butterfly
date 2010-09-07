/*
   Copyright (c) 2010, Jack Poulson
   All rights reserved.

   This file is part of ButterflyFIO.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are met:

    - Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    - Neither the name of the owner nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
   ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
   LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
   CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
   SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
   INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
   CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
   ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
   POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once
#ifndef BFIO_FREQ_TO_SPATIAL_SPATIAL_WEIGHT_RECURSION_HPP
#define BFIO_FREQ_TO_SPATIAL_SPATIAL_WEIGHT_RECURSION_HPP 1

#include "bfio/tools/blas.hpp"
#include "bfio/tools/lagrange.hpp"

namespace bfio {
namespace freq_to_spatial {

template<typename R,unsigned d,unsigned q>
void
SpatialWeightRecursion
( const AmplitudeFunctor<R,d>& Amp,
  const PhaseFunctor<R,d>& Phi,
  const unsigned log2Procs,
  const unsigned myTeamRank,
  const unsigned N, 
  const std::vector< Array<R,d> >& chebyGrid,
  const unsigned ARelativeToAp,
  const Array<R,d>& x0A,
  const Array<R,d>& x0Ap,
  const Array<R,d>& p0B,
  const R wA,
  const R wB,
  const unsigned parentOffset,
  const WeightGridList<R,d,q>& oldWeightGridList,
        WeightGrid<R,d,q>& weightGrid
)
{
    typedef std::complex<R> C;
    const unsigned q_to_d = Pow<q,d>::val;
    const unsigned q_to_2d = Pow<q,2*d>::val;

    static bool initialized = false;
    static std::vector<R> LSpatial( q_to_2d << d );

    if( !initialized )
    {
        for( unsigned p=0; p<(1u<<d); ++p )
        {
            for( unsigned tPrime=0; tPrime<q_to_d; ++tPrime )
            {
                for( unsigned t=0; t<q_to_d; ++t )
                {
                    // Map x_t(A) to the reference domain of its parent
                    Array<R,d> xtARefAp;
                    for( unsigned j=0; j<d; ++j )
                    {
                        xtARefAp[j] = 
                            ( (p>>j)&1 ? (2*chebyGrid[t][j]+1)/4 :
                                         (2*chebyGrid[t][j]-1)/4  );
                    }

                    LSpatial[p*q_to_2d + t+tPrime*q_to_d] = 
                        Lagrange<R,d,q>( tPrime, xtARefAp );
                }
            }
        }
        initialized = true;
    }

    for( unsigned t=0; t<q_to_d; ++t )
        weightGrid[t] = 0;

    // We seek performance by isolating the Lagrangian interpolation as 
    // a matrix-vector multiplication.
    //
    // To do so, the spatial weight recursion is broken into three updates:
    // For each child c:
    //  1) scale the old weights with the appropriate exponentials
    //  2) multiply the lagrangian matrix against the scaled weights
    //  3) scale and accumulate the result of the lagrangian mat-vec
    for( unsigned cLocal=0; cLocal<(1u<<(d-log2Procs)); ++cLocal )
    {
        // Step 1: scale the old weights
        WeightGrid<R,d,q> scaledWeightGrid;
        const unsigned c = (cLocal<<log2Procs) + myTeamRank;
        const unsigned key = parentOffset + cLocal;
        Array<R,d> p0Bc;
        for( unsigned j=0; j<d; ++j )
            p0Bc[j] = p0B[j] + ( (c>>j)&1 ? wB/4 : -wB/4 );
        for( unsigned tPrime=0; tPrime<q_to_d; ++tPrime )
        {
            Array<R,d> xtPrimeAp;
            for( unsigned j=0; j<d; ++j )
                xtPrimeAp[j] = x0Ap[j] + (2*wA)*chebyGrid[tPrime][j];
            const R alpha = TwoPi * Phi(xtPrimeAp,p0Bc);
            if( Amp.algorithm == MiddleSwitch )
            {
                scaledWeightGrid[tPrime] = 
                    oldWeightGridList[key][tPrime] / C(cos(alpha),sin(alpha));
            }
            else if( Amp.algorithm == Prefactor )
            {
                scaledWeightGrid[tPrime] = 
                    oldWeightGridList[key][tPrime] / 
                    ( Amp(xtPrimeAp,p0Bc) * C(cos(alpha),sin(alpha)) );
            }
        }

        // Step 2: perform the matrix-vector multiply
        WeightGrid<R,d,q> expandedWeightGrid;
        RealMatrixComplexVec
        ( q_to_d, q_to_d, 
          (R)1, &LSpatial[ARelativeToAp*q_to_2d], q_to_d, 
                &scaledWeightGrid[0],
          (R)0, &expandedWeightGrid[0] );

        // Step 3: scale the result
        for( unsigned t=0; t<Pow<q,d>::val; ++t )
        {
            Array<R,d> xtA;
            for( unsigned j=0; j<d; ++j )
                xtA[j] = x0A[j] + wA*chebyGrid[t][j];
            const R alpha = TwoPi * Phi(xtA,p0Bc);
            if( Amp.algorithm == MiddleSwitch )
            {
                weightGrid[t] += C(cos(alpha),sin(alpha)) * 
                                 expandedWeightGrid[t];
            }
            else if( Amp.algorithm == Prefactor )
            {
                weightGrid[t] += Amp(xtA,p0Bc) * C(cos(alpha),sin(alpha)) *
                                 expandedWeightGrid[t];
            }
        }
    }
}

} // freq_to_spatial
} // bfio

#endif /* BFIO_FREQ_TO_SPATIAL_SPATIAL_WEIGHT_RECURSION_HPP */

