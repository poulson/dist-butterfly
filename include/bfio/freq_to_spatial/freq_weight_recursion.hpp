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
#ifndef BFIO_FREQ_TO_SPATIAL_FREQ_WEIGHT_RECURSION_HPP
#define BFIO_FREQ_TO_SPATIAL_FREQ_WEIGHT_RECURSION_HPP 1

#include "bfio/tools/lagrange.hpp"

namespace bfio {
namespace freq_to_spatial {

template<typename R,unsigned d,unsigned q>
void
FreqWeightRecursion
( const AmplitudeFunctor<R,d>& Amp,
  const PhaseFunctor<R,d>& Phi,
  const unsigned log2Procs,
  const unsigned myTeamRank,
  const unsigned N, 
  const std::vector< Array<R,d> >& chebyGrid,
  const Array<R,d>& x0A,
  const Array<R,d>& p0B,
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
    static std::vector<R> pRefB( (q_to_d << d)*d );
    static std::vector<R> LFreq( q_to_2d << d );

    if( !initialized )
    {
        for( unsigned c=0; c<(1u<<d); ++c )
        {
            for( unsigned t=0; t<q_to_d; ++t )
            {
                for( unsigned tPrime=0; tPrime<q_to_d; ++tPrime )
                {
                    // Map p_t'(Bc) to the reference domain of B and 
                    // store the Lagrangian evaluation
                    Array<R,d> ptPrimeBcRefB;
                    for( unsigned j=0; j<d; ++j )
                    {
                        pRefB[c*q_to_d*d+tPrime*d+j] = ptPrimeBcRefB[j] = 
                            ( (c>>j)&1 ? (2*chebyGrid[tPrime][j]+1)/4 :
                                         (2*chebyGrid[tPrime][j]-1)/4  );
                    }
                    LFreq[c*q_to_2d+t+tPrime*q_to_d] = 
                        Lagrange<R,d,q>( t, ptPrimeBcRefB );
                }
            }
        }
        initialized = true;
    }

    // We seek performance by isolating the Lagrangian interpolation as
    // a matrix-vector multiplication
    //
    // To do so, the frequency weight recursion is broken into 3 steps.
    // 
    // For each child:
    //  1) scale the old weights with the appropriate exponentials
    //  2) accumulate the lagrangian matrix against the scaled weights
    // Finally:
    // 3) scale the accumulated weights

    for( unsigned t=0; t<q_to_d; ++t )
        weightGrid[t] = 0;

    for( unsigned cLocal=0; cLocal<(1u<<(d-log2Procs)); ++cLocal )
    {
        // Step 1
        const unsigned c = (cLocal<<log2Procs) + myTeamRank;
        const unsigned key = parentOffset + cLocal;

        WeightGrid<R,d,q> scaledWeightGrid;
        for( unsigned tPrime=0; tPrime<q_to_d; ++tPrime )
        {
            Array<R,d> ptPrime;
            for( unsigned j=0; j<d; ++j )
                ptPrime[j] = p0B[j] + wB*pRefB[c*q_to_d*d+tPrime*d+j];

            const R alpha = TwoPi*Phi(x0A,ptPrime);
            if( Amp.algorithm == MiddleSwitch )
            {
                scaledWeightGrid[tPrime] = 
                    C(cos(alpha),sin(alpha)) * oldWeightGridList[key][tPrime];
            }
            else if( Amp.algorithm == Prefactor )
            {
                scaledWeightGrid[tPrime] = Amp(x0A,ptPrime) * 
                    C(cos(alpha),sin(alpha)) * oldWeightGridList[key][tPrime];
            }
        }
        
        // Step 2
        RealMatrixComplexVec
        ( q_to_d, q_to_d, (R)1, &LFreq[c*q_to_2d], q_to_d, 
          &scaledWeightGrid[0], (R)1, &weightGrid[0] );
    }

    // Step 3
    for( unsigned t=0; t<q_to_d; ++t )
    {
        Array<R,d> ptB;
        for( unsigned j=0; j<d; ++j )
            ptB[j] = p0B[j] + wB*chebyGrid[t][j];

        const R alpha = TwoPi * Phi(x0A,ptB);
        if( Amp.algorithm == MiddleSwitch )
        {
            weightGrid[t] /= C(cos(alpha),sin(alpha));
        }
        else if( Amp.algorithm == Prefactor )
        {
            weightGrid[t] /= Amp(x0A,ptB) * C(cos(alpha),sin(alpha));
        }
    }
}

} // freq_to_spatial
} // bfio

#endif /* BFIO_FREQ_TO_SPATIAL_FREQ_WEIGHT_RECURSION_HPP */

