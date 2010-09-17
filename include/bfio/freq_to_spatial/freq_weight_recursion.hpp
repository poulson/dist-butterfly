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
#pragma once
#ifndef BFIO_FREQ_TO_SPATIAL_FREQ_WEIGHT_RECURSION_HPP
#define BFIO_FREQ_TO_SPATIAL_FREQ_WEIGHT_RECURSION_HPP 1

namespace bfio {
namespace freq_to_spatial {

template<typename R,unsigned d,unsigned q>
void
FreqWeightRecursion
( const AmplitudeFunctor<R,d>& Amp,
  const PhaseFunctor<R,d>& Phi,
  const unsigned log2NumMergingProcesses,
  const unsigned myTeamRank,
  const unsigned N, 
  const Context<R,d,q>& context,
  const Array<R,d>& x0A,
  const Array<R,d>& p0B,
  const Array<R,d>& wB,
  const unsigned parentOffset,
  const WeightGridList<R,d,q>& oldWeightGridList,
        WeightGrid<R,d,q>& weightGrid
)
{
    typedef std::complex<R> C;
    const unsigned q_to_d = Pow<q,d>::val;
    const unsigned q_to_2d = Pow<q,2*d>::val;

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

    const std::vector<R>& freqMaps = context.GetFreqMaps();
    const std::vector< Array<R,d> >& freqChildGrids = 
        context.GetFreqChildGrids();
    for( unsigned cLocal=0; cLocal<(1u<<(d-log2NumMergingProcesses)); ++cLocal )
    {
        // Step 1
        const unsigned c = (cLocal<<log2NumMergingProcesses) + myTeamRank;
        const unsigned key = parentOffset + cLocal;

        WeightGrid<R,d,q> scaledWeightGrid;
        for( unsigned tPrime=0; tPrime<q_to_d; ++tPrime )
        {
            Array<R,d> ptPrime;
            for( unsigned j=0; j<d; ++j )
            {
                ptPrime[j] = 
                    p0B[j] + wB[j]*freqChildGrids[c*q_to_d*d+tPrime*d][j];
            }

            const C beta = ImagExp( TwoPi*Phi(x0A,ptPrime) );
            if( Amp.algorithm == MiddleSwitch )
            {
                scaledWeightGrid[tPrime] = 
                    beta * oldWeightGridList[key][tPrime];
            }
            else if( Amp.algorithm == Prefactor )
            {
                scaledWeightGrid[tPrime] = Amp(x0A,ptPrime) * 
                    beta * oldWeightGridList[key][tPrime];
            }
        }
        
        // Step 2
        RealMatrixComplexVec
        ( q_to_d, q_to_d, (R)1, &freqMaps[c*q_to_2d], q_to_d, 
          &scaledWeightGrid[0], (R)1, &weightGrid[0] );
    }

    // Step 3
    const std::vector< Array<R,d> >& chebyshevGrid = context.GetChebyshevGrid();
    for( unsigned t=0; t<q_to_d; ++t )
    {
        Array<R,d> ptB;
        for( unsigned j=0; j<d; ++j )
            ptB[j] = p0B[j] + wB[j]*chebyshevGrid[t][j];

        const C beta = ImagExp( TwoPi*Phi(x0A,ptB) );
        if( Amp.algorithm == MiddleSwitch )
        {
            weightGrid[t] /= beta;
        }
        else if( Amp.algorithm == Prefactor )
        {
            weightGrid[t] /= Amp(x0A,ptB) * beta;
        }
    }
}

} // freq_to_spatial
} // bfio

#endif // BFIO_FREQ_TO_SPATIAL_FREQ_WEIGHT_RECURSION_HPP

