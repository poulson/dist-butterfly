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

    // Form a vector out of the point x0A so that we can batch evaluations. 
    // Also create vectors for the p points and the results.
    const std::vector< Array<R,d> > xPoint( 1, x0A );
    std::vector< Array<R,d> > pPoints( q_to_d );
    std::vector<R> phiResults( q_to_d );
    std::vector<C> imagExpResults( q_to_d );

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

        // Form the vector of p points to evaluate at
        for( unsigned tPrime=0; tPrime<q_to_d; ++tPrime )
            for( unsigned j=0; j<d; ++j )
                pPoints[tPrime][j] = 
                    p0B[j] + wB[j]*freqChildGrids[c*q_to_d*d+tPrime*d][j];

        // Form all of the phase factors
        Phi.BatchEvaluate( xPoint, pPoints, phiResults );
        for( unsigned j=0; j<q_to_d; ++j )
            phiResults[j] *= TwoPi;
        ImagExpBatch<R>( phiResults, imagExpResults );

        WeightGrid<R,d,q> scaledWeightGrid;
        for( unsigned tPrime=0; tPrime<q_to_d; ++tPrime )
            scaledWeightGrid[tPrime] = 
                imagExpResults[tPrime]*oldWeightGridList[key][tPrime];
        
        // Step 2
        RealMatrixComplexVec
        ( q_to_d, q_to_d, (R)1, &freqMaps[c*q_to_2d], q_to_d, 
          &scaledWeightGrid[0], (R)1, &weightGrid[0] );
    }

    // Step 3
    const std::vector< Array<R,d> >& chebyshevGrid = context.GetChebyshevGrid();
    for( unsigned t=0; t<q_to_d; ++t )
        for( unsigned j=0; j<d; ++j )
            pPoints[t][j] = p0B[j] + wB[j]*chebyshevGrid[t][j];
    Phi.BatchEvaluate( xPoint, pPoints, phiResults );
    for( unsigned t=0; t<q_to_d; ++t )
        phiResults[t] *= TwoPi;
    ImagExpBatch<R>( phiResults, imagExpResults );
    for( unsigned t=0; t<q_to_d; ++t )
        weightGrid[t] /= imagExpResults[t];
}

} // freq_to_spatial
} // bfio

#endif // BFIO_FREQ_TO_SPATIAL_FREQ_WEIGHT_RECURSION_HPP

