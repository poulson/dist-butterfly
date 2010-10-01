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
#ifndef BFIO_FREQ_TO_SPATIAL_FREQ_WEIGHT_RECURSION_HPP
#define BFIO_FREQ_TO_SPATIAL_FREQ_WEIGHT_RECURSION_HPP 1

#include <cstddef>
#include <vector>

#include "bfio/constants.hpp"

#include "bfio/structures/array.hpp"
#include "bfio/structures/weight_grid.hpp"
#include "bfio/structures/weight_grid_list.hpp"

#include "bfio/functors/phase_functor.hpp"

#include "bfio/tools/special_functions.hpp"

namespace bfio {
namespace freq_to_spatial {

template<typename R,std::size_t d,std::size_t q>
void
FreqWeightRecursion
( const PhaseFunctor<R,d>& Phi,
  const std::size_t log2NumMergingProcesses,
  const std::size_t myTeamRank,
  const std::size_t N, 
  const Context<R,d,q>& context,
  const Array<R,d>& x0A,
  const Array<R,d>& p0B,
  const Array<R,d>& wB,
  const std::size_t parentOffset,
  const WeightGridList<R,d,q>& oldWeightGridList,
        WeightGrid<R,d,q>& weightGrid
)
{
    const std::size_t q_to_d = Pow<q,d>::val;
    const std::size_t q_to_2d = Pow<q,2*d>::val;

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

    for( std::size_t t=0; t<q_to_d; ++t )
    {
        weightGrid.RealWeight(t) = 0;
        weightGrid.ImagWeight(t) = 0;
    }

    std::vector<R> phiResults;
    std::vector<R> sinResults;
    std::vector<R> cosResults;
    std::vector< Array<R,d> > xPoint( 1, x0A );
    std::vector< Array<R,d> > pPoints( q_to_d );
    const std::vector<R>& freqMaps = context.GetFreqMaps();
    const std::vector< Array<R,d> >& freqChildGrids = 
        context.GetFreqChildGrids();
    for( std::size_t cLocal=0; 
         cLocal<(1u<<(d-log2NumMergingProcesses)); 
         ++cLocal )
    {
        // Step 1
        const std::size_t c = (cLocal<<log2NumMergingProcesses) + myTeamRank;
        const std::size_t key = parentOffset + cLocal;

        // Form the set of p points to evaluate
        for( std::size_t tPrime=0; tPrime<q_to_d; ++tPrime )
            for( std::size_t j=0; j<d; ++j )
                pPoints[tPrime][j] = 
                    p0B[j] + wB[j]*freqChildGrids[c*q_to_d+tPrime][j];

        // Form the phase factors
        Phi.BatchEvaluate( xPoint, pPoints, phiResults );
        for( std::size_t tPrime=0; tPrime<q_to_d; ++tPrime )
            phiResults[tPrime] *= TwoPi;
        SinCosBatch( phiResults, sinResults, cosResults );

        WeightGrid<R,d,q> scaledWeightGrid;
        for( std::size_t tPrime=0; tPrime<q_to_d; ++tPrime )
        {
            const R realWeight = oldWeightGridList[key].RealWeight(tPrime);
            const R imagWeight = oldWeightGridList[key].ImagWeight(tPrime);
            scaledWeightGrid.RealWeight(tPrime) = 
                cosResults[tPrime]*realWeight - sinResults[tPrime]*imagWeight;
            scaledWeightGrid.ImagWeight(tPrime) = 
                sinResults[tPrime]*realWeight + cosResults[tPrime]*imagWeight;
        }
        
        // Step 2
        Gemm
        ( 'N', 'N', q_to_d, 2, q_to_d, 
          (R)1, &freqMaps[c*q_to_2d],      q_to_d,
                scaledWeightGrid.Buffer(), q_to_d,
          (R)1,       weightGrid.Buffer(), q_to_d );
    }

    // Step 3
    const std::vector< Array<R,d> >& chebyshevGrid = context.GetChebyshevGrid();
    for( std::size_t t=0; t<q_to_d; ++t )
        for( std::size_t j=0; j<d; ++j )
            pPoints[t][j] = p0B[j] + wB[j]*chebyshevGrid[t][j];
    Phi.BatchEvaluate( xPoint, pPoints, phiResults );
    for( std::size_t t=0; t<q_to_d; ++t )
        phiResults[t] *= -TwoPi;
    SinCosBatch( phiResults, sinResults, cosResults );
    for( std::size_t t=0; t<q_to_d; ++t )
    {
        const R realWeight = weightGrid.RealWeight(t);
        const R imagWeight = weightGrid.ImagWeight(t);
        weightGrid.RealWeight(t) = 
            cosResults[t]*realWeight - sinResults[t]*imagWeight;
        weightGrid.ImagWeight(t) = 
            sinResults[t]*realWeight + cosResults[t]*imagWeight;
    }
}

} // freq_to_spatial
} // bfio

#endif // BFIO_FREQ_TO_SPATIAL_FREQ_WEIGHT_RECURSION_HPP

