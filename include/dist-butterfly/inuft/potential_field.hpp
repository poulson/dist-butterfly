/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of DistButterfly and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef DBF_INUFT_POTENTIAL_FIELD_HPP
#define DBF_INUFT_POTENTIAL_FIELD_HPP

#include <array>
#include <complex>
#include <stdexcept>
#include <vector>

#include "dist-butterfly/structures/box.hpp"
#include "dist-butterfly/structures/constrained_htree_walker.hpp"
#include "dist-butterfly/structures/low_rank_potential.hpp"
#include "dist-butterfly/structures/weight_grid.hpp"
#include "dist-butterfly/structures/weight_grid_list.hpp"

#include "dist-butterfly/inuft/context.hpp"

#include "dist-butterfly/tools/special_functions.hpp"

namespace dbf {

using std::array;
using std::complex;
using std::size_t;
using std::vector;

namespace inuft {
template<typename R,size_t d,size_t q>
class PotentialField
{
    const Context<R,d,q>& context_;
    const Box<R,d> sBox_;
    const Box<R,d> myTBox_;
    const array<size_t,d> log2TSubboxesPerDim_;

    array<R,d> wA_;
    vector<array<R,d>> sChebyshevGrid_;
    array<size_t,d> log2TSubboxesUpToDim_;
    vector<LRP<R,d,q>> LRPs_;

public:
    PotentialField
    ( const Context<R,d,q>& context,
      const Box<R,d>& sBox,
      const Box<R,d>& myTBox,
      const array<size_t,d>& log2TSubboxesPerDim,
      const WeightGridList<R,d,q>& weightGridList );

    complex<R> Evaluate( const array<R,d>& x ) const;

    const Box<R,d>& GetMyTargetBox() const;
    size_t GetNumSubboxes() const;
    const array<R,d>& GetSubboxWidths() const;
    const array<size_t,d>& GetLog2SubboxesPerDim() const;
    const array<size_t,d>& GetLog2SubboxesUpToDim() const;
};

// Implementations

/*
 * Remark: There is significant code duplication in 
 *         inuft::PotentialField from butterfly::PotentialField, but this was 
 *         chosen as an alternative to coupling the two classes. 
 */

template<typename R,size_t d,size_t q>
inline
PotentialField<R,d,q>::PotentialField
( const Context<R,d,q>& context,
  const Box<R,d>& sBox,
  const Box<R,d>& myTBox,
  const array<size_t,d>& log2TSubboxesPerDim,
  const WeightGridList<R,d,q>& weightGridList )
: context_(context), sBox_(sBox), myTBox_(myTBox),
  log2TSubboxesPerDim_(log2TSubboxesPerDim)
{ 
    // Compute the widths of the target subboxes
    for( size_t j=0; j<d; ++j )
        wA_[j] = myTBox.widths[j] / (1<<log2TSubboxesPerDim[j]);

    // Compute the array of the partial sums
    log2TSubboxesUpToDim_[0] = 0;
    for( size_t j=1; j<d; ++j )
    {
        log2TSubboxesUpToDim_[j] = 
            log2TSubboxesUpToDim_[j-1] + log2TSubboxesPerDim[j-1];
    }

    // Figure out the size of our LRP vector by summing log2TSubboxesPerDim
    size_t log2TSubboxes = 0;
    for( size_t j=0; j<d; ++j )
        log2TSubboxes += log2TSubboxesPerDim[j];
    LRPs_.resize( 1<<log2TSubboxes );

    // The weightGridList is assumed to be ordered by the constrained 
    // HTree described by log2TSubboxesPerDim. We will unroll it
    // lexographically into the LRP vector.
    ConstrainedHTreeWalker<d> AWalker( log2TSubboxesPerDim );
    for( size_t tIndex=0; tIndex<LRPs_.size(); ++tIndex, AWalker.Walk() )
    {
        const array<size_t,d>& A = AWalker.State();

        // Unroll the indices of A into its lexographic position
        size_t k=0;
        for( size_t j=0; j<d; ++j )
            k += A[j] << log2TSubboxesUpToDim_[j];

        // Now fill the k'th LRP index
        for( size_t j=0; j<d; ++j )
            LRPs_[k].x0[j] = myTBox.offsets[j] + (A[j]+R(1)/R(2))*wA_[j];
        LRPs_[k].weightGrid = weightGridList[tIndex];
    }

    // Compute the source center
    array<R,d> p0;
    for( size_t j=0; j<d; ++j )
        p0[j] = sBox.offsets[j] + sBox.widths[j]/2;

    // Fill the Chebyshev grid on the source box
    const vector<array<R,d>>& chebyshevGrid = context.GetChebyshevGrid();
    sChebyshevGrid_.resize( Pow<q,d>::val );
    for( size_t t=0; t<Pow<q,d>::val; ++t )
        for( size_t j=0; j<d; ++j )
            sChebyshevGrid_[t][j] = p0[j] + chebyshevGrid[t][j]*sBox.widths[j];
}

template<typename R,size_t d,size_t q>
inline complex<R>
PotentialField<R,d,q>::Evaluate( const array<R,d>& x ) const
{
    typedef complex<R> C;

#ifndef RELEASE
    for( size_t j=0; j<d; ++j )
    {
        if( x[j] < myTBox.offsets_[j] ||
            x[j] > myTBox.offsets_[j] + myTBox_.widths[j] )
        {
            throw std::runtime_error
            ("Tried to evaluate outside of potential range");
        }
    }
#endif

    // Compute the lexographic position of the LRP to use for evaluation
    size_t k = 0;
    for( size_t j=0; j<d; ++j ) 
    {
        size_t owningIndex = size_t((x[j]-myTBox_.offsets[j])/wA_[j]);
        const size_t maxIndex = size_t((1u<<log2TSubboxesPerDim_[j])-1);
        owningIndex = std::min(owningIndex,maxIndex);
        k += owningIndex << log2TSubboxesUpToDim_[j];
    }
    const LRP<R,d,q>& lrp = LRPs_[k];

    const Direction direction = context_.GetDirection();
    const R SignedTwoPi = ( direction==FORWARD ? -TwoPi<R>() : TwoPi<R>() );

    C potential = 0;
    for( size_t t=0; t<Pow<q,d>::val; ++t )
    {
        const R realWeight = lrp.weightGrid.RealWeight(t);
        const R imagWeight = lrp.weightGrid.ImagWeight(t);
        const C weight = C( realWeight, imagWeight );

        // Compute the dot product of the gridpoint with the target location
        R dot = 0;
        for( size_t j=0; j<d; ++j )
            dot += x[j]*sChebyshevGrid_[t][j];
        potential += ImagExp<R>( SignedTwoPi*dot )*weight;
    }
    return potential;
}

template<typename R,size_t d,size_t q>
inline const Box<R,d>&
PotentialField<R,d,q>::GetMyTargetBox() const
{ return myTBox_; }

template<typename R,size_t d,size_t q>
inline size_t
PotentialField<R,d,q>::GetNumSubboxes() const
{ return LRPs_.size(); }

template<typename R,size_t d,size_t q>
inline const array<R,d>&
PotentialField<R,d,q>::GetSubboxWidths() const
{ return wA_; }

template<typename R,size_t d,size_t q>
inline const array<size_t,d>&
PotentialField<R,d,q>::GetLog2SubboxesPerDim() const
{ return log2TSubboxesPerDim_; }

template<typename R,size_t d,size_t q>
inline const array<size_t,d>&
PotentialField<R,d,q>::GetLog2SubboxesUpToDim() const
{ return log2TSubboxesUpToDim_; }

} // inuft
} // dbf

#endif // ifndef DBF_INUFT_POTENTIAL_FIELD_HPP
