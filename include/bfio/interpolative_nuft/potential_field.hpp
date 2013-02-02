/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of ButterflyFIO and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef BFIO_INTERPOLATIVE_NUFT_POTENTIAL_FIELD_HPP
#define BFIO_INTERPOLATIVE_NUFT_POTENTIAL_FIELD_HPP

#include <array>
#include <complex>
#include <stdexcept>
#include <vector>

#include "bfio/structures/box.hpp"
#include "bfio/structures/constrained_htree_walker.hpp"
#include "bfio/structures/low_rank_potential.hpp"
#include "bfio/structures/weight_grid.hpp"
#include "bfio/structures/weight_grid_list.hpp"

#include "bfio/interpolative_nuft/context.hpp"

#include "bfio/tools/special_functions.hpp"

namespace bfio {

using std::array;
using std::complex;
using std::size_t;
using std::vector;

namespace interpolative_nuft {
template<typename R,size_t d,size_t q>
class PotentialField
{
    const interpolative_nuft::Context<R,d,q>& _context;
    const Box<R,d> _sourceBox;
    const Box<R,d> _myTargetBox;
    const array<size_t,d> _log2TargetSubboxesPerDim;

    array<R,d> _wA;
    vector<array<R,d>> _sourceChebyshevGrid;
    array<size_t,d> _log2TargetSubboxesUpToDim;
    vector<LRP<R,d,q>> _LRPs;

public:
    PotentialField
    ( const interpolative_nuft::Context<R,d,q>& context,
      const Box<R,d>& sourceBox,
      const Box<R,d>& myTargetBox,
      const array<size_t,d>& log2TargetSubboxesPerDim,
      const WeightGridList<R,d,q>& weightGridList );

    complex<R> Evaluate( const array<R,d>& x ) const;

    const Box<R,d>& GetMyTargetBox() const;
    size_t GetNumSubboxes() const;
    const array<R,d>& GetSubboxWidths() const;
    const array<size_t,d>& GetLog2SubboxesPerDim() const;
    const array<size_t,d>& GetLog2SubboxesUpToDim() const;
};
} // interpolative_nuft

// Implementations

/*
 * Remark: There is significant code duplication in 
 *         interpolative_nuft::PotentialField from
 *         general_fio::PotentialField, but this was chosen as an alternative
 *         to coupling the two classes. Since LagrangianNUFT is a slight 
 *         specialization of the GeneralFIO approach, 
 *         lagrangian_nuft::PotentialField _is_ built from the GeneralFIO 
 *         potential field.
 */

template<typename R,size_t d,size_t q>
interpolative_nuft::PotentialField<R,d,q>::PotentialField
( const interpolative_nuft::Context<R,d,q>& context,
  const Box<R,d>& sourceBox,
  const Box<R,d>& myTargetBox,
  const array<size_t,d>& log2TargetSubboxesPerDim,
  const WeightGridList<R,d,q>& weightGridList )
: _context(context), _sourceBox(sourceBox), _myTargetBox(myTargetBox),
  _log2TargetSubboxesPerDim(log2TargetSubboxesPerDim)
{ 
    // Compute the widths of the target subboxes
    for( size_t j=0; j<d; ++j )
        _wA[j] = myTargetBox.widths[j] / (1<<log2TargetSubboxesPerDim[j]);

    // Compute the array of the partial sums
    _log2TargetSubboxesUpToDim[0] = 0;
    for( size_t j=1; j<d; ++j )
    {
        _log2TargetSubboxesUpToDim[j] = 
            _log2TargetSubboxesUpToDim[j-1] + log2TargetSubboxesPerDim[j-1];
    }

    // Figure out the size of our LRP vector by summing log2TargetSubboxesPerDim
    size_t log2TargetSubboxes = 0;
    for( size_t j=0; j<d; ++j )
        log2TargetSubboxes += log2TargetSubboxesPerDim[j];
    _LRPs.resize( 1<<log2TargetSubboxes );

    // The weightGridList is assumed to be ordered by the constrained 
    // HTree described by log2TargetSubboxesPerDim. We will unroll it
    // lexographically into the LRP vector.
    ConstrainedHTreeWalker<d> AWalker( log2TargetSubboxesPerDim );
    for( size_t targetIndex=0; 
         targetIndex<_LRPs.size();
         ++targetIndex, AWalker.Walk() )
    {
        const array<size_t,d> A = AWalker.State();

        // Unroll the indices of A into its lexographic position
        size_t k=0;
        for( size_t j=0; j<d; ++j )
            k += A[j] << _log2TargetSubboxesUpToDim[j];

        // Now fill the k'th LRP index
        for( size_t j=0; j<d; ++j )
            _LRPs[k].x0[j] = myTargetBox.offsets[j] + (A[j]+0.5)*_wA[j];
        _LRPs[k].weightGrid = weightGridList[targetIndex];
    }

    // Compute the source center
    array<R,d> p0;
    for( size_t j=0; j<d; ++j )
        p0[j] = sourceBox.offsets[j] + sourceBox.widths[j]/2;

    // Fill the Chebyshev grid on the source box
    const vector<array<R,d>>& chebyshevGrid = context.GetChebyshevGrid();
    _sourceChebyshevGrid.resize( Pow<q,d>::val );
    for( size_t t=0; t<Pow<q,d>::val; ++t )
        for( size_t j=0; j<d; ++j )
            _sourceChebyshevGrid[t][j] = 
                p0[j] + chebyshevGrid[t][j]*sourceBox.widths[j];
}

template<typename R,size_t d,size_t q>
complex<R>
interpolative_nuft::PotentialField<R,d,q>::Evaluate( const array<R,d>& x ) const
{
    typedef complex<R> C;

#ifndef RELEASE
    for( size_t j=0; j<d; ++j )
    {
        if( x[j] < _myTargetBox.offsets[j] ||
            x[j] > _myTargetBox.offsets[j] + _myTargetBox.widths[j] )
        {
            throw std::runtime_error
                  ( "Tried to evaluate outside of potential range." );
        }
    }
#endif

    // Compute the lexographic position of the LRP to use for evaluation
    size_t k = 0;
    for( size_t j=0; j<d; ++j ) 
    {
        size_t owningIndex = size_t((x[j]-_myTargetBox.offsets[j])/_wA[j]);
        k += owningIndex << _log2TargetSubboxesUpToDim[j];
    }
    const LRP<R,d,q>& lrp = _LRPs[k];

    const Direction direction = _context.GetDirection();
    const R SignedTwoPi = ( direction==FORWARD ? -TwoPi : TwoPi );

    C potential = 0;
    for( size_t t=0; t<Pow<q,d>::val; ++t )
    {
        const R realWeight = lrp.weightGrid.RealWeight(t);
        const R imagWeight = lrp.weightGrid.ImagWeight(t);
        const C weight = C( realWeight, imagWeight );

        // Compute the dot product of the gridpoint with the target location
        R dot = 0;
        for( size_t j=0; j<d; ++j )
            dot += x[j]*_sourceChebyshevGrid[t][j];
        potential += ImagExp<R>( SignedTwoPi*dot )*weight;
    }
    return potential;
}

template<typename R,size_t d,size_t q>
inline const Box<R,d>&
interpolative_nuft::PotentialField<R,d,q>::GetMyTargetBox() const
{ return _myTargetBox; }

template<typename R,size_t d,size_t q>
inline size_t
interpolative_nuft::PotentialField<R,d,q>::GetNumSubboxes() const
{ return _LRPs.size(); }

template<typename R,size_t d,size_t q>
inline const array<R,d>&
interpolative_nuft::PotentialField<R,d,q>::GetSubboxWidths() const
{ return _wA; }

template<typename R,size_t d,size_t q>
inline const array<size_t,d>&
interpolative_nuft::PotentialField<R,d,q>::GetLog2SubboxesPerDim() const
{ return _log2TargetSubboxesPerDim; }

template<typename R,size_t d,size_t q>
inline const array<size_t,d>&
interpolative_nuft::PotentialField<R,d,q>::GetLog2SubboxesUpToDim() const
{ return _log2TargetSubboxesUpToDim; }

} // bfio

#endif // ifndef BFIO_INTERPOLATIVE_NUFT_POTENTIAL_FIELD_HPP
