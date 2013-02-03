/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of ButterflyFIO and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef BFIO_INUFT_POTENTIAL_FIELD_HPP
#define BFIO_INUFT_POTENTIAL_FIELD_HPP

#include <array>
#include <complex>
#include <stdexcept>
#include <vector>

#include "bfio/structures/box.hpp"
#include "bfio/structures/constrained_htree_walker.hpp"
#include "bfio/structures/low_rank_potential.hpp"
#include "bfio/structures/weight_grid.hpp"
#include "bfio/structures/weight_grid_list.hpp"

#include "bfio/inuft/context.hpp"

#include "bfio/tools/special_functions.hpp"

namespace bfio {

using std::array;
using std::complex;
using std::size_t;
using std::vector;

namespace inuft {
template<typename R,size_t d,size_t q>
class PotentialField
{
    const Context<R,d,q>& _context;
    const Box<R,d> _sBox;
    const Box<R,d> _myTBox;
    const array<size_t,d> _log2TSubboxesPerDim;

    array<R,d> _wA;
    vector<array<R,d>> _sChebyshevGrid;
    array<size_t,d> _log2TSubboxesUpToDim;
    vector<LRP<R,d,q>> _LRPs;

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
 *         inuft::PotentialField from
 *         general_fio::PotentialField, but this was chosen as an alternative
 *         to coupling the two classes. Since LagrangianNUFT is a slight 
 *         specialization of the GeneralFIO approach, 
 *         lagrangian_nuft::PotentialField _is_ built from the GeneralFIO 
 *         potential field.
 */

template<typename R,size_t d,size_t q>
PotentialField<R,d,q>::PotentialField
( const Context<R,d,q>& context,
  const Box<R,d>& sBox,
  const Box<R,d>& myTBox,
  const array<size_t,d>& log2TSubboxesPerDim,
  const WeightGridList<R,d,q>& weightGridList )
: _context(context), _sBox(sBox), _myTBox(myTBox),
  _log2TSubboxesPerDim(log2TSubboxesPerDim)
{ 
    // Compute the widths of the target subboxes
    for( size_t j=0; j<d; ++j )
        _wA[j] = myTBox.widths[j] / (1<<log2TSubboxesPerDim[j]);

    // Compute the array of the partial sums
    _log2TSubboxesUpToDim[0] = 0;
    for( size_t j=1; j<d; ++j )
    {
        _log2TSubboxesUpToDim[j] = 
            _log2TSubboxesUpToDim[j-1] + log2TSubboxesPerDim[j-1];
    }

    // Figure out the size of our LRP vector by summing log2TSubboxesPerDim
    size_t log2TSubboxes = 0;
    for( size_t j=0; j<d; ++j )
        log2TSubboxes += log2TSubboxesPerDim[j];
    _LRPs.resize( 1<<log2TSubboxes );

    // The weightGridList is assumed to be ordered by the constrained 
    // HTree described by log2TSubboxesPerDim. We will unroll it
    // lexographically into the LRP vector.
    ConstrainedHTreeWalker<d> AWalker( log2TSubboxesPerDim );
    for( size_t tIndex=0; tIndex<_LRPs.size(); ++tIndex, AWalker.Walk() )
    {
        const array<size_t,d> A = AWalker.State();

        // Unroll the indices of A into its lexographic position
        size_t k=0;
        for( size_t j=0; j<d; ++j )
            k += A[j] << _log2TSubboxesUpToDim[j];

        // Now fill the k'th LRP index
        for( size_t j=0; j<d; ++j )
            _LRPs[k].x0[j] = myTBox.offsets[j] + (A[j]+0.5)*_wA[j];
        _LRPs[k].weightGrid = weightGridList[tIndex];
    }

    // Compute the source center
    array<R,d> p0;
    for( size_t j=0; j<d; ++j )
        p0[j] = sBox.offsets[j] + sBox.widths[j]/2;

    // Fill the Chebyshev grid on the source box
    const vector<array<R,d>>& chebyshevGrid = context.GetChebyshevGrid();
    _sChebyshevGrid.resize( Pow<q,d>::val );
    for( size_t t=0; t<Pow<q,d>::val; ++t )
        for( size_t j=0; j<d; ++j )
            _sChebyshevGrid[t][j] = p0[j] + chebyshevGrid[t][j]*sBox.widths[j];
}

template<typename R,size_t d,size_t q>
complex<R>
PotentialField<R,d,q>::Evaluate( const array<R,d>& x ) const
{
    typedef complex<R> C;

#ifndef RELEASE
    for( size_t j=0; j<d; ++j )
    {
        if( x[j] < _myTBox.offsets[j] ||
            x[j] > _myTBox.offsets[j] + _myTBox.widths[j] )
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
        size_t owningIndex = size_t((x[j]-_myTBox.offsets[j])/_wA[j]);
        k += owningIndex << _log2TSubboxesUpToDim[j];
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
            dot += x[j]*_sChebyshevGrid[t][j];
        potential += ImagExp<R>( SignedTwoPi*dot )*weight;
    }
    return potential;
}

template<typename R,size_t d,size_t q>
inline const Box<R,d>&
PotentialField<R,d,q>::GetMyTargetBox() const
{ return _myTBox; }

template<typename R,size_t d,size_t q>
inline size_t
PotentialField<R,d,q>::GetNumSubboxes() const
{ return _LRPs.size(); }

template<typename R,size_t d,size_t q>
inline const array<R,d>&
PotentialField<R,d,q>::GetSubboxWidths() const
{ return _wA; }

template<typename R,size_t d,size_t q>
inline const array<size_t,d>&
PotentialField<R,d,q>::GetLog2SubboxesPerDim() const
{ return _log2TSubboxesPerDim; }

template<typename R,size_t d,size_t q>
inline const array<size_t,d>&
PotentialField<R,d,q>::GetLog2SubboxesUpToDim() const
{ return _log2TSubboxesUpToDim; }

} // inuft
} // bfio

#endif // ifndef BFIO_INUFT_POTENTIAL_FIELD_HPP
