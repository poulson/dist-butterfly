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
#ifndef BFIO_INTERPOLATIVE_NUFT_POTENTIAL_FIELD_HPP
#define BFIO_INTERPOLATIVE_NUFT_POTENTIAL_FIELD_HPP 1

#include <complex>
#include <stdexcept>
#include <vector>

#include "bfio/structures/array.hpp"
#include "bfio/structures/box.hpp"
#include "bfio/structures/constrained_htree_walker.hpp"
#include "bfio/structures/low_rank_potential.hpp"
#include "bfio/structures/weight_grid.hpp"
#include "bfio/structures/weight_grid_list.hpp"

#include "bfio/interpolative_nuft/context.hpp"

#include "bfio/tools/special_functions.hpp"

namespace bfio {

namespace interpolative_nuft {
template<typename R,std::size_t d,std::size_t q>
class PotentialField
{
    const interpolative_nuft::Context<R,d,q>& _context;
    const Box<R,d> _sourceBox;
    const Box<R,d> _targetBox;
    const Array<std::size_t,d> _log2TargetSubboxesPerDim;

    Array<R,d> _wA;
    std::vector< Array<R,d> > _sourceChebyshevGrid;
    Array<std::size_t,d> _log2TargetSubboxesUpToDim;
    std::vector< LRP<R,d,q> > _LRPs;

public:
    PotentialField
    ( const interpolative_nuft::Context<R,d,q>& context,
      const Box<R,d>& sourceBox,
      const Box<R,d>& targetBox,
      const Array<std::size_t,d>& log2TargetSubboxesPerDim,
      const WeightGridList<R,d,q>& weightGridList );

    std::complex<R> Evaluate( const Array<R,d>& x ) const;

    const Box<R,d>& GetBox() const;
    std::size_t GetNumSubboxes() const;
    const Array<R,d>& GetSubboxWidths() const;
    const Array<std::size_t,d>& GetLog2SubboxesPerDim() const;
    const Array<std::size_t,d>& GetLog2SubboxesUpToDim() const;
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

template<typename R,std::size_t d,std::size_t q>
interpolative_nuft::PotentialField<R,d,q>::PotentialField
( const interpolative_nuft::Context<R,d,q>& context,
  const Box<R,d>& sourceBox,
  const Box<R,d>& targetBox,
  const Array<std::size_t,d>& log2TargetSubboxesPerDim,
  const WeightGridList<R,d,q>& weightGridList )
: _context(context), _sourceBox(sourceBox), _targetBox(targetBox),
  _log2TargetSubboxesPerDim(log2TargetSubboxesPerDim)
{ 
    // Compute the widths of the target subboxes
    for( std::size_t j=0; j<d; ++j )
        _wA[j] = targetBox.widths[j] / (1<<log2TargetSubboxesPerDim[j]);

    // Compute the array of the partial sums
    _log2TargetSubboxesUpToDim[0] = 0;
    for( std::size_t j=1; j<d; ++j )
    {
        _log2TargetSubboxesUpToDim[j] = 
            _log2TargetSubboxesUpToDim[j-1] + log2TargetSubboxesPerDim[j-1];
    }

    // Figure out the size of our LRP vector by summing log2TargetSubboxesPerDim
    std::size_t log2TargetSubboxes = 0;
    for( std::size_t j=0; j<d; ++j )
        log2TargetSubboxes += log2TargetSubboxesPerDim[j];
    _LRPs.resize( 1<<log2TargetSubboxes );

    // The weightGridList is assumed to be ordered by the constrained 
    // HTree described by log2TargetSubboxesPerDim. We will unroll it
    // lexographically into the LRP vector.
    ConstrainedHTreeWalker<d> AWalker( log2TargetSubboxesPerDim );
    for( std::size_t targetIndex=0; 
         targetIndex<_LRPs.size();
         ++targetIndex, AWalker.Walk() )
    {
        const Array<std::size_t,d> A = AWalker.State();

        // Unroll the indices of A into its lexographic position
        std::size_t k=0;
        for( std::size_t j=0; j<d; ++j )
            k += A[j] << _log2TargetSubboxesUpToDim[j];

        // Now fill the k'th LRP index
        for( std::size_t j=0; j<d; ++j )
            _LRPs[k].x0[j] = targetBox.offsets[j] + (A[j]+0.5)*_wA[j];
        _LRPs[k].weightGrid = weightGridList[targetIndex];
    }

    // Compute the source center
    Array<R,d> p0;
    for( std::size_t j=0; j<d; ++j )
        p0[j] = sourceBox.offsets[j] + sourceBox.widths[j]/2;

    // Fill the Chebyshev grid on the source box
    const std::vector< Array<R,d> >& chebyshevGrid = context.GetChebyshevGrid();
    _sourceChebyshevGrid.resize( Pow<q,d>::val );
    for( std::size_t t=0; t<Pow<q,d>::val; ++t )
        for( std::size_t j=0; j<d; ++j )
            _sourceChebyshevGrid[t][j] = 
                p0[j] + chebyshevGrid[t][j]*sourceBox.widths[j];
}

template<typename R,std::size_t d,std::size_t q>
std::complex<R>
interpolative_nuft::PotentialField<R,d,q>::Evaluate( const Array<R,d>& x ) const
{
    typedef std::complex<R> C;

#ifndef RELEASE
    for( std::size_t j=0; j<d; ++j )
    {
        if( x[j] < _targetBox.offsets[j] ||
            x[j] > _targetBox.offsets[j] + _targetBox.widths[j] )
        {
            throw std::runtime_error
                  ( "Tried to evaluate outside of potential range." );
        }
    }
#endif

    // Compute the lexographic position of the LRP to use for evaluation
    std::size_t k = 0;
    for( std::size_t j=0; j<d; ++j ) 
    {
        std::size_t owningIndex = 
            static_cast<std::size_t>((x[j]-_targetBox.offsets[j])/_wA[j]);
        k += owningIndex << _log2TargetSubboxesUpToDim[j];
    }
    const LRP<R,d,q>& lrp = _LRPs[k];

    C potential = 0;
    for( std::size_t t=0; t<Pow<q,d>::val; ++t )
    {
        const R realWeight = lrp.weightGrid.RealWeight(t);
        const R imagWeight = lrp.weightGrid.ImagWeight(t);
        const C weight = C( realWeight, imagWeight );

        // Compute the dot product of the gridpoint with the target location
        R dot = 0;
        for( std::size_t j=0; j<d; ++j )
            dot += x[j]*_sourceChebyshevGrid[t][j];
        potential += ImagExp<R>( TwoPi*dot )*weight;
    }
    return potential;
}

template<typename R,std::size_t d,std::size_t q>
inline const Box<R,d>&
interpolative_nuft::PotentialField<R,d,q>::GetBox() const
{ return _targetBox; }

template<typename R,std::size_t d,std::size_t q>
inline std::size_t
interpolative_nuft::PotentialField<R,d,q>::GetNumSubboxes() const
{ return _LRPs.size(); }

template<typename R,std::size_t d,std::size_t q>
inline const Array<R,d>&
interpolative_nuft::PotentialField<R,d,q>::GetSubboxWidths() const
{ return _wA; }

template<typename R,std::size_t d,std::size_t q>
inline const Array<std::size_t,d>&
interpolative_nuft::PotentialField<R,d,q>::GetLog2SubboxesPerDim() const
{ return _log2TargetSubboxesPerDim; }

template<typename R,std::size_t d,std::size_t q>
inline const Array<std::size_t,d>&
interpolative_nuft::PotentialField<R,d,q>::GetLog2SubboxesUpToDim() const
{ return _log2TargetSubboxesUpToDim; }

} // bfio

#endif // BFIO_INTERPOLATIVE_NUFT_POTENTIAL_FIELD_HPP

