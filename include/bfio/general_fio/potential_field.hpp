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
#ifndef BFIO_GENERAL_FIO_POTENTIAL_FIELD_HPP
#define BFIO_GENERAL_FIO_POTENTIAL_FIELD_HPP 1

#include <stdexcept>
#include <complex>
#include <vector>

#include "bfio/structures/array.hpp"
#include "bfio/structures/box.hpp"
#include "bfio/structures/constrained_htree_walker.hpp"
#include "bfio/structures/weight_grid.hpp"
#include "bfio/structures/weight_grid_list.hpp"

#include "bfio/general_fio/context.hpp"

#include "bfio/functors/amplitude_functor.hpp"
#include "bfio/functors/phase_functor.hpp"
#include "bfio/tools/special_functions.hpp"

namespace bfio {
namespace general_fio {

template<typename R,std::size_t d,std::size_t q>
struct LRP
{
    Array<R,d> x0;
    WeightGrid<R,d,q> weightGrid;
};

template<typename R,std::size_t d,std::size_t q>
class PotentialField
{
    const general_fio::Context<R,d,q>& _context;
    const PhaseFunctor<R,d>& _Phi;
    const Box<R,d> _sourceBox;
    const Box<R,d> _targetBox;
    const Array<std::size_t,d> _log2TargetSubboxesPerDim;

    Array<R,d> _wA;
    Array<R,d> _p0;
    Array<std::size_t,d> _log2TargetSubboxesUpToDim;
    std::vector< LRP<R,d,q> > _LRPs;

public:
    PotentialField
    ( const general_fio::Context<R,d,q>& context,
      const PhaseFunctor<R,d>& Phi,
      const Box<R,d>& sourceBox,
      const Box<R,d>& targetBox,
      const Array<std::size_t,d>& log2TargetSubboxesPerDim,
      const WeightGridList<R,d,q>& weightGridList );

    // This is the point of the potential field
    std::complex<R> Evaluate( const Array<R,d>& x ) const;

    const Box<R,d>& GetBox() const;
    std::size_t GetNumSubboxes() const;
    const Array<R,d>& GetSubboxWidths() const;
    const Array<std::size_t,d>& GetLog2SubboxesPerDim() const;
    const Array<std::size_t,d>& GetLog2SubboxesUpToDim() const;
};

// Implementations

template<typename R,std::size_t d,std::size_t q>
PotentialField<R,d,q>::PotentialField
( const general_fio::Context<R,d,q>& context,
  const PhaseFunctor<R,d>& Phi,
  const Box<R,d>& sourceBox,
  const Box<R,d>& targetBox,
  const Array<std::size_t,d>& log2TargetSubboxesPerDim,
  const WeightGridList<R,d,q>& weightGridList )
: _context(context), _Phi(Phi), _sourceBox(sourceBox), _targetBox(targetBox),
  _log2TargetSubboxesPerDim(log2TargetSubboxesPerDim)
{ 
    // Compute the widths of the target subboxes and the source center
    for( std::size_t j=0; j<d; ++j )
        _wA[j] = targetBox.widths[j] / (1<<log2TargetSubboxesPerDim[j]);
    for( std::size_t j=0; j<d; ++j )
        _p0[j] = sourceBox.offsets[j] + 0.5*sourceBox.widths[j];

    // Compute the array of the partial sums
    _log2TargetSubboxesUpToDim[0] = 0;
    for( std::size_t j=1; j<d; ++j )
    {
        _log2TargetSubboxesUpToDim[j] = 
            _log2TargetSubboxesUpToDim[j-1]+log2TargetSubboxesPerDim[j-1];
    }

    // Figure out the size of our LRP vector by summing log2TargetSubboxesPerDim
    std::size_t log2TargetSubboxes = 0;
    for( std::size_t j=0; j<d; ++j )
        log2TargetSubboxes += log2TargetSubboxesPerDim[j];
    _LRPs.resize( 1<<log2TargetSubboxes );

    // The weightGridList is assumed to be ordered by the constrained 
    // HTree described by log2TargetSubboxesPerDim. We will unroll it 
    // lexographically into the _LRPs vector.
    ConstrainedHTreeWalker<d> AWalker( log2TargetSubboxesPerDim );
    for( std::size_t targetIndex=0; 
         targetIndex<_LRPs.size(); 
         ++targetIndex, AWalker.Walk() )
    {
        const Array<std::size_t,d> A = AWalker.State();

        // Unroll the indices of A into its lexographic integer
        std::size_t k=0; 
        for( std::size_t j=0; j<d; ++j )
            k += A[j] << _log2TargetSubboxesUpToDim[j];

        // Now fill the k'th LRP index
        for( std::size_t j=0; j<d; ++j )
            _LRPs[k].x0[j] = targetBox.offsets[j] + (A[j]+0.5)*_wA[j];
        _LRPs[k].weightGrid = weightGridList[targetIndex];
    }
}

template<typename R,std::size_t d,std::size_t q>
std::complex<R>
PotentialField<R,d,q>::Evaluate( const Array<R,d>& x ) const
{
    typedef std::complex<R> C;

    for( std::size_t j=0; j<d; ++j )
    {
        if( x[j] < _targetBox.offsets[j] || 
            x[j] > _targetBox.offsets[j]+_targetBox.widths[j] )
        {
            throw std::runtime_error
                  ( "Tried to evaluate outside of potential range." );
        }
    }

    // Compute the lexographic index of the LRP to use for evaluation
    std::size_t k = 0;
    for( std::size_t j=0; j<d; ++j )
    {
        std::size_t owningIndex = 
            static_cast<std::size_t>((x[j]-_targetBox.offsets[j])/_wA[j]);
        k += owningIndex << _log2TargetSubboxesUpToDim[j];
    }

    // Convert x to the reference domain of [-1/2,+1/2]^d for box k
    const LRP<R,d,q>& lrp = _LRPs[k];
    Array<R,d> xRef;
    for( std::size_t j=0; j<d; ++j )
        xRef[j] = (x[j]-lrp.x0[j])/_wA[j];

    const std::vector< Array<R,d> > chebyshevGrid = _context.GetChebyshevGrid();
    R realValue = 0;
    R imagValue = 0;
    for( std::size_t t=0; t<Pow<q,d>::val; ++t )
    {
        // Construct the t'th translated Chebyshev gridpoint
        Array<R,d> xt;
        for( std::size_t j=0; j<d; ++j )
            xt[j] = lrp.x0[j] + _wA[j]*chebyshevGrid[t][j];

        // TODO: Batch these together
        const C beta = ImagExp<R>( -TwoPi*_Phi(xt,_p0) );
        const R lambda = _context.Lagrange(t,xRef);
        const R realWeight = lrp.weightGrid.RealWeight(t);
        const R imagWeight = lrp.weightGrid.ImagWeight(t);
        realValue += lambda*(realWeight*real(beta)-imagWeight*imag(beta));
        imagValue += lambda*(imagWeight*real(beta)+realWeight*imag(beta));
    }
    const C beta = ImagExp<R>( TwoPi*_Phi(x,_p0) );
    realValue = realValue*real(beta)-imagValue*imag(beta);
    imagValue = imagValue*real(beta)+realValue*imag(beta);
    return C( realValue, imagValue );
}

template<typename R,std::size_t d,std::size_t q>
inline const Box<R,d>&
PotentialField<R,d,q>::GetBox() const
{ return _targetBox; }

template<typename R,std::size_t d,std::size_t q>
inline std::size_t
PotentialField<R,d,q>::GetNumSubboxes() const
{ return _LRPs.size(); }

template<typename R,std::size_t d,std::size_t q>
inline const Array<R,d>&
PotentialField<R,d,q>::GetSubboxWidths() const
{ return _wA; }

template<typename R,std::size_t d,std::size_t q>
inline const Array<std::size_t,d>&
PotentialField<R,d,q>::GetLog2SubboxesPerDim() const
{ return _log2TargetSubboxesPerDim; }

template<typename R,std::size_t d,std::size_t q>
inline const Array<std::size_t,d>&
PotentialField<R,d,q>::GetLog2SubboxesUpToDim() const
{ return _log2TargetSubboxesUpToDim; }

} // general_fio
} // bfio

#endif // BFIO_GENERAL_FIO_POTENTIAL_FIELD_HPP

