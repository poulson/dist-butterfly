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
#ifndef BFIO_STRUCTURES_POTENTIAL_FIELD_HPP
#define BFIO_STRUCTURES_POTENTIAL_FIELD_HPP 1

#include <stdexcept>

#include "bfio/structures/data.hpp"
#include "bfio/structures/htree_walker.hpp"
#include "bfio/structures/amplitude_functor.hpp"
#include "bfio/structures/phase_functor.hpp"

namespace bfio {

template<typename R,unsigned d,unsigned q>
struct LRP
{
    Array<R,d> x0;
    WeightGrid<R,d,q> weightGrid;
};

template<typename R,unsigned d,unsigned q>
class PotentialField
{
    const Box<R,d> _spatialBox;
    const Box<R,d> _freqBox;
    const Array<unsigned,d> _log2SpatialSubboxesPerDim;
    const AmplitudeFunctor<R,d>& _Amp;
    const PhaseFunctor<R,d>& _Phi;
    const Context<R,d,q>& _context;

    Array<R,d> _wA;
    Array<R,d> _p0;
    Array<unsigned,d> _log2SpatialSubboxesUpToDim;
    std::vector< LRP<R,d,q> > _LRPs;

public:
    PotentialField
    ( const Box<R,d>& spatialBox,
      const Box<R,d>& freqBox,
      const Array<unsigned,d>& log2SpatialSubboxesPerDim,
      const AmplitudeFunctor<R,d>& Amp, 
      const PhaseFunctor<R,d>& Phi,
      const Context<R,d,q>& context,
      const WeightGridList<R,d,q>& weightGridList );

    // This is the point of the potential field
    std::complex<R> Evaluate( const Array<R,d>& x ) const;

    const Box<R,d>& GetBox() const;
    unsigned GetNumSubboxes() const;
    const Array<R,d>& GetSubboxWidths() const;
    const Array<unsigned,d>& GetLog2SubboxesPerDim() const;
    const Array<unsigned,d>& GetLog2SubboxesUpToDim() const;
};

} // bfio

// Implementations
namespace bfio {

template<typename R,unsigned d,unsigned q>
PotentialField<R,d,q>::PotentialField
( const Box<R,d>& spatialBox,
  const Box<R,d>& freqBox,
  const Array<unsigned,d>& log2SpatialSubboxesPerDim,
  const AmplitudeFunctor<R,d>& Amp, 
  const PhaseFunctor<R,d>& Phi,
  const Context<R,d,q>& context,
  const WeightGridList<R,d,q>& weightGridList )
: _spatialBox(spatialBox), _freqBox(freqBox), 
  _log2SpatialSubboxesPerDim(log2SpatialSubboxesPerDim), 
  _Amp(Amp), _Phi(Phi), _context(context)
{ 
    // Compute the widths of the spatial subboxes and the freq center
    for( unsigned j=0; j<d; ++j )
        _wA[j] = spatialBox.widths[j] / (1<<log2SpatialSubboxesPerDim[j]);
    for( unsigned j=0; j<d; ++j )
        _p0[j] = freqBox.offsets[j] + 0.5*freqBox.widths[j];

    // Compute the array of the partial sums
    _log2SpatialSubboxesUpToDim[0] = 0;
    for( unsigned j=1; j<d; ++j )
    {
        _log2SpatialSubboxesUpToDim[j] = 
            _log2SpatialSubboxesUpToDim[j-1]+log2SpatialSubboxesPerDim[j-1];
    }

    // Figure out the size of our LRP vector by summing 
    // log2SpatialSubboxesPerDim
    unsigned log2SpatialSubboxes = 0;
    for( unsigned j=0; j<d; ++j )
        log2SpatialSubboxes += log2SpatialSubboxesPerDim[j];
    _LRPs.resize( 1<<log2SpatialSubboxes );

    // The weightGridList is assumed to be ordered by the constrained 
    // HTree described by log2SpatialSubboxesPerDim. We will unroll it 
    // lexographically into the _LRPs vector.
    ConstrainedHTreeWalker<d> AWalker( log2SpatialSubboxesPerDim );
    for( unsigned i=0; i<_LRPs.size(); ++i, AWalker.Walk() )
    {
        const Array<unsigned,d> A = AWalker.State();

        // Unroll the indices of A into its lexographic integer
        unsigned k=0; 
        for( unsigned j=0; j<d; ++j )
            k += A[j] << _log2SpatialSubboxesUpToDim[j];

        // Now fill the k'th LRP index
        for( unsigned j=0; j<d; ++j )
            _LRPs[k].x0[j] = spatialBox.offsets[j] + (A[j]+0.5)*_wA[j];
        _LRPs[k].weightGrid = weightGridList[i];
    }
}

template<typename R,unsigned d,unsigned q>
inline std::complex<R>
PotentialField<R,d,q>::Evaluate( const Array<R,d>& x ) const
{
    typedef std::complex<R> C;

    for( unsigned j=0; j<d; ++j )
    {
        if( x[j] < _spatialBox.offsets[j] || 
            x[j] > _spatialBox.offsets[j]+_spatialBox.widths[j] )
        {
            throw std::runtime_error
                  ( "Tried to evaluate outside of potential range." );
        }
    }

    // Compute the lexographic index of the LRP to use for evaluation
    unsigned k = 0;
    for( unsigned j=0; j<d; ++j )
    {
        unsigned owningIndex = 
            static_cast<unsigned>((x[j]-_spatialBox.offsets[j])/_wA[j]);
        k += owningIndex << _log2SpatialSubboxesUpToDim[j];
    }

    // Convert x to the reference domain of [-1/2,+1/2]^d for box k
    const LRP<R,d,q>& lrp = _LRPs[k];
    Array<R,d> xRef;
    for( unsigned j=0; j<d; ++j )
        xRef[j] = (x[j]-lrp.x0[j])/_wA[j];

    // Grab a reference to the Chebyshev grid
    const std::vector< Array<R,d> > chebyshevGrid = _context.GetChebyshevGrid();

    C value(0.,0.);
    for( unsigned t=0; t<Pow<q,d>::val; ++t )
    {
        // Construct the t'th translated Chebyshev gridpoint
        Array<R,d> xt;
        for( unsigned j=0; j<d; ++j )
            xt[j] = lrp.x0[j] + _wA[j]*chebyshevGrid[t][j];
        C beta = ImagExp( TwoPi*_Phi(xt,_p0) );
        value += _context.Lagrange(t,xRef) * lrp.weightGrid[t] / beta;
    }
    value *= ImagExp( TwoPi*_Phi(x,_p0) );
    return value;
}

template<typename R,unsigned d,unsigned q>
inline const Box<R,d>&
PotentialField<R,d,q>::GetBox() const
{ return _spatialBox; }

template<typename R,unsigned d,unsigned q>
inline unsigned
PotentialField<R,d,q>::GetNumSubboxes() const
{ return _LRPs.size(); }

template<typename R,unsigned d,unsigned q>
inline const Array<R,d>&
PotentialField<R,d,q>::GetSubboxWidths() const
{ return _wA; }

template<typename R,unsigned d,unsigned q>
inline const Array<unsigned,d>&
PotentialField<R,d,q>::GetLog2SubboxesPerDim() const
{ return _log2SpatialSubboxesPerDim; }

template<typename R,unsigned d,unsigned q>
inline const Array<unsigned,d>&
PotentialField<R,d,q>::GetLog2SubboxesUpToDim() const
{ return _log2SpatialSubboxesUpToDim; }

} // bfio

#endif // BFIO_STRUCTURES_POTENTIAL_FIELD_HPP

