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
#ifndef BFIO_NUFT_POTENTIAL_FIELD_HPP
#define BFIO_NUFT_POTENTIAL_FIELD_HPP 1

#include "bfio/general_fio/potential_field.hpp"
#include "bfio/nuft/dot_product.hpp"

namespace bfio {

namespace nuft {
template<typename R,std::size_t d,std::size_t q>
class PotentialField
{
    const nuft::Context<R,d,q>& _nuftContext;
    const nuft::DotProduct<R,d>& _dotProduct;
    const general_fio::PotentialField<R,d,q>& _generalPotential;

public:
    PotentialField
    ( const nuft::Context<R,d,q>& context,
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
} // nuft

// Implementations

template<typename R,std::size_t d,std::size_t q>
nuft::PotentialField<R,d,q>::PotentialField
( const nuft::Context<R,d,q>& context,
  const Box<R,d>& sourceBox,
  const Box<R,d>& targetBox,
  const Array<std::size_t,d>& log2TargetSubboxesPerDim,
  const WeightGridList<R,d,q>& weightGridList )
: _context(context), 
  _dotProduct(),
  _generalPotential
  ( context.GeneralFIOContext(),
    this->_dotProduct,
    sourceBox,
    targetBox,
    log2TargetSubboxesPerDim,
    weightGridList )
{ }

template<typename R,std::size_t d,std::size_t q>
std::complex<R>
nuft::PotentialField<R,d,q>::Evaluate( const Array<R,d>& x ) const
{ return _generalPotential->Evaluate( x ); }

template<typename R,std::size_t d,std::size_t q>
inline const Box<R,d>&
nuft::PotentialField<R,d,q>::GetBox() const
{ return _generalPotential->GetBox(); }

template<typename R,std::size_t d,std::size_t q>
inline std::size_t
nuft::PotentialField<R,d,q>::GetNumSubboxes() const
{ return _generalPotential.GetNumSubboxes(); }

template<typename R,std::size_t d,std::size_t q>
inline const Array<R,d>&
nuft::PotentialField<R,d,q>::GetSubboxWidths() const
{ return _generalPotential.GetSubboxWidths(); }

template<typename R,std::size_t d,std::size_t q>
inline const Array<std::size_t,d>&
nuft::PotentialField<R,d,q>::GetLog2SubboxesPerDim() const
{ return _generalPotential.GetLog2SubboxesPerDim(); }

template<typename R,std::size_t d,std::size_t q>
inline const Array<std::size_t,d>&
nuft::PotentialField<R,d,q>::GetLog2SubboxesUpToDim() const
{ return _generalPotential.GetLog2SubboxesUpToDim(); }

} // bfio

#endif // BFIO_NUFT_POTENTIAL_FIELD_HPP

