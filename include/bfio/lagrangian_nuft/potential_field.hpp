/*
   ButterflyFIO: a distributed-memory fast algorithm for applying FIOs.
   Copyright (C) 2010-2011 Jack Poulson <jack.poulson@gmail.com>
 
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
#ifndef BFIO_LAGRANGIAN_NUFT_POTENTIAL_FIELD_HPP
#define BFIO_LAGRANGIAN_NUFT_POTENTIAL_FIELD_HPP 1

#include "bfio/general_fio/potential_field.hpp"
#include "bfio/lagrangian_nuft/dot_product.hpp"

namespace bfio {

namespace lagrangian_nuft {
template<typename R,std::size_t d,std::size_t q>
class PotentialField
{
    const lagrangian_nuft::Context<R,d,q>& _nuftContext;
    const lagrangian_nuft::DotProduct<R,d> _dotProduct;
    const general_fio::PotentialField<R,d,q> _generalPotential;

public:
    PotentialField
    ( const lagrangian_nuft::Context<R,d,q>& context,
      const Box<R,d>& sourceBox,
      const Box<R,d>& targetBox,
      const Array<std::size_t,d>& myTargetBoxCoords,
      const Array<std::size_t,d>& log2TargetSubboxesPerDim,
      const WeightGridList<R,d,q>& weightGridList );

    // This is the point of the potential field
    std::complex<R> Evaluate( const Array<R,d>& x ) const;

    const Box<R,d>& GetBox() const;
    std::size_t GetNumSubboxes() const;
    const Array<R,d>& GetSubboxWidths() const;
    const Array<std::size_t,d>& GetMyTargetBoxCoords() const;
    const Array<std::size_t,d>& GetLog2SubboxesPerDim() const;
    const Array<std::size_t,d>& GetLog2SubboxesUpToDim() const;
    const general_fio::PotentialField<R,d,q>& GetGeneralPotentialField() const;
};

template<std::size_t d,std::size_t q>
void WriteVtkXmlPImageData
( MPI_Comm comm, 
  const std::size_t N,
  const PotentialField<double,d,q>& u,
  const std::string& basename );

} // lagrangian_nuft

// Implementations

template<typename R,std::size_t d,std::size_t q>
lagrangian_nuft::PotentialField<R,d,q>::PotentialField
( const lagrangian_nuft::Context<R,d,q>& nuftContext,
  const Box<R,d>& sourceBox,
  const Box<R,d>& targetBox,
  const Array<std::size_t,d>& myTargetBoxCoords,
  const Array<std::size_t,d>& log2TargetSubboxesPerDim,
  const WeightGridList<R,d,q>& weightGridList )
: _nuftContext(nuftContext), 
  _dotProduct(),
  _generalPotential
  ( nuftContext.GetGeneralContext(),
    this->_dotProduct,
    sourceBox,
    targetBox,
    myTargetBoxCoords,
    log2TargetSubboxesPerDim,
    weightGridList )
{ }

template<typename R,std::size_t d,std::size_t q>
std::complex<R>
lagrangian_nuft::PotentialField<R,d,q>::Evaluate( const Array<R,d>& x ) const
{ return _generalPotential.Evaluate( x ); }

template<typename R,std::size_t d,std::size_t q>
inline const Box<R,d>&
lagrangian_nuft::PotentialField<R,d,q>::GetBox() const
{ return _generalPotential.GetBox(); }

template<typename R,std::size_t d,std::size_t q>
inline std::size_t
lagrangian_nuft::PotentialField<R,d,q>::GetNumSubboxes() const
{ return _generalPotential.GetNumSubboxes(); }

template<typename R,std::size_t d,std::size_t q>
inline const Array<R,d>&
lagrangian_nuft::PotentialField<R,d,q>::GetSubboxWidths() const
{ return _generalPotential.GetSubboxWidths(); }

template<typename R,std::size_t d,std::size_t q>
inline const Array<std::size_t,d>&
lagrangian_nuft::PotentialField<R,d,q>::GetMyTargetBoxCoords() const
{ return _generalPotential.GetMyTargetBoxCoords(); }

template<typename R,std::size_t d,std::size_t q>
inline const Array<std::size_t,d>&
lagrangian_nuft::PotentialField<R,d,q>::GetLog2SubboxesPerDim() const
{ return _generalPotential.GetLog2SubboxesPerDim(); }

template<typename R,std::size_t d,std::size_t q>
inline const Array<std::size_t,d>&
lagrangian_nuft::PotentialField<R,d,q>::GetLog2SubboxesUpToDim() const
{ return _generalPotential.GetLog2SubboxesUpToDim(); }

template<typename R,std::size_t d,std::size_t q>
const general_fio::PotentialField<R,d,q>& 
lagrangian_nuft::PotentialField<R,d,q>::GetGeneralPotentialField() const
{ return _generalPotential; }

template<std::size_t d,std::size_t q>
inline void 
lagrangian_nuft::WriteVtkXmlPImageData
( MPI_Comm comm, 
  const std::size_t N,
  const lagrangian_nuft::PotentialField<double,d,q>& u,
  const std::string& basename )
{
    general_fio::WriteVtkXmlPImageData
    ( comm, N, u.GetGeneralPotentialField(), basename );
}

} // bfio

#endif // BFIO_LAGRANGIAN_NUFT_POTENTIAL_FIELD_HPP

