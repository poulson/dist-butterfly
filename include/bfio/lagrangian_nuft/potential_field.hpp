/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of ButterflyFIO and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef BFIO_LAGRANGIAN_NUFT_POTENTIAL_FIELD_HPP
#define BFIO_LAGRANGIAN_NUFT_POTENTIAL_FIELD_HPP

#include <array>
#include <complex>
#include <string>
#include <vector>

#include "bfio/rfio/potential_field.hpp"
#include "bfio/lagrangian_nuft/ft_phases.hpp"

namespace bfio {

using std::array;
using std::complex;
using std::size_t;
using std::string;
using std::vector;

namespace lagrangian_nuft {
template<typename R,size_t d,size_t q>
class PotentialField
{
    const lagrangian_nuft::Context<R,d,q>& _nuftContext;
    const rfio::PotentialField<R,d,q> _rfioPotential;

public:
    PotentialField
    ( const lagrangian_nuft::Context<R,d,q>& context,
      const Box<R,d>& sourceBox,
      const Box<R,d>& myTargetBox,
      const array<size_t,d>& myTargetBoxCoords,
      const array<size_t,d>& log2TargetSubboxesPerDim,
      const WeightGridList<R,d,q>& weightGridList );

    // This is the point of the potential field
    complex<R> Evaluate( const array<R,d>& x ) const;

    const Amplitude<R,d>& GetAmplitude() const;
    const Phase<R,d>& GetPhase() const;
    const Box<R,d>& GetMyTargetBox() const;
    size_t GetNumSubboxes() const;
    const array<R,d>& GetSubboxWidths() const;
    const array<size_t,d>& GetMyTargetBoxCoords() const;
    const array<size_t,d>& GetLog2SubboxesPerDim() const;
    const array<size_t,d>& GetLog2SubboxesUpToDim() const;
    const rfio::PotentialField<R,d,q>& GetReducedFIOPotentialField() const;
};

template<typename R,size_t d,size_t q>
void PrintErrorEstimates
( MPI_Comm comm,
  const PotentialField<R,d,q>& u,
  const vector<Source<R,d>>& globalSources );

template<typename R,size_t d,size_t q>
void WriteVtkXmlPImageData
( MPI_Comm comm, 
  const size_t N,
  const Box<R,d>& targetBox,
  const PotentialField<R,d,q>& u,
  const string& basename );

template<typename R,size_t d,size_t q>
void WriteVtkXmlPImageData
( MPI_Comm comm, 
  const size_t N,
  const Box<R,d>& targetBox,
  const PotentialField<R,d,q>& u,
  const string& basename,
  const vector<Source<R,d>>& globalSources );

} // lagrangian_nuft

// Implementations

template<typename R,size_t d,size_t q>
lagrangian_nuft::PotentialField<R,d,q>::PotentialField
( const lagrangian_nuft::Context<R,d,q>& nuftContext,
  const Box<R,d>& sourceBox,
  const Box<R,d>& targetBox,
  const array<size_t,d>& myTargetBoxCoords,
  const array<size_t,d>& log2TargetSubboxesPerDim,
  const WeightGridList<R,d,q>& weightGridList )
: _nuftContext(nuftContext), 
  _rfioPotential
  ( nuftContext.GetReducedFIOContext(),
    UnitAmplitude<R,d>(),
    ( nuftContext.GetDirection()==FORWARD ? 
      (const FTPhase<R,d>&)lagrangian_nuft::ForwardFTPhase<R,d>() : 
      (const FTPhase<R,d>&)lagrangian_nuft::AdjointFTPhase<R,d>() ),
    sourceBox,
    targetBox,
    myTargetBoxCoords,
    log2TargetSubboxesPerDim,
    weightGridList )
{ }

template<typename R,size_t d,size_t q>
complex<R>
lagrangian_nuft::PotentialField<R,d,q>::Evaluate( const array<R,d>& x ) const
{ return _rfioPotential.Evaluate( x ); }

template<typename R,size_t d,size_t q>
inline const Amplitude<R,d>&
lagrangian_nuft::PotentialField<R,d,q>::GetAmplitude() const
{ return _rfioPotential.GetAmplitude(); }

template<typename R,size_t d,size_t q>
inline const Phase<R,d>&
lagrangian_nuft::PotentialField<R,d,q>::GetPhase() const
{ return _rfioPotential.GetPhase(); }

template<typename R,size_t d,size_t q>
inline const Box<R,d>&
lagrangian_nuft::PotentialField<R,d,q>::GetMyTargetBox() const
{ return _rfioPotential.GetMyTargetBox(); }

template<typename R,size_t d,size_t q>
inline size_t
lagrangian_nuft::PotentialField<R,d,q>::GetNumSubboxes() const
{ return _rfioPotential.GetNumSubboxes(); }

template<typename R,size_t d,size_t q>
inline const array<R,d>&
lagrangian_nuft::PotentialField<R,d,q>::GetSubboxWidths() const
{ return _rfioPotential.GetSubboxWidths(); }

template<typename R,size_t d,size_t q>
inline const array<size_t,d>&
lagrangian_nuft::PotentialField<R,d,q>::GetMyTargetBoxCoords() const
{ return _rfioPotential.GetMyTargetBoxCoords(); }

template<typename R,size_t d,size_t q>
inline const array<size_t,d>&
lagrangian_nuft::PotentialField<R,d,q>::GetLog2SubboxesPerDim() const
{ return _rfioPotential.GetLog2SubboxesPerDim(); }

template<typename R,size_t d,size_t q>
inline const array<size_t,d>&
lagrangian_nuft::PotentialField<R,d,q>::GetLog2SubboxesUpToDim() const
{ return _rfioPotential.GetLog2SubboxesUpToDim(); }

template<typename R,size_t d,size_t q>
const rfio::PotentialField<R,d,q>& 
lagrangian_nuft::PotentialField<R,d,q>::GetReducedFIOPotentialField() const
{ return _rfioPotential; }

template<typename R,size_t d,size_t q>
inline void 
lagrangian_nuft::PrintErrorEstimates
( MPI_Comm comm,
  const lagrangian_nuft::PotentialField<R,d,q>& u,
  const vector<Source<R,d>>& globalSources )
{
    rfio::PrintErrorEstimates    
    ( comm, u.GetReducedFIOPotentialField(), globalSources );
}

template<typename R,size_t d,size_t q>
inline void 
lagrangian_nuft::WriteVtkXmlPImageData
( MPI_Comm comm, 
  const size_t N,
  const Box<R,d>& targetBox,
  const lagrangian_nuft::PotentialField<R,d,q>& u,
  const string& basename )
{
    rfio::WriteVtkXmlPImageData
    ( comm, N, targetBox, u.GetReducedFIOPotentialField(), basename );
}

template<typename R,size_t d,size_t q>
inline void 
lagrangian_nuft::WriteVtkXmlPImageData
( MPI_Comm comm, 
  const size_t N,
  const Box<R,d>& targetBox,
  const lagrangian_nuft::PotentialField<R,d,q>& u,
  const string& basename,
  const vector<Source<R,d>>& globalSources )
{
    rfio::WriteVtkXmlPImageData
    ( comm, N, targetBox, u.GetReducedFIOPotentialField(), basename, 
      globalSources );
}

} // bfio

#endif // ifndef BFIO_LAGRANGIAN_NUFT_POTENTIAL_FIELD_HPP
