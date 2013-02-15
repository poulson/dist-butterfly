/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of ButterflyFIO and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef BFIO_LNUFT_POTENTIAL_FIELD_HPP
#define BFIO_LNUFT_POTENTIAL_FIELD_HPP

#include <array>
#include <complex>
#include <string>
#include <vector>

#include "bfio/rfio/potential_field.hpp"
#include "bfio/lnuft/ft_phases.hpp"

namespace bfio {

using std::array;
using std::complex;
using std::size_t;
using std::string;
using std::vector;

namespace lnuft {

template<typename R,size_t d,size_t q>
class PotentialField
{
    const Context<R,d,q>& nuftContext_;
    const rfio::PotentialField<R,d,q> rfioPotential_;

public:
    PotentialField
    ( const Context<R,d,q>& context,
      const Box<R,d>& sBox,
      const Box<R,d>& myTBox,
      const array<size_t,d>& myTBoxCoords,
      const array<size_t,d>& log2TSubboxesPerDim,
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
    const rfio::PotentialField<R,d,q>& GetRFIOPotentialField() const;
};

template<typename R,size_t d,size_t q>
void PrintErrorEstimates
( MPI_Comm comm,
  const PotentialField<R,d,q>& u,
  const vector<Source<R,d>>& sources );

template<typename R,size_t d,size_t q>
void WriteImage
( MPI_Comm comm, 
  const size_t N,
  const Box<R,d>& tBox,
  const PotentialField<R,d,q>& u,
  const string& basename );

template<typename R,size_t d,size_t q>
void WriteImage
( MPI_Comm comm, 
  const size_t N,
  const Box<R,d>& tBox,
  const PotentialField<R,d,q>& u,
  const string& basename,
  const vector<Source<R,d>>& sources );

// Implementations

template<typename R,size_t d,size_t q>
inline
PotentialField<R,d,q>::PotentialField
( const Context<R,d,q>& nuftContext,
  const Box<R,d>& sBox,
  const Box<R,d>& tBox,
  const array<size_t,d>& myTBoxCoords,
  const array<size_t,d>& log2TSubboxesPerDim,
  const WeightGridList<R,d,q>& weightGridList )
: nuftContext_(nuftContext), 
  rfioPotential_
  ( nuftContext.GetRFIOContext(),
    UnitAmplitude<R,d>(),
    ( nuftContext.GetDirection()==FORWARD ? 
      (const FTPhase<R,d>&)ForwardFTPhase<R,d>() : 
      (const FTPhase<R,d>&)AdjointFTPhase<R,d>() ),
    sBox,
    tBox,
    myTBoxCoords,
    log2TSubboxesPerDim,
    weightGridList )
{ }

template<typename R,size_t d,size_t q>
inline complex<R>
PotentialField<R,d,q>::Evaluate( const array<R,d>& x ) const
{ return rfioPotential_.Evaluate( x ); }

template<typename R,size_t d,size_t q>
inline const Amplitude<R,d>&
PotentialField<R,d,q>::GetAmplitude() const
{ return rfioPotential_.GetAmplitude(); }

template<typename R,size_t d,size_t q>
inline const Phase<R,d>&
PotentialField<R,d,q>::GetPhase() const
{ return rfioPotential_.GetPhase(); }

template<typename R,size_t d,size_t q>
inline const Box<R,d>&
PotentialField<R,d,q>::GetMyTargetBox() const
{ return rfioPotential_.GetMyTargetBox(); }

template<typename R,size_t d,size_t q>
inline size_t
PotentialField<R,d,q>::GetNumSubboxes() const
{ return rfioPotential_.GetNumSubboxes(); }

template<typename R,size_t d,size_t q>
inline const array<R,d>&
PotentialField<R,d,q>::GetSubboxWidths() const
{ return rfioPotential_.GetSubboxWidths(); }

template<typename R,size_t d,size_t q>
inline const array<size_t,d>&
PotentialField<R,d,q>::GetMyTargetBoxCoords() const
{ return rfioPotential_.GetMyTargetBoxCoords(); }

template<typename R,size_t d,size_t q>
inline const array<size_t,d>&
PotentialField<R,d,q>::GetLog2SubboxesPerDim() const
{ return rfioPotential_.GetLog2SubboxesPerDim(); }

template<typename R,size_t d,size_t q>
inline const array<size_t,d>&
PotentialField<R,d,q>::GetLog2SubboxesUpToDim() const
{ return rfioPotential_.GetLog2SubboxesUpToDim(); }

template<typename R,size_t d,size_t q>
const rfio::PotentialField<R,d,q>& 
PotentialField<R,d,q>::GetRFIOPotentialField() const
{ return rfioPotential_; }

template<typename R,size_t d,size_t q>
inline void 
PrintErrorEstimates
( MPI_Comm comm,
  const PotentialField<R,d,q>& u,
  const vector<Source<R,d>>& sources )
{
    rfio::PrintErrorEstimates( comm, u.GetRFIOPotentialField(), sources );
}

template<typename R,size_t d,size_t q>
inline void 
WriteImage
( MPI_Comm comm, 
  const size_t N,
  const Box<R,d>& tBox,
  const PotentialField<R,d,q>& u,
  const string& basename )
{
    rfio::WriteImage( comm, N, tBox, u.GetRFIOPotentialField(), basename );
}

template<typename R,size_t d,size_t q>
inline void 
WriteImage
( MPI_Comm comm, 
  const size_t N,
  const Box<R,d>& tBox,
  const PotentialField<R,d,q>& u,
  const string& basename,
  const vector<Source<R,d>>& sources )
{
    rfio::WriteImage
    ( comm, N, tBox, u.GetRFIOPotentialField(), basename, sources );
}

} // lnuft
} // bfio

#endif // ifndef BFIO_LNUFT_POTENTIAL_FIELD_HPP
