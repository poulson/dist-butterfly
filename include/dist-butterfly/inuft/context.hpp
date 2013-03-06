/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of DistButterfly and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef DBF_INUFT_CONTEXT_HPP
#define DBF_INUFT_CONTEXT_HPP

#include <array>
#include <memory>
#include <vector>

#include "dist-butterfly/constants.hpp"
#include "dist-butterfly/tools/lapack.hpp"

namespace dbf {

using std::array;
using std::complex;
using std::size_t;
using std::vector;

namespace inuft {
template<typename R,size_t d,size_t q>
class Context
{
    const Direction direction_;
    const size_t N_;
    const Box<R,d> sBox_, tBox_;

    vector<R> chebyshevNodes_;
    vector<array<R,d>> chebyshevGrid_;
    array<vector<R>,d> realInverseMaps_, imagInverseMaps_,
                       realForwardMaps_, imagForwardMaps_;

    void GenerateChebyshevNodes();
    void GenerateChebyshevGrid();
    void GenerateOffsetMaps();
    
public:        
    Context
    ( const Direction direction,
      const size_t N,
      const Box<R,d>& sBox,
      const Box<R,d>& tBox );

    Direction
    GetDirection() const;

    const vector<R>&          GetChebyshevNodes() const;
    const vector<array<R,d>>& GetChebyshevGrid() const;

    const vector<R>& GetRealInverseMap( const size_t j ) const;
    const vector<R>& GetImagInverseMap( const size_t j ) const;
    const vector<R>& GetRealForwardMap( const size_t j ) const;
    const vector<R>& GetImagForwardMap( const size_t j ) const;
};

// Implementations

template<typename R,size_t d,size_t q>
inline void 
Context<R,d,q>::GenerateChebyshevNodes()
{
    const R pi = Pi<R>();
    for( size_t t=0; t<q; ++t )
        chebyshevNodes_[t] = cos(R(t*pi/(q-1)))/2;
}

template<typename R,size_t d,size_t q>
inline void
Context<R,d,q>::GenerateChebyshevGrid()
{
    const R pi = Pi<R>();
    const size_t q_to_d = chebyshevGrid_.size();

    for( size_t t=0; t<q_to_d; ++t )
    {
        size_t q_to_j = 1;
        for( size_t j=0; j<d; ++j )
        {
            size_t i = (t/q_to_j) % q;
            chebyshevGrid_[t][j] = cos(R(i*pi/(q-1)))/2;
            q_to_j *= q;
        }
    }
}

template<typename R,size_t d,size_t q>
inline void
Context<R,d,q>::GenerateOffsetMaps()
{
    const R twoPi = TwoPi<R>();
    for( size_t j=0; j<d; ++j )
    {
        realInverseMaps_[j].resize( q*q );
        imagInverseMaps_[j].resize( q*q );
        realForwardMaps_[j].resize( q*q );
        imagForwardMaps_[j].resize( q*q );
    }

    array<R,d> productWidths;
    for( size_t j=0; j<d; ++j )
        productWidths[j] = sBox_.widths[j]*tBox_.widths[j]/N_;

    // Form the initialization offset map
    vector<int> pivot(q);
    vector<complex<R>> A( q*q ), work( q*q );
    for( size_t j=0; j<d; ++j )
    {
        // Form
        if( direction_ == FORWARD )
        {
            for( size_t t=0; t<q; ++t )
            {
                for( size_t tPrime=0; tPrime<q; ++tPrime )
                {
                    A[t*q+tPrime] = 
                        ImagExp<R>
                        ( -twoPi*chebyshevNodes_[t]*chebyshevNodes_[tPrime]*
                          productWidths[j] );
                }
            }
        }
        else
        {
            for( size_t t=0; t<q; ++t )
            {
                for( size_t tPrime=0; tPrime<q; ++tPrime )
                {
                    A[t*q+tPrime] = 
                        ImagExp<R>
                        ( twoPi*chebyshevNodes_[t]*chebyshevNodes_[tPrime]*
                          productWidths[j] );
                }
            }
        }
        // Factor and invert
        LU( q, q, &A[0], q, &pivot[0] );
        InvertLU( q, &A[0], q, &pivot[0], &work[0], q*q );
        // Separate the real and imaginary parts of the inverse
        for( size_t t=0; t<q; ++t )
        {
            for( size_t tPrime=0; tPrime<q; ++tPrime )
            {
                realInverseMaps_[j][t*q+tPrime] = A[t*q+tPrime].real();
                imagInverseMaps_[j][t*q+tPrime] = A[t*q+tPrime].imag();
            }
        }
    }

    // Form the weight recursion offset map
    if( direction_ == FORWARD )
    {
        for( size_t j=0; j<d; ++j )
        {
            for( size_t t=0; t<q; ++t )
            {
                for( size_t tPrime=0; tPrime<q; ++tPrime ) 
                {
                    complex<R> alpha = 
                        ImagExp<R>
                        ( -twoPi*chebyshevNodes_[t]*chebyshevNodes_[tPrime]*
                          productWidths[j]/2 );
                    realForwardMaps_[j][t*q+tPrime] = alpha.real();
                    imagForwardMaps_[j][t*q+tPrime] = alpha.imag();
                }
            }
        }
    }
    else
    {
        for( size_t j=0; j<d; ++j )
        {
            for( size_t t=0; t<q; ++t )
            {
                for( size_t tPrime=0; tPrime<q; ++tPrime ) 
                {
                    complex<R> alpha = 
                        ImagExp<R>
                        ( twoPi*chebyshevNodes_[t]*chebyshevNodes_[tPrime]*
                          productWidths[j]/2 );
                    realForwardMaps_[j][t*q+tPrime] = alpha.real();
                    imagForwardMaps_[j][t*q+tPrime] = alpha.imag();
                }
            }
        }
    }
}

template<typename R,size_t d,size_t q>
inline
Context<R,d,q>::Context
( const Direction direction,
  const size_t N,
  const Box<R,d>& sBox,
  const Box<R,d>& tBox ) 
: direction_(direction), N_(N), sBox_(sBox), tBox_(tBox), 
  chebyshevNodes_( q ), chebyshevGrid_( Pow<q,d>::val )
{
    GenerateChebyshevNodes();
    GenerateChebyshevGrid();
    GenerateOffsetMaps();
}

template<typename R,size_t d,size_t q>
inline Direction
Context<R,d,q>::GetDirection() const
{ return direction_; }

template<typename R,size_t d,size_t q>
inline const vector<R>&
Context<R,d,q>::GetChebyshevNodes() const
{ return chebyshevNodes_; }

template<typename R,size_t d,size_t q>
inline const vector<array<R,d>>&
Context<R,d,q>::GetChebyshevGrid() const
{ return chebyshevGrid_; }

template<typename R,size_t d,size_t q>
inline const vector<R>&
Context<R,d,q>::GetRealInverseMap( const size_t j ) const
{ return realInverseMaps_[j]; }

template<typename R,size_t d,size_t q>
inline const vector<R>&
Context<R,d,q>::GetImagInverseMap( const size_t j ) const
{ return imagInverseMaps_[j]; }

template<typename R,size_t d,size_t q>
inline const vector<R>&
Context<R,d,q>::GetRealForwardMap( const size_t j ) const
{ return realForwardMaps_[j]; }

template<typename R,size_t d,size_t q>
inline const vector<R>&
Context<R,d,q>::GetImagForwardMap( const size_t j ) const
{ return imagForwardMaps_[j]; }

} // inuft
} // dbf

#endif // ifndef DBF_INUFT_CONTEXT_HPP
