/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of ButterflyFIO and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef BFIO_INUFT_CONTEXT_HPP
#define BFIO_INUFT_CONTEXT_HPP

#include <array>
#include <complex>
#include <memory>
#include <vector>

#include "bfio/constants.hpp"
#include "bfio/tools/lapack.hpp"

namespace bfio {

using std::array;
using std::complex;
using std::size_t;
using std::vector;

namespace inuft {
template<typename R,size_t d,size_t q>
class Context
{
    const Direction _direction;
    const size_t _N;
    const Box<R,d> _sourceBox;
    const Box<R,d> _targetBox;

    vector<R> _chebyshevNodes;
    vector<array<R,d>> _chebyshevGrid;
    array<vector<R>,d> _realInverseMaps, _imagInverseMaps,
                       _realForwardMaps, _imagForwardMaps;

    void GenerateChebyshevNodes();
    void GenerateChebyshevGrid();
    void GenerateOffsetMaps();
    
public:        
    Context
    ( const Direction direction,
      const size_t N,
      const Box<R,d>& sourceBox,
      const Box<R,d>& targetBox );

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
void 
Context<R,d,q>::GenerateChebyshevNodes()
{
    for( size_t t=0; t<q; ++t )
        _chebyshevNodes[t] = 0.5*cos(static_cast<R>(t*Pi/(q-1)));
}

template<typename R,size_t d,size_t q>
void
Context<R,d,q>::GenerateChebyshevGrid()
{
    const size_t q_to_d = _chebyshevGrid.size();   

    for( size_t t=0; t<q_to_d; ++t )
    {
        size_t q_to_j = 1;
        for( size_t j=0; j<d; ++j )
        {
            size_t i = (t/q_to_j) % q;
            _chebyshevGrid[t][j] = 0.5*cos(R(i*Pi/(q-1)));
            q_to_j *= q;
        }
    }
}

template<typename R,size_t d,size_t q>
void
Context<R,d,q>::GenerateOffsetMaps()
{
    for( size_t j=0; j<d; ++j )
    {
        _realInverseMaps[j].resize( q*q );
        _imagInverseMaps[j].resize( q*q );
        _realForwardMaps[j].resize( q*q );
        _imagForwardMaps[j].resize( q*q );
    }

    array<R,d> productWidths;
    for( size_t j=0; j<d; ++j )
        productWidths[j] = _sourceBox.widths[j]*_targetBox.widths[j]/_N;

    // Form the initialization offset map
    vector<int> pivot(q);
    vector<complex<R>> A( q*q );
    vector<complex<R>> work( q*q );
    for( size_t j=0; j<d; ++j )
    {
        // Form
        if( _direction == FORWARD )
        {
            for( size_t t=0; t<q; ++t )
            {
                for( size_t tPrime=0; tPrime<q; ++tPrime )
                {
                    A[t*q+tPrime] = 
                        ImagExp<R>
                        ( -TwoPi*_chebyshevNodes[t]*_chebyshevNodes[tPrime]*
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
                        ( TwoPi*_chebyshevNodes[t]*_chebyshevNodes[tPrime]*
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
                _realInverseMaps[j][t*q+tPrime] = std::real( A[t*q+tPrime] );
                _imagInverseMaps[j][t*q+tPrime] = std::imag( A[t*q+tPrime] );
            }
        }
    }

    // Form the weight recursion offset map
    if( _direction == FORWARD )
    {
        for( size_t j=0; j<d; ++j )
        {
            for( size_t t=0; t<q; ++t )
            {
                for( size_t tPrime=0; tPrime<q; ++tPrime ) 
                {
                    complex<R> alpha = 
                        ImagExp<R>
                        ( -TwoPi*_chebyshevNodes[t]*_chebyshevNodes[tPrime]*
                          productWidths[j]/2 );
                    _realForwardMaps[j][t*q+tPrime] = std::real( alpha );
                    _imagForwardMaps[j][t*q+tPrime] = std::imag( alpha );
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
                        ( TwoPi*_chebyshevNodes[t]*_chebyshevNodes[tPrime]*
                          productWidths[j]/2 );
                    _realForwardMaps[j][t*q+tPrime] = std::real( alpha );
                    _imagForwardMaps[j][t*q+tPrime] = std::imag( alpha );
                }
            }
        }
    }
}

template<typename R,size_t d,size_t q>
Context<R,d,q>::Context
( const Direction direction,
  const size_t N,
  const Box<R,d>& sourceBox,
  const Box<R,d>& targetBox ) 
: _direction(direction), _N(N), _sourceBox(sourceBox), _targetBox(targetBox), 
  _chebyshevNodes( q ), _chebyshevGrid( Pow<q,d>::val )
{
    GenerateChebyshevNodes();
    GenerateChebyshevGrid();
    GenerateOffsetMaps();
}

template<typename R,size_t d,size_t q>
inline Direction
Context<R,d,q>::GetDirection() const
{ return _direction; }

template<typename R,size_t d,size_t q>
inline const vector<R>&
Context<R,d,q>::GetChebyshevNodes() const
{ return _chebyshevNodes; }

template<typename R,size_t d,size_t q>
inline const vector<array<R,d>>&
Context<R,d,q>::GetChebyshevGrid() const
{ return _chebyshevGrid; }

template<typename R,size_t d,size_t q>
inline const vector<R>&
Context<R,d,q>::GetRealInverseMap( const size_t j ) const
{ return _realInverseMaps[j]; }

template<typename R,size_t d,size_t q>
inline const vector<R>&
Context<R,d,q>::GetImagInverseMap( const size_t j ) const
{ return _imagInverseMaps[j]; }

template<typename R,size_t d,size_t q>
inline const vector<R>&
Context<R,d,q>::GetRealForwardMap( const size_t j ) const
{ return _realForwardMaps[j]; }

template<typename R,size_t d,size_t q>
inline const vector<R>&
Context<R,d,q>::GetImagForwardMap( const size_t j ) const
{ return _imagForwardMaps[j]; }

} // inuft
} // bfio

#endif // ifndef BFIO_INUFT_CONTEXT_HPP
