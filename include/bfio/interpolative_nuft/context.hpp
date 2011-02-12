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
#ifndef BFIO_INTERPOLATIVE_NUFT_CONTEXT_HPP
#define BFIO_INTERPOLATIVE_NUFT_CONTEXT_HPP 1

#include <memory>
#include <vector>
#include "bfio/constants.hpp"
#include "bfio/structures/array.hpp"

#include "bfio/tools/lapack.hpp"

namespace bfio {

namespace interpolative_nuft {
template<typename R,std::size_t d,std::size_t q>
class Context
{
    const std::size_t _N;
    const Box<R,d> _sourceBox;
    const Box<R,d> _targetBox;

    std::vector<R> _chebyshevNodes;
    std::vector< Array<R,d> > _chebyshevGrid;
    Array< std::vector<R>, d > _realInverseMaps;
    Array< std::vector<R>, d > _imagInverseMaps;
    Array< std::vector<R>, d > _realForwardMaps;
    Array< std::vector<R>, d > _imagForwardMaps;

    void GenerateChebyshevNodes();
    void GenerateChebyshevGrid();
    void GenerateOffsetMaps();
    
public:        
    Context
    ( const std::size_t N,
      const Box<R,d>& sourceBox,
      const Box<R,d>& targetBox );

    const std::vector<R>&
    GetChebyshevNodes() const;

    const std::vector< Array<R,d> >&
    GetChebyshevGrid() const;

    const std::vector<R>&
    GetRealInverseMap( const std::size_t j ) const;

    const std::vector<R>&
    GetImagInverseMap( const std::size_t j ) const;

    const std::vector<R>&
    GetRealForwardMap( const std::size_t j ) const;

    const std::vector<R>&
    GetImagForwardMap( const std::size_t j ) const;
};
} // interpolative_nuft

// Implementations

template<typename R,std::size_t d,std::size_t q>
void 
interpolative_nuft::Context<R,d,q>::GenerateChebyshevNodes()
{
    for( std::size_t t=0; t<q; ++t )
        _chebyshevNodes[t] = 0.5*cos(static_cast<R>(t*Pi/(q-1)));
}

template<typename R,std::size_t d,std::size_t q>
void
interpolative_nuft::Context<R,d,q>::GenerateChebyshevGrid()
{
    const std::size_t q_to_d = _chebyshevGrid.size();   

    for( std::size_t t=0; t<q_to_d; ++t )
    {
        std::size_t q_to_j = 1;
        for( std::size_t j=0; j<d; ++j )
        {
            std::size_t i = (t/q_to_j) % q;
            _chebyshevGrid[t][j] = 0.5*cos(static_cast<R>(i*Pi/(q-1)));
            q_to_j *= q;
        }
    }
}

template<typename R,std::size_t d,std::size_t q>
void
interpolative_nuft::Context<R,d,q>::GenerateOffsetMaps()
{
    for( std::size_t j=0; j<d; ++j )
    {
        _realInverseMaps[j].resize( q*q );
        _imagInverseMaps[j].resize( q*q );
        _realForwardMaps[j].resize( q*q );
        _imagForwardMaps[j].resize( q*q );
    }

    Array<R,d> productWidths;
    for( std::size_t j=0; j<d; ++j )
        productWidths[j] = _sourceBox.widths[j]*_targetBox.widths[j]/_N;

    // Form the initialization offset map
    std::vector<int> pivot(q);
    std::vector< std::complex<R> > A( q*q );
    std::vector< std::complex<R> > work( q*q );
    for( std::size_t j=0; j<d; ++j )
    {
        // Form
        for( std::size_t t=0; t<q; ++t )
        {
            for( std::size_t tPrime=0; tPrime<q; ++tPrime )
            {
                A[t*q+tPrime] = 
                    ImagExp<R>
                    ( TwoPi*_chebyshevNodes[t]*_chebyshevNodes[tPrime]*
                      productWidths[j] );
            }
        }
        // Factor and invert
        LU( q, q, &A[0], q, &pivot[0] );
        InvertLU( q, &A[0], q, &pivot[0], &work[0], q*q );
        // Separate the real and imaginary parts of the inverse
        for( std::size_t t=0; t<q; ++t )
        {
            for( std::size_t tPrime=0; tPrime<q; ++tPrime )
            {
                _realInverseMaps[j][t*q+tPrime] = std::real( A[t*q+tPrime] );
                _imagInverseMaps[j][t*q+tPrime] = std::imag( A[t*q+tPrime] );
            }
        }
    }

    // Form the weight recursion offset map
    for( std::size_t j=0; j<d; ++j )
    {
        for( std::size_t t=0; t<q; ++t )
        {
            for( std::size_t tPrime=0; tPrime<q; ++tPrime ) 
            {
                std::complex<R> alpha = 
                    ImagExp<R>
                    ( TwoPi*_chebyshevNodes[t]*_chebyshevNodes[tPrime]*
                      productWidths[j]/2 );
                _realForwardMaps[j][t*q+tPrime] = std::real( alpha );
                _imagForwardMaps[j][t*q+tPrime] = std::imag( alpha );
            }
        }
    }
}

template<typename R,std::size_t d,std::size_t q>
interpolative_nuft::Context<R,d,q>::Context
( const std::size_t N,
  const Box<R,d>& sourceBox,
  const Box<R,d>& targetBox ) 
: _N(N), _sourceBox(sourceBox), _targetBox(targetBox), 
  _chebyshevNodes( q ), _chebyshevGrid( Pow<q,d>::val )
{
    GenerateChebyshevNodes();
    GenerateChebyshevGrid();
    GenerateOffsetMaps();
}

template<typename R,std::size_t d,std::size_t q>
inline const std::vector<R>&
interpolative_nuft::Context<R,d,q>::GetChebyshevNodes() const
{ return _chebyshevNodes; }

template<typename R,std::size_t d,std::size_t q>
inline const std::vector< Array<R,d> >&
interpolative_nuft::Context<R,d,q>::GetChebyshevGrid() const
{ return _chebyshevGrid; }

template<typename R,std::size_t d,std::size_t q>
inline const std::vector<R>&
interpolative_nuft::Context<R,d,q>::GetRealInverseMap
( const std::size_t j ) const
{ return _realInverseMaps[j]; }

template<typename R,std::size_t d,std::size_t q>
inline const std::vector<R>&
interpolative_nuft::Context<R,d,q>::GetImagInverseMap
( const std::size_t j ) const
{ return _imagInverseMaps[j]; }

template<typename R,std::size_t d,std::size_t q>
inline const std::vector<R>&
interpolative_nuft::Context<R,d,q>::GetRealForwardMap
( const std::size_t j ) const
{ return _realForwardMaps[j]; }

template<typename R,std::size_t d,std::size_t q>
inline const std::vector<R>&
interpolative_nuft::Context<R,d,q>::GetImagForwardMap
( const std::size_t j ) const
{ return _imagForwardMaps[j]; }

} // bfio

#endif // BFIO_INTERPOLATIVE_NUFT_CONTEXT_HPP

