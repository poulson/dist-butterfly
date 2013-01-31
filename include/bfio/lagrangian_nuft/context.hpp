/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of ButterflyFIO and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef BFIO_LAGRANGIAN_NUFT_CONTEXT_HPP
#define BFIO_LAGRANGIAN_NUFT_CONTEXT_HPP

#include "bfio/rfio/context.hpp"

namespace bfio {

namespace lagrangian_nuft {
template<typename R,std::size_t d,std::size_t q>
class Context
{
    const rfio::Context<R,d,q> _rfioContext;
    const Direction _direction;
    const std::size_t _N;
    const Box<R,d> _sourceBox;
    const Box<R,d> _targetBox;

    Array< std::vector<R>, d > _realOffsetEvaluations;
    Array< std::vector<R>, d > _imagOffsetEvaluations;

    void GenerateOffsetEvaluations();

public:        
    Context
    ( const Direction direction,
      const std::size_t N,
      const Box<R,d>& sourceBox,
      const Box<R,d>& targetBox );

    const rfio::Context<R,d,q>&
    GetReducedFIOContext() const;

    Direction
    GetDirection() const;

    const Array< std::vector<R>, d >&
    GetRealOffsetEvaluations() const;

    const Array< std::vector<R>, d >&
    GetImagOffsetEvaluations() const;
};
} // lagrangian_nuft

// Implementations

template<typename R,std::size_t d,std::size_t q>
void
lagrangian_nuft::Context<R,d,q>::GenerateOffsetEvaluations()
{
    const std::size_t log2N = Log2( _N );
    const std::size_t middleLevel = log2N/2;

    Array<R,d> wAMiddle, wBMiddle;
    for( std::size_t j=0; j<d; ++j )
    {
        wAMiddle[j] = _targetBox.widths[j] / (1<<middleLevel);
        wBMiddle[j] = _sourceBox.widths[j] / (1<<(log2N-middleLevel));
    }

    // Form the offset grid evaluations
    const R SignedTwoPi = ( _direction==FORWARD ? -TwoPi : TwoPi ); 
    std::vector<R> phaseEvaluations(q*q);
    const std::vector<R>& chebyshevNodes = _rfioContext.GetChebyshevNodes();
    const R* chebyshevBuffer = &chebyshevNodes[0];
    for( std::size_t j=0; j<d; ++j )
    {
        for( std::size_t t=0; t<q; ++t )
            for( std::size_t tPrime=0; tPrime<q; ++tPrime )
                phaseEvaluations[t*q+tPrime] =
                    SignedTwoPi*wAMiddle[j]*wBMiddle[j]*
                    chebyshevBuffer[t]*chebyshevBuffer[tPrime];
        SinCosBatch
        ( phaseEvaluations, 
          _imagOffsetEvaluations[j], _realOffsetEvaluations[j] );
    }
}

template<typename R,std::size_t d,std::size_t q>
inline
lagrangian_nuft::Context<R,d,q>::Context
( Direction direction, std::size_t N, 
  const Box<R,d>& sourceBox, const Box<R,d>& targetBox ) 
: _rfioContext(), _direction(direction), _N(N), 
  _sourceBox(sourceBox), _targetBox(targetBox)
{ GenerateOffsetEvaluations(); }

template<typename R,std::size_t d,std::size_t q>
inline const rfio::Context<R,d,q>&
lagrangian_nuft::Context<R,d,q>::GetReducedFIOContext() const
{ return _rfioContext; }

template<typename R,std::size_t d,std::size_t q>
inline Direction
lagrangian_nuft::Context<R,d,q>::GetDirection() const
{ return _direction; }

template<typename R,std::size_t d,std::size_t q>
inline const Array< std::vector<R>, d >&
lagrangian_nuft::Context<R,d,q>::GetRealOffsetEvaluations() const
{ return _realOffsetEvaluations; }

template<typename R,std::size_t d,std::size_t q>
inline const Array< std::vector<R>, d >&
lagrangian_nuft::Context<R,d,q>::GetImagOffsetEvaluations() const
{ return _imagOffsetEvaluations; }

} // bfio

#endif // ifndef BFIO_LAGRANGIAN_NUFT_CONTEXT_HPP
