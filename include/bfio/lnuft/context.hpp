/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of ButterflyFIO and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef BFIO_LNUFT_CONTEXT_HPP
#define BFIO_LNUFT_CONTEXT_HPP

#include <array>
#include <vector>
#include "bfio/rfio/context.hpp"

namespace bfio {

using std::array;
using std::size_t;
using std::vector;

namespace lnuft {

template<typename R,size_t d,size_t q>
class Context
{
    const rfio::Context<R,d,q> _rfioContext;
    const Direction _direction;
    const size_t _N;
    const Box<R,d> _sourceBox;
    const Box<R,d> _targetBox;

    array<vector<R>,d> _realOffsetEvaluations;
    array<vector<R>,d> _imagOffsetEvaluations;

    void GenerateOffsetEvaluations();

public:        
    Context
    ( const Direction direction,
      const size_t N,
      const Box<R,d>& sourceBox,
      const Box<R,d>& targetBox );

    const rfio::Context<R,d,q>& GetRFIOContext() const;
    Direction GetDirection() const;
    const array<vector<R>,d>& GetRealOffsetEvaluations() const;
    const array<vector<R>,d>& GetImagOffsetEvaluations() const;
};

// Implementations

template<typename R,size_t d,size_t q>
void
Context<R,d,q>::GenerateOffsetEvaluations()
{
    const size_t log2N = Log2( _N );
    const size_t middleLevel = log2N/2;

    array<R,d> wAMiddle, wBMiddle;
    for( std::size_t j=0; j<d; ++j )
    {
        wAMiddle[j] = _targetBox.widths[j] / (1<<middleLevel);
        wBMiddle[j] = _sourceBox.widths[j] / (1<<(log2N-middleLevel));
    }

    // Form the offset grid evaluations
    const R SignedTwoPi = ( _direction==FORWARD ? -TwoPi : TwoPi ); 
    vector<R> phaseEvaluations(q*q);
    const vector<R>& chebyshevNodes = _rfioContext.GetChebyshevNodes();
    const R* chebyshevBuffer = &chebyshevNodes[0];
    for( size_t j=0; j<d; ++j )
    {
        for( size_t t=0; t<q; ++t )
            for( size_t tPrime=0; tPrime<q; ++tPrime )
                phaseEvaluations[t*q+tPrime] =
                    SignedTwoPi*wAMiddle[j]*wBMiddle[j]*
                    chebyshevBuffer[t]*chebyshevBuffer[tPrime];
        SinCosBatch
        ( phaseEvaluations, 
          _imagOffsetEvaluations[j], _realOffsetEvaluations[j] );
    }
}

template<typename R,size_t d,size_t q>
inline
Context<R,d,q>::Context
( Direction direction, size_t N, 
  const Box<R,d>& sourceBox, const Box<R,d>& targetBox ) 
: _rfioContext(), _direction(direction), _N(N), 
  _sourceBox(sourceBox), _targetBox(targetBox)
{ GenerateOffsetEvaluations(); }

template<typename R,size_t d,size_t q>
inline const rfio::Context<R,d,q>&
Context<R,d,q>::GetRFIOContext() const
{ return _rfioContext; }

template<typename R,size_t d,size_t q>
inline Direction
Context<R,d,q>::GetDirection() const
{ return _direction; }

template<typename R,size_t d,size_t q>
inline const array<vector<R>,d>&
Context<R,d,q>::GetRealOffsetEvaluations() const
{ return _realOffsetEvaluations; }

template<typename R,size_t d,size_t q>
inline const array<vector<R>,d>&
Context<R,d,q>::GetImagOffsetEvaluations() const
{ return _imagOffsetEvaluations; }

} // lnuft
} // bfio

#endif // ifndef BFIO_LNUFT_CONTEXT_HPP
