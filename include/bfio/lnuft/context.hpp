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
    const rfio::Context<R,d,q> rfioContext_;
    const Direction direction_;
    const size_t N_;
    const Box<R,d> sBox_, tBox_;

    array<vector<R>,d> realOffsetEvaluations_,
                       imagOffsetEvaluations_;

    void GenerateOffsetEvaluations();

public:        
    Context
    ( const Direction direction,
      const size_t N,
      const Box<R,d>& sBox,
      const Box<R,d>& tBox );

    const rfio::Context<R,d,q>& GetRFIOContext() const;
    Direction GetDirection() const;
    const array<vector<R>,d>& GetRealOffsetEvaluations() const;
    const array<vector<R>,d>& GetImagOffsetEvaluations() const;
};

// Implementations

template<typename R,size_t d,size_t q>
inline void
Context<R,d,q>::GenerateOffsetEvaluations()
{
    const size_t log2N = Log2( N_ );
    const size_t middleLevel = log2N/2;

    array<R,d> wAMiddle, wBMiddle;
    for( std::size_t j=0; j<d; ++j )
    {
        wAMiddle[j] = tBox_.widths[j] / (1<<middleLevel);
        wBMiddle[j] = sBox_.widths[j] / (1<<(log2N-middleLevel));
    }

    // Form the offset grid evaluations
    const R SignedTwoPi = ( direction_==FORWARD ? -TwoPi : TwoPi ); 
    vector<R> phaseEvaluations(q*q);
    const vector<R>& chebyshevNodes = rfioContext_.GetChebyshevNodes();
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
          imagOffsetEvaluations_[j], realOffsetEvaluations_[j] );
    }
}

template<typename R,size_t d,size_t q>
inline
Context<R,d,q>::Context
( Direction direction, size_t N, const Box<R,d>& sBox, const Box<R,d>& tBox ) 
: rfioContext_(), direction_(direction), N_(N), sBox_(sBox), tBox_(tBox)
{ GenerateOffsetEvaluations(); }

template<typename R,size_t d,size_t q>
inline const rfio::Context<R,d,q>&
Context<R,d,q>::GetRFIOContext() const
{ return rfioContext_; }

template<typename R,size_t d,size_t q>
inline Direction
Context<R,d,q>::GetDirection() const
{ return direction_; }

template<typename R,size_t d,size_t q>
inline const array<vector<R>,d>&
Context<R,d,q>::GetRealOffsetEvaluations() const
{ return realOffsetEvaluations_; }

template<typename R,size_t d,size_t q>
inline const array<vector<R>,d>&
Context<R,d,q>::GetImagOffsetEvaluations() const
{ return imagOffsetEvaluations_; }

} // lnuft
} // bfio

#endif // ifndef BFIO_LNUFT_CONTEXT_HPP
