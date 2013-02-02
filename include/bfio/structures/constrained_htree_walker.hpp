/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of ButterflyFIO and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef BFIO_STRUCTURES_CONSTRAINED_HTREE_WALKER_HPP
#define BFIO_STRUCTURES_CONSTRAINED_HTREE_WALKER_HPP

#include <array>
#include <cstddef>
#include <stdexcept>

#include "bfio/tools/twiddle.hpp"

namespace bfio {

using std::array;
using std::size_t;

// Constrained HTree Walker
template<size_t d>
class ConstrainedHTreeWalker
{
    bool _overflowed;
    size_t _firstOpenDim;
    size_t _nextZeroDim;
    size_t _nextZeroLevel;
    array<size_t,d> _state;
    array<size_t,d> _log2BoxesPerDim;
public:
    ConstrainedHTreeWalker( const array<size_t,d>& log2BoxesPerDim );
    ~ConstrainedHTreeWalker();

    array<size_t,d> State() const;

    void Walk();
};

// Implementations

template<size_t d>
ConstrainedHTreeWalker<d>::ConstrainedHTreeWalker
( const array<size_t,d>& log2BoxesPerDim ) 
: _overflowed(false), _nextZeroLevel(0),
  _log2BoxesPerDim(log2BoxesPerDim) 
{
    _state.fill(0);
    for( _firstOpenDim=0; _firstOpenDim<d; ++_firstOpenDim )
        if( log2BoxesPerDim[_firstOpenDim] != 0 )
            break;
    _nextZeroDim = _firstOpenDim;
}

template<size_t d>
inline
ConstrainedHTreeWalker<d>::~ConstrainedHTreeWalker() 
{ }

template<size_t d>
inline array<size_t,d> 
ConstrainedHTreeWalker<d>::State() const
{ 
#ifndef RELEASE
    if( _overflowed )
        throw std::logic_error( "Overflowed HTree" );
#endif
    return _state; 
}

template<size_t d>
void 
ConstrainedHTreeWalker<d>::Walk()
{
#ifndef RELEASE
    if( _nextZeroDim == d )
    {
        _overflowed = true;
        return;
    }
#endif

    const size_t zeroDim = _nextZeroDim;
    const size_t zeroLevel = _nextZeroLevel;

    if( zeroDim == _firstOpenDim )
    {
        // Zero the first (zeroLevel-1) bits of all coordinates
        // and then increment at level zeroLevel
        for( size_t j=0; j<d; ++j )
            _state[j] &= ~((1u<<zeroLevel)-1);
        _state[zeroDim] |= 1u<<zeroLevel;

        // Set up for the next walk
        // We need to find the dimension with the first unconstrained
        // zero bit.
        size_t minDim = d;
        size_t minTrailingOnes = sizeof(size_t)*8+1; 
        array<size_t,d> numberOfTrailingOnes;
        for( size_t j=0; j<d; ++j )
        {
            numberOfTrailingOnes[j] = NumberOfTrailingOnes( _state[j] );
            if( (numberOfTrailingOnes[j] < minTrailingOnes) &&
                (numberOfTrailingOnes[j] != _log2BoxesPerDim[j]) )
            {
                minDim = j;
                minTrailingOnes = numberOfTrailingOnes[j];
            }
        }
        _nextZeroDim = minDim;
        _nextZeroLevel = minTrailingOnes;
    }
    else
    {
        for( size_t j=0; j<=zeroDim; ++j )
            _state[j] &= ~((1u<<(zeroLevel+1))-1);
        for( size_t j=zeroDim+1; j<d; ++j )
            _state[j] &= ~((1u<<zeroLevel)-1);
        _state[zeroDim] |= 1u<<zeroLevel;

        // Set up for the next walk
        _nextZeroDim = _firstOpenDim;
        _nextZeroLevel = 0;
    }
}

} // bfio

#endif // ifndef BFIO_STRUCTURES_CONSTRAINED_HTREE_WALKER_HPP
