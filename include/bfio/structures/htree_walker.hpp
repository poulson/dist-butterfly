/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of ButterflyFIO and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef BFIO_STRUCTURES_HTREE_WALKER_HPP
#define BFIO_STRUCTURES_HTREE_WALKER_HPP

#include <cstddef>
#include <stdexcept>
#include "bfio/structures/array.hpp"
#include "bfio/tools/twiddle.hpp"

namespace bfio {

template<std::size_t d>
class HTreeWalker
{
    std::size_t _nextZeroDim;
    std::size_t _nextZeroLevel;
    Array<std::size_t,d> _state;
public:
    HTreeWalker();
    ~HTreeWalker();

    Array<std::size_t,d> State() const;

    void Walk();
};

// Implementations

template<std::size_t d>
inline
HTreeWalker<d>::HTreeWalker() 
: _nextZeroDim(0), _nextZeroLevel(0), _state(0)
{ }

template<std::size_t d>
inline 
HTreeWalker<d>::~HTreeWalker() 
{ }

template<std::size_t d>
inline Array<std::size_t,d> 
HTreeWalker<d>::State() const
{ return _state; }

template<std::size_t d>
void 
HTreeWalker<d>::Walk()
{
    const std::size_t zeroDim = _nextZeroDim;
    const std::size_t zeroLevel = _nextZeroLevel;

    if( zeroDim == 0 )
    {
        // Zero the first (zeroLevel-1) bits of all coordinates
        // and then increment at level zeroLevel
        for( std::size_t j=0; j<d; ++j )
            _state[j] &= ~((1u<<zeroLevel)-1);
        _state[zeroDim] |= 1u<<zeroLevel;

        // Set up for the next walk
        // We need to find the dimension with the first zero bit.
        std::size_t minDim = d;
        std::size_t minTrailingOnes = sizeof(std::size_t)*8+1;
        Array<std::size_t,d> numberOfTrailingOnes;
        for( std::size_t j=0; j<d; ++j )
        {
            numberOfTrailingOnes[j] = NumberOfTrailingOnes( _state[j] );
            if( numberOfTrailingOnes[j] < minTrailingOnes )
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
        for( std::size_t j=0; j<=zeroDim; ++j )
            _state[j] &= ~((1u<<(zeroLevel+1))-1);
        for( std::size_t j=zeroDim+1; j<d; ++j )
            _state[j] &= ~((1u<<zeroLevel)-1);
        _state[zeroDim] |= 1u<<zeroLevel;

        // Set up for the next walk
        _nextZeroDim = 0;
        _nextZeroLevel = 0;
    }
}

} // bfio

#endif // ifndef BFIO_STRUCTURES_HTREE_WALKER_HPP
