/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of DistButterfly and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef DBF_STRUCTURES_HTREE_WALKER_HPP
#define DBF_STRUCTURES_HTREE_WALKER_HPP

#include <array>
#include <cstddef>
#include <stdexcept>

#include "dist-butterfly/tools/twiddle.hpp"

namespace dbf {

using std::array;
using std::size_t;

template<size_t d>
class HTreeWalker
{
    size_t nextZeroDim_;
    size_t nextZeroLevel_;
    array<size_t,d> state_;
public:
    HTreeWalker();
    ~HTreeWalker();

    array<size_t,d> State() const;

    void Walk();
};

// Implementations

template<size_t d>
inline
HTreeWalker<d>::HTreeWalker() 
: nextZeroDim_(0), nextZeroLevel_(0)
{ 
    state_.fill(0);
}

template<size_t d>
inline 
HTreeWalker<d>::~HTreeWalker() 
{ }

template<size_t d>
inline array<size_t,d> 
HTreeWalker<d>::State() const
{ return state_; }

template<size_t d>
void 
HTreeWalker<d>::Walk()
{
    const size_t zeroDim = nextZeroDim_;
    const size_t zeroLevel = nextZeroLevel_;

    if( zeroDim == 0 )
    {
        // Zero the first (zeroLevel-1) bits of all coordinates
        // and then increment at level zeroLevel
        for( size_t j=0; j<d; ++j )
            state_[j] &= ~((1u<<zeroLevel)-1);
        state_[zeroDim] |= 1u<<zeroLevel;

        // Set up for the next walk
        // We need to find the dimension with the first zero bit.
        size_t minDim = d;
        size_t minTrailingOnes = sizeof(size_t)*8+1;
        array<size_t,d> numberOfTrailingOnes;
        for( size_t j=0; j<d; ++j )
        {
            numberOfTrailingOnes[j] = NumberOfTrailingOnes( state_[j] );
            if( numberOfTrailingOnes[j] < minTrailingOnes )
            {
                minDim = j;
                minTrailingOnes = numberOfTrailingOnes[j];
            }
        }
        nextZeroDim_ = minDim;
        nextZeroLevel_ = minTrailingOnes;
    }
    else
    {
        for( size_t j=0; j<=zeroDim; ++j )
            state_[j] &= ~((1u<<(zeroLevel+1))-1);
        for( size_t j=zeroDim+1; j<d; ++j )
            state_[j] &= ~((1u<<zeroLevel)-1);
        state_[zeroDim] |= 1u<<zeroLevel;

        // Set up for the next walk
        nextZeroDim_ = 0;
        nextZeroLevel_ = 0;
    }
}

} // dbf

#endif // ifndef DBF_STRUCTURES_HTREE_WALKER_HPP
