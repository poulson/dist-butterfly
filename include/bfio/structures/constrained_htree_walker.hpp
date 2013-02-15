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
    bool overflowed_;
    size_t firstOpenDim_;
    size_t nextZeroDim_;
    size_t nextZeroLevel_;
    array<size_t,d> state_;
    array<size_t,d> log2BoxesPerDim_;
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
: overflowed_(false), nextZeroLevel_(0),
  log2BoxesPerDim_(log2BoxesPerDim) 
{
    state_.fill(0);
    for( firstOpenDim_=0; firstOpenDim_<d; ++firstOpenDim_ )
        if( log2BoxesPerDim[firstOpenDim_] != 0 )
            break;
    nextZeroDim_ = firstOpenDim_;
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
    if( overflowed_ )
        throw std::logic_error( "Overflowed HTree" );
#endif
    return state_; 
}

template<size_t d>
void 
ConstrainedHTreeWalker<d>::Walk()
{
#ifndef RELEASE
    if( nextZeroDim_ == d )
    {
        overflowed_ = true;
        return;
    }
#endif

    const size_t zeroDim = nextZeroDim_;
    const size_t zeroLevel = nextZeroLevel_;

    if( zeroDim == firstOpenDim_ )
    {
        // Zero the first (zeroLevel-1) bits of all coordinates
        // and then increment at level zeroLevel
        for( size_t j=0; j<d; ++j )
            state_[j] &= ~((1u<<zeroLevel)-1);
        state_[zeroDim] |= 1u<<zeroLevel;

        // Set up for the next walk
        // We need to find the dimension with the first unconstrained
        // zero bit.
        size_t minDim = d;
        size_t minTrailingOnes = sizeof(size_t)*8+1; 
        array<size_t,d> numberOfTrailingOnes;
        for( size_t j=0; j<d; ++j )
        {
            numberOfTrailingOnes[j] = NumberOfTrailingOnes( state_[j] );
            if( (numberOfTrailingOnes[j] < minTrailingOnes) &&
                (numberOfTrailingOnes[j] != log2BoxesPerDim_[j]) )
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
        nextZeroDim_ = firstOpenDim_;
        nextZeroLevel_ = 0;
    }
}

} // bfio

#endif // ifndef BFIO_STRUCTURES_CONSTRAINED_HTREE_WALKER_HPP
