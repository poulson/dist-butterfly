/*
   Copyright (c) 2010, Jack Poulson
   All rights reserved.

   This file is part of ButterflyFIO.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are met:

    - Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    - Neither the name of the owner nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
   ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
   LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
   CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
   SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
   INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
   CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
   ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
   POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once
#ifndef BFIO_HTREE_WALKER_HPP
#define BFIO_HTREE_WALKER_HPP 1

#include <stdexcept>
#include "bfio/structures/data.hpp"
#include "bfio/tools/twiddle.hpp"

namespace bfio {

template<unsigned d>
class HTreeWalker
{
    unsigned _nextZeroDim;
    unsigned _nextZeroLevel;
    Array<unsigned,d> _state;
public:
    HTreeWalker() : _nextZeroDim(0), _nextZeroLevel(0), _state(0) {}
    ~HTreeWalker() {}

    Array<unsigned,d> State()
    { return _state; }

    void Walk()
    {
        const unsigned zeroDim = _nextZeroDim;
        const unsigned zeroLevel = _nextZeroLevel;

        if( zeroDim == 0 )
        {
            // Zero the first (zeroLevel-1) bits of all coordinates
            // and then increment at level zeroLevel
            for( unsigned j=0; j<d; ++j )
                _state[j] &= ~((1u<<zeroLevel)-1);
            _state[zeroDim] |= 1u<<zeroLevel;

            // Set up for the next walk
            // We need to find the dimension with the first zero bit.
            unsigned minDim = d;
            unsigned minTrailingOnes = sizeof(unsigned)*8+1;
            Array<unsigned,d> numberOfTrailingOnes;
            for( unsigned j=0; j<d; ++j )
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
            for( unsigned j=0; j<=zeroDim; ++j )
                _state[j] &= ~((1u<<(zeroLevel+1))-1);
            for( unsigned j=zeroDim+1; j<d; ++j )
                _state[j] &= ~((1u<<zeroLevel)-1);
            _state[zeroDim] |= 1u<<zeroLevel;

            // Set up for the next walk
            _nextZeroDim = 0;
            _nextZeroLevel = 0;
        }
    }

    Array<unsigned,d> NextState()
    {
        Walk();
        return State();
    }
};

// Constrained HTree Walker
template<unsigned d>
class CHTreeWalker
{
    bool _overflowed;
    unsigned _firstOpenDim;
    unsigned _nextZeroDim;
    unsigned _nextZeroLevel;
    Array<unsigned,d> _state;
    Array<unsigned,d> _log2BoxesPerDim;
public:
    CHTreeWalker( const Array<unsigned,d>& log2BoxesPerDim ) 
    : _overflowed(false), _nextZeroLevel(0), _state(0), 
      _log2BoxesPerDim(log2BoxesPerDim) 
    {
        for( _firstOpenDim=0; _firstOpenDim<d; ++_firstOpenDim )
            if( log2BoxesPerDim[_firstOpenDim] != 0 )
                break;
        _nextZeroDim = _firstOpenDim;
    }

    ~CHTreeWalker() {}

    Array<unsigned,d> State()
    { 
        if( _overflowed )
            throw std::logic_error( "Overflowed HTree" );
        return _state; 
    }

    void Walk()
    {
        if( _nextZeroDim == d )
        {
            _overflowed = true;
            return;
        }

        const unsigned zeroDim = _nextZeroDim;
        const unsigned zeroLevel = _nextZeroLevel;

        if( zeroDim == _firstOpenDim )
        {
            // Zero the first (zeroLevel-1) bits of all coordinates
            // and then increment at level zeroLevel
            for( unsigned j=0; j<d; ++j )
                _state[j] &= ~((1u<<zeroLevel)-1);
            _state[zeroDim] |= 1u<<zeroLevel;

            // Set up for the next walk
            // We need to find the dimension with the first unconstrained
            // zero bit.
            unsigned minDim = d;
            unsigned minTrailingOnes = sizeof(unsigned)*8+1; 
            Array<unsigned,d> numberOfTrailingOnes;
            for( unsigned j=0; j<d; ++j )
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
            for( unsigned j=0; j<=zeroDim; ++j )
                _state[j] &= ~((1u<<(zeroLevel+1))-1);
            for( unsigned j=zeroDim+1; j<d; ++j )
                _state[j] &= ~((1u<<zeroLevel)-1);
            _state[zeroDim] |= 1u<<zeroLevel;

            // Set up for the next walk
            _nextZeroDim = _firstOpenDim;
            _nextZeroLevel = 0;
        }
    }
};

} // bfio

#endif /* BFIO_HTREE_WALKER_HPP */

