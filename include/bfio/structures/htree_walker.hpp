/*
  Copyright 2010 Jack Poulson

  This file is part of ButterflyFIO.

  This program is free software: you can redistribute it and/or modify it under
  the terms of the GNU Lesser General Public License as published by the
  Free Software Foundation; either version 3 of the License, or 
  (at your option) any later version.

  This program is distributed in the hope that it will be useful, but 
  WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef BFIO_HTREE_WALKER_HPP
#define BFIO_HTREE_WALKER_HPP 1

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
            throw "Overflowed HTree.";
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

