/*
   ButterflyFIO: a distributed-memory fast algorithm for applying FIOs.
   Copyright (C) 2010 Jack Poulson <jack.poulson@gmail.com>
 
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
#ifndef BFIO_STRUCTURES_WEIGHT_GRID_HPP
#define BFIO_STRUCTURES_WEIGHT_GRID_HPP 1

#include <cstddef>
#include <cstring>
#include "bfio/constants.hpp"

namespace bfio {

template<typename R,std::size_t d,std::size_t q>
class WeightGrid
{
    // We know the size is 2*q^d, but it's a bad idea to keep this on the stack.
    // We will use this to contiguously store the real, and then imaginary, 
    // components of the weights.
    bool _hasBuffer;
    bool _ownsBuffer;
    R* _buffer;
    R* _realBuffer;
    R* _imagBuffer;

public:
    WeightGrid()
    : _hasBuffer(true), _ownsBuffer(true)
    {
        const std::size_t q_to_d = Pow<q,d>::val;
        _buffer = new R[2*q_to_d];
        _realBuffer = &_buffer[0];
        _imagBuffer = &_buffer[q_to_d];
    }

    WeightGrid( bool createBuffer ) 
    : _hasBuffer(createBuffer), _ownsBuffer(createBuffer)
    {
        if( createBuffer )
        {
            const std::size_t q_to_d = Pow<q,d>::val;
            _buffer = new R[2*q_to_d];
            _realBuffer = &_buffer[0];
            _imagBuffer = &_buffer[q_to_d];
        }
        else
        {
            _buffer = 0;
            _realBuffer = 0;
            _imagBuffer = 0;
        }
    }

    WeightGrid( const WeightGrid<R,d,q>& weightGrid )
    : _hasBuffer(weightGrid.HasBuffer()), _ownsBuffer(weightGrid.HasBuffer())
    {
        if( weightGrid.HasBuffer() )
        {
            const std::size_t q_to_d = Pow<q,d>::val;
            _buffer = new R[2*q_to_d];
            std::memcpy( _buffer, weightGrid.Buffer(), 2*q_to_d*sizeof(R) );
            _realBuffer = &_buffer[0];
            _imagBuffer = &_buffer[q_to_d];
        }
        else
        {
            _buffer = 0;
            _realBuffer = 0;
            _imagBuffer = 0;
        }
    }

    ~WeightGrid() 
    { 
        if( _ownsBuffer ) 
            delete[] _buffer;
    }

    // This buffer must be of length 2*q^d
    void
    AttachBuffer( R* buffer, bool givingBuffer )
    { 
        if( _ownsBuffer )
            delete[] _buffer;
        _buffer = buffer;
        _realBuffer = &_buffer[0];
        _imagBuffer = &_buffer[Pow<q,d>::val];

        _hasBuffer = true;
        _ownsBuffer = givingBuffer;
    }

    bool
    HasBuffer() const
    { return _hasBuffer; }

    bool
    OwnsBuffer() const
    { return _ownsBuffer; }

    const R*
    Buffer() const
    { return _buffer; }

    R*
    Buffer()
    { return _buffer; }

    const R*
    RealBuffer() const
    { return _realBuffer; }

    R*
    RealBuffer()
    { return _realBuffer; }

    const R*
    ImagBuffer() const
    { return _imagBuffer; }

    R*
    ImagBuffer()
    { return _imagBuffer; }

    const R&
    RealWeight( std::size_t i ) const
    { return _realBuffer[i]; }

    R&
    RealWeight( std::size_t i )
    { return _realBuffer[i]; }

    const R&
    ImagWeight( std::size_t i ) const
    { return _imagBuffer[i]; }

    R&
    ImagWeight( std::size_t i ) 
    { return _imagBuffer[i]; }

    const WeightGrid<R,d,q>&
    operator= ( const WeightGrid<R,d,q>& weightGrid )
    { 
        if( weightGrid.HasBuffer() )
        {
            if( !_hasBuffer )
            {
                _buffer = new R[2*Pow<q,d>::val];    
                _hasBuffer = true;
                _ownsBuffer = true;
            }
            std::memcpy
            ( _buffer, weightGrid.Buffer(), 2*Pow<q,d>::val*sizeof(R) );
        }
        else
        {
            if( _ownsBuffer )
            {
                delete[] _buffer;
                _ownsBuffer = false;
            }
            _hasBuffer = false;
            _buffer = 0;
            _realBuffer = 0;
            _imagBuffer = 0;
        }
        return *this;
    }
};

} // bfio

#endif // BFIO_STRUCTURES_WEIGHT_GRID_HPP

