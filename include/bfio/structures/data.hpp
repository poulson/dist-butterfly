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
#ifndef BFIO_STRUCTURES_DATA_HPP
#define BFIO_STRUCTURES_DATA_HPP 1

#include <complex>
#include <cstring>
#include <tr1/array>

#include "bfio/constants.hpp"

namespace bfio {

template<typename R,std::size_t d>
struct Box
{
    std::tr1::array<R,d> widths;
    std::tr1::array<R,d> offsets;
};

template<typename R,std::size_t d>
struct Potential
{
    std::tr1::array<R,d> x;
    std::complex<R> magnitude;
};

template<typename R,std::size_t d>
struct Source 
{ 
    std::tr1::array<R,d> p;
    std::complex<R> magnitude;
};

template<typename R,std::size_t d,std::size_t q>
class PointGrid
{
    // We know the size should be q^d at compile time, but we do not want the
    // data stored on the stack
    std::vector< std::tr1::array<R,d> > _points;

public:
    PointGrid() : _points(Pow<q,d>::val) {}
    ~PointGrid() {}

    const std::tr1::array<R,d>&
    operator[] ( std::size_t i ) const
    { return _points[i]; }

    std::tr1::array<R,d>&
    operator[] ( std::size_t i )
    { return _points[i]; }

    const PointGrid<R,d,q>&
    operator= ( const PointGrid<R,d,q>& pointGrid )
    {
        const std::size_t q_to_d = Pow<q,d>::val;
        for( std::size_t j=0; j<q_to_d; ++j )
            _points[j] = pointGrid[j];
        return *this;
    }
};

template<typename R,std::size_t d,std::size_t q>
class WeightGrid
{
    // We know the size is 2*q^d, but it's a bad idea to keep this on the stack.
    // We will use this to contiguously store the real, and then imaginary, 
    // components of the weights.
    bool _hasBuffer;
    bool _ownsBuffer;
    R* _buffer;
    R* _realWeights;
    R* _imagWeights;

public:
    WeightGrid()
    : _hasBuffer(true), _ownsBuffer(true)
    {
        _buffer = new R[2*Pow<q,d>::val];
        _realWeights = &_buffer[0];
        _imagWeights = &_buffer[Pow<q,d>::val];
    }

    WeightGrid( bool createBuffer ) 
    : _hasBuffer(createBuffer), _ownsBuffer(createBuffer)
    {
        if( createBuffer )
        {
            _buffer = new R[2*Pow<q,d>::val];
            _realWeights = &_buffer[0];
            _imagWeights = &_buffer[Pow<q,d>::val];
        }
        else
        {
            _buffer = 0;
            _realWeights = 0;
            _imagWeights = 0;
        }
    }

    WeightGrid( const WeightGrid<R,d,q>& weightGrid )
    : _hasBuffer(weightGrid.HasBuffer()), _ownsBuffer(weightGrid.HasBuffer())
    {
        if( weightGrid.HasBuffer() )
        {
            _buffer = new R[2*Pow<q,d>::val];
            std::memcpy
            ( _buffer, weightGrid.Buffer(), 2*Pow<q,d>::val*sizeof(R) );
            _realWeights = &_buffer[0];
            _imagWeights = &_buffer[Pow<q,d>::val];
        }
        else
        {
            _buffer = 0;
            _realWeights = 0;
            _imagWeights = 0;
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
        _realWeights = &_buffer[0];
        _imagWeights = &_buffer[Pow<q,d>::val];

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

    const R&
    RealWeight( std::size_t i ) const
    { return _realWeights[i]; }

    R&
    RealWeight( std::size_t i )
    { return _realWeights[i]; }

    const R&
    ImagWeight( std::size_t i ) const
    { return _imagWeights[i]; }

    R&
    ImagWeight( std::size_t i ) 
    { return _imagWeights[i]; }

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
            _realWeights = 0;
            _imagWeights = 0;
        }
        return *this;
    }
};

// This class provides a list of weight grids whose buffers are guaranteed to 
// be stored contiguously
template<typename R,std::size_t d,std::size_t q>
class WeightGridList
{
    const std::size_t _length;
    std::vector<R> _buffer;
    std::vector< WeightGrid<R,d,q> > _weightGrids;

public:
    WeightGridList( std::size_t length ) 
    : _length(length)
    { 
        // Create space for the data
        const std::size_t weightGridSize = 2*Pow<q,d>::val;
        _buffer.resize( weightGridSize*length );

        // Create the views of the data
        _weightGrids.reserve( length );
        for( std::size_t j=0; j<length; ++j )
        {
            _weightGrids.push_back( WeightGrid<R,d,q>( false ) );
            _weightGrids[j].AttachBuffer( &_buffer[j*weightGridSize], false );
        }
    }

    WeightGridList( const WeightGridList<R,d,q>& weightGridList )
    : _length(weightGridList.Length())
    {
        // Copy the data
        const std::size_t weightGridSize = 2*Pow<q,d>::val;
        _buffer.resize( weightGridSize*_length );
        std::memcpy
        ( &_buffer[0], weightGridList.Buffer(), 
          _length*weightGridSize*sizeof(R) );

        // Create the views of the copied data
        _weightGrids.reserve( _length );
        for( std::size_t j=0; j<_length; ++j )
        {
            _weightGrids.push_back( WeightGrid<R,d,q>( false ) );
            _weightGrids[j].AttachBuffer( &_buffer[j*weightGridSize], false );
        }
    }

    ~WeightGridList() {}

    const R*
    Buffer() const
    { return &_buffer[0]; }

    R*
    Buffer()
    { return &_buffer[0]; }

    std::size_t
    Length() const
    { return _length; }

    const WeightGrid<R,d,q>& 
    operator[] ( std::size_t i ) const
    { return _weightGrids[i]; }

    WeightGrid<R,d,q>& 
    operator[] ( std::size_t i )
    { return _weightGrids[i]; }

    const WeightGridList<R,d,q>&
    operator=  ( const WeightGridList<R,d,q>& weightGridList )
    { 
        // Ensure that we have a large enough buffer
        const std::size_t weightGridSize = 2*Pow<q,d>::val;
        _length = weightGridList.Length();
        _buffer.resize( weightGridSize*_length );

        // Copy the data over
        std::memcpy
        ( &_buffer[0], weightGridList.Buffer(), 
          _length*weightGridSize*sizeof(R) );

        // Create the views of the data
        _weightGrids.reserve( _length );
        for( std::size_t j=0; j<_length; ++j )
        {
            _weightGrids.push_back( WeightGrid<R,d,q>( false ) );
            _weightGrids[j].AttachBuffer( &_buffer[j*weightGridSize], false );
        }

        return *this;
    }
};

} // bfio

#endif // BFIO_STRUCTURES_DATA_HPP

