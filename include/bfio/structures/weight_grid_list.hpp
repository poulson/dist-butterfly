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
#ifndef BFIO_STRUCTURES_WEIGHT_GRID_LIST_HPP
#define BFIO_STRUCTURES_WEIGHT_GRID_LIST_HPP 1

#include <vector>
#include "bfio/structures/weight_grid.hpp"

namespace bfio {

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

#endif // BFIO_STRUCTURES_WEIGHT_GRID_LIST_HPP

