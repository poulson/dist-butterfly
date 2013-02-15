/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of ButterflyFIO and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef BFIO_STRUCTURES_WEIGHT_GRID_LIST_HPP
#define BFIO_STRUCTURES_WEIGHT_GRID_LIST_HPP

#include <vector>
#include "bfio/structures/weight_grid.hpp"

namespace bfio {

// This class provides a list of weight grids whose buffers are guaranteed to 
// be stored contiguously
template<typename R,std::size_t d,std::size_t q>
class WeightGridList
{
    const std::size_t length_;
    std::vector<R> buffer_;
    std::vector<WeightGrid<R,d,q>> weightGrids_;

public:
    WeightGridList( std::size_t length );
    WeightGridList( const WeightGridList<R,d,q>& weightGridList );
    ~WeightGridList();

    const R* Buffer() const;
          R* Buffer();
    std::size_t Length() const;

    const WeightGrid<R,d,q>& 
    operator[] ( std::size_t i ) const;

    WeightGrid<R,d,q>& 
    operator[] ( std::size_t i );

    const WeightGridList<R,d,q>&
    operator=  ( const WeightGridList<R,d,q>& weightGridList );
};

// Implementations

template<typename R,std::size_t d,std::size_t q>
WeightGridList<R,d,q>::WeightGridList( std::size_t length ) 
: length_(length)
{ 
    // Create space for the data
    const std::size_t weightGridSize = 2*Pow<q,d>::val;
    buffer_.resize( weightGridSize*length );

    // Create the views of the data
    weightGrids_.reserve( length );
    for( std::size_t j=0; j<length; ++j )
    {
        weightGrids_.push_back( WeightGrid<R,d,q>( false ) );
        weightGrids_[j].AttachBuffer( &buffer_[j*weightGridSize], false );
    }
}

template<typename R,std::size_t d,std::size_t q>
WeightGridList<R,d,q>::WeightGridList
( const WeightGridList<R,d,q>& weightGridList )
: length_(weightGridList.Length())
{
    // Copy the data
    const std::size_t weightGridSize = 2*Pow<q,d>::val;
    buffer_.resize( weightGridSize*length_ );
    std::memcpy
    ( &buffer_[0], weightGridList.Buffer(), length_*weightGridSize*sizeof(R) );

    // Create the views of the copied data
    weightGrids_.reserve( length_ );
    for( std::size_t j=0; j<length_; ++j )
    {
        weightGrids_.push_back( WeightGrid<R,d,q>( false ) );
        weightGrids_[j].AttachBuffer( &buffer_[j*weightGridSize], false );
    }
}

template<typename R,std::size_t d,std::size_t q>
inline 
WeightGridList<R,d,q>::~WeightGridList() 
{ }

template<typename R,std::size_t d,std::size_t q>
inline const R*
WeightGridList<R,d,q>::Buffer() const
{ return &buffer_[0]; }

template<typename R,std::size_t d,std::size_t q>
inline R*
WeightGridList<R,d,q>::Buffer()
{ return &buffer_[0]; }

template<typename R,std::size_t d,std::size_t q>
inline std::size_t
WeightGridList<R,d,q>::Length() const
{ return length_; }

template<typename R,std::size_t d,std::size_t q>
inline const WeightGrid<R,d,q>& 
WeightGridList<R,d,q>::operator[]
( std::size_t i ) const
{ return weightGrids_[i]; }

template<typename R,std::size_t d,std::size_t q>
inline WeightGrid<R,d,q>& 
WeightGridList<R,d,q>::operator[]
( std::size_t i )
{ return weightGrids_[i]; }

template<typename R,std::size_t d,std::size_t q>
const WeightGridList<R,d,q>&
WeightGridList<R,d,q>::operator=
( const WeightGridList<R,d,q>& weightGridList )
{ 
    // Ensure that we have a large enough buffer
    const std::size_t weightGridSize = 2*Pow<q,d>::val;
    length_ = weightGridList.Length();
    buffer_.resize( weightGridSize*length_ );

    // Copy the data over
    std::memcpy
    ( &buffer_[0], weightGridList.Buffer(), length_*weightGridSize*sizeof(R) );

    // Create the views of the data
    weightGrids_.reserve( length_ );
    for( std::size_t j=0; j<length_; ++j )
    {
        weightGrids_.push_back( WeightGrid<R,d,q>( false ) );
        weightGrids_[j].AttachBuffer( &buffer_[j*weightGridSize], false );
    }

    return *this;
}

} // bfio

#endif // ifndef BFIO_STRUCTURES_WEIGHT_GRID_LIST_HPP
