/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of ButterflyFIO and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef BFIO_STRUCTURES_WEIGHT_GRID_HPP
#define BFIO_STRUCTURES_WEIGHT_GRID_HPP

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
    bool hasBuffer_;
    bool ownsBuffer_;
    R* buffer_;
    R* realBuffer_;
    R* imagBuffer_;

public:
    WeightGrid();
    WeightGrid( bool createBuffer );
    WeightGrid( const WeightGrid<R,d,q>& weightGrid );
    ~WeightGrid();

    // This buffer must be of length 2*q^d
    void AttachBuffer( R* buffer, bool givingBuffer );

    bool HasBuffer() const;
    bool OwnsBuffer() const;
    const R* Buffer() const;
    R* Buffer();
    const R* RealBuffer() const;
          R* RealBuffer();
    const R* ImagBuffer() const;
          R* ImagBuffer();
    const R& RealWeight( std::size_t i ) const;
          R& RealWeight( std::size_t i );
    const R& ImagWeight( std::size_t i ) const;
          R& ImagWeight( std::size_t i );

    const WeightGrid<R,d,q>&
    operator= ( const WeightGrid<R,d,q>& weightGrid );
};

// Implementations

template<typename R,std::size_t d,std::size_t q>
WeightGrid<R,d,q>::WeightGrid()
: hasBuffer_(true), ownsBuffer_(true)
{
    const std::size_t q_to_d = Pow<q,d>::val;
    buffer_ = new R[2*q_to_d];
    realBuffer_ = &buffer_[0];
    imagBuffer_ = &buffer_[q_to_d];
}

template<typename R,std::size_t d,std::size_t q>
WeightGrid<R,d,q>::WeightGrid( bool createBuffer ) 
: hasBuffer_(createBuffer), ownsBuffer_(createBuffer)
{
    if( createBuffer )
    {
        const std::size_t q_to_d = Pow<q,d>::val;
        buffer_ = new R[2*q_to_d];
        realBuffer_ = &buffer_[0];
        imagBuffer_ = &buffer_[q_to_d];
    }
    else
    {
        buffer_ = 0;
        realBuffer_ = 0;
        imagBuffer_ = 0;
    }
}

template<typename R,std::size_t d,std::size_t q>
WeightGrid<R,d,q>::WeightGrid( const WeightGrid<R,d,q>& weightGrid )
: hasBuffer_(weightGrid.HasBuffer()), ownsBuffer_(weightGrid.HasBuffer())
{
    if( weightGrid.HasBuffer() )
    {
        const std::size_t q_to_d = Pow<q,d>::val;
        buffer_ = new R[2*q_to_d];
        std::memcpy( buffer_, weightGrid.Buffer(), 2*q_to_d*sizeof(R) );
        realBuffer_ = &buffer_[0];
        imagBuffer_ = &buffer_[q_to_d];
    }
    else
    {
        buffer_ = 0;
        realBuffer_ = 0;
        imagBuffer_ = 0;
    }
}

template<typename R,std::size_t d,std::size_t q>
inline
WeightGrid<R,d,q>::~WeightGrid() 
{ 
    if( ownsBuffer_ ) 
        delete[] buffer_;
}

// This buffer must be of length 2*q^d
template<typename R,std::size_t d,std::size_t q>
void
WeightGrid<R,d,q>::AttachBuffer( R* buffer, bool givingBuffer )
{ 
    if( ownsBuffer_ )
        delete[] buffer_;
    buffer_ = buffer;
    realBuffer_ = &buffer_[0];
    imagBuffer_ = &buffer_[Pow<q,d>::val];

    hasBuffer_ = true;
    ownsBuffer_ = givingBuffer;
}

template<typename R,std::size_t d,std::size_t q>
inline bool
WeightGrid<R,d,q>::HasBuffer() const
{ return hasBuffer_; }

template<typename R,std::size_t d,std::size_t q>
inline bool
WeightGrid<R,d,q>::OwnsBuffer() const
{ return ownsBuffer_; }

template<typename R,std::size_t d,std::size_t q>
inline const R*
WeightGrid<R,d,q>::Buffer() const
{ return buffer_; }

template<typename R,std::size_t d,std::size_t q>
inline R*
WeightGrid<R,d,q>::Buffer()
{ return buffer_; }

template<typename R,std::size_t d,std::size_t q>
inline const R*
WeightGrid<R,d,q>::RealBuffer() const
{ return realBuffer_; }

template<typename R,std::size_t d,std::size_t q>
inline R*
WeightGrid<R,d,q>::RealBuffer()
{ return realBuffer_; }

template<typename R,std::size_t d,std::size_t q>
inline const R*
WeightGrid<R,d,q>::ImagBuffer() const
{ return imagBuffer_; }

template<typename R,std::size_t d,std::size_t q>
inline R*
WeightGrid<R,d,q>::ImagBuffer()
{ return imagBuffer_; }

template<typename R,std::size_t d,std::size_t q>
inline const R&
WeightGrid<R,d,q>::RealWeight( std::size_t i ) const
{ return realBuffer_[i]; }

template<typename R,std::size_t d,std::size_t q>
inline R&
WeightGrid<R,d,q>::RealWeight( std::size_t i )
{ return realBuffer_[i]; }

template<typename R,std::size_t d,std::size_t q>
inline const R&
WeightGrid<R,d,q>::ImagWeight( std::size_t i ) const
{ return imagBuffer_[i]; }

template<typename R,std::size_t d,std::size_t q>
inline R&
WeightGrid<R,d,q>::ImagWeight( std::size_t i ) 
{ return imagBuffer_[i]; }

template<typename R,std::size_t d,std::size_t q>
const WeightGrid<R,d,q>&
WeightGrid<R,d,q>::operator=( const WeightGrid<R,d,q>& weightGrid )
{ 
    if( weightGrid.HasBuffer() )
    {
        if( !hasBuffer_ )
        {
            buffer_ = new R[2*Pow<q,d>::val];    
            hasBuffer_ = true;
            ownsBuffer_ = true;
        }
        std::memcpy( buffer_, weightGrid.Buffer(), 2*Pow<q,d>::val*sizeof(R) );
    }
    else
    {
        if( ownsBuffer_ )
        {
            delete[] buffer_;
            ownsBuffer_ = false;
        }
        hasBuffer_ = false;
        buffer_ = 0;
        realBuffer_ = 0;
        imagBuffer_ = 0;
    }
    return *this;
}

} // bfio

#endif // ifndef BFIO_STRUCTURES_WEIGHT_GRID_HPP
