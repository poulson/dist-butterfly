/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of ButterflyFIO and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef BFIO_STRUCTURES_ARRAY_HPP
#define BFIO_STRUCTURES_ARRAY_HPP

#include <cstddef>
#include <cstring>

namespace bfio {

// A d-dimensional point over arbitrary datatype T.
// Both boost::array and Array provide similar functionality, but 
// TR1 is not yet standardized and Boost is not always available.
template<typename T,std::size_t d>
class Array
{
    T _x[d];
public:
    Array();
    Array( T alpha );
    ~Array();

    T& operator[]( std::size_t j );
    const T& operator[]( std::size_t j ) const;

    const Array<T,d>&
    operator=( const Array<T,d>& array );
};

// Implementations

template<typename T,std::size_t d>
inline Array<T,d>::Array() 
{ }

template<typename T,std::size_t d>
inline Array<T,d>::Array( T alpha ) 
{ for( std::size_t j=0; j<d; ++j ) _x[j] = alpha; }

template<typename T,std::size_t d>
inline Array<T,d>::~Array() 
{ }

template<typename T,std::size_t d>
inline T& 
Array<T,d>::operator[]( std::size_t j ) 
{ return _x[j]; }

template<typename T,std::size_t d>
inline const T& 
Array<T,d>::operator[]( std::size_t j ) const 
{ return _x[j]; }

template<typename T,std::size_t d>
inline const Array<T,d>&
Array<T,d>::operator=( const Array<T,d>& array )
{
    std::memcpy( _x, &array[0], d*sizeof(T) );
    return *this;
}

} // bfio

#endif // ifndef BFIO_STRUCTURES_ARRAY_HPP
