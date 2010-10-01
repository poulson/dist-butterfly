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
#ifndef BFIO_STRUCTURES_ARRAY_HPP
#define BFIO_STRUCTURES_ARRAY_HPP 1

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
    Array() { }
    Array( T alpha ) { for( std::size_t j=0; j<d; ++j ) _x[j] = alpha; }
    ~Array() { }

    T& operator[]( std::size_t j ) { return _x[j]; }
    const T& operator[]( std::size_t j ) const { return _x[j]; }

    const Array<T,d>&
    operator=( const Array<T,d>& array )
    {
        std::memcpy( _x, &array[0], d*sizeof(T) );
        return *this;
    }
};

} // bfio

#endif // BFIO_STRUCTURES_ARRAY_HPP

