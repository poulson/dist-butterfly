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
#pragma once
#ifndef BFIO_DATA_HPP
#define BFIO_DATA_HPP 1

#include <complex>

#include "bfio/constants.hpp"

namespace bfio {

// A d-dimensional point over arbitrary datatype T
template<typename T,unsigned d>
class Array
{
    T _x[d];
public:
    Array() { }
    Array( T alpha ) { for( unsigned j=0; j<d; ++j ) _x[j] = alpha; }
    ~Array() { }

    T& operator[]( unsigned j ) { return _x[j]; }
    const T& operator[]( unsigned j ) const { return _x[j]; }

    const Array<T,d>&
    operator=( const Array<T,d>& array )
    {
        for( unsigned j=0; j<d; ++j )
            _x[j] = array[j];
        return *this;
    }
};

template<typename R,unsigned d>
struct Box
{
    Array<R,d> widths;
    Array<R,d> offsets;
};

template<typename R,unsigned d>
struct Potential
{
    Array<R,d> x;
    std::complex<R> magnitude;
};

template<typename R,unsigned d>
struct Source 
{ 
    Array<R,d> p;
    std::complex<R> magnitude;
};

template<typename R,unsigned d,unsigned q>
class PointGrid
{
    Array< Array<R,d>, Pow<q,d>::val > _points;

public:
    PointGrid() {}
    ~PointGrid() {}

    const Array<R,d>&
    operator[] ( unsigned i ) const
    { return _points[i]; }

    Array<R,d>&
    operator[] ( unsigned i )
    { return _points[i]; }

    const PointGrid<R,d,q>&
    operator= ( const PointGrid<R,d,q>& pointGrid )
    {
        const unsigned q_to_d = Pow<q,d>::val;
        for( unsigned j=0; j<q_to_d; ++j )
            _points[j] = pointGrid[j];
        return *this;
    }
};

template<typename R,unsigned d,unsigned q>
class WeightGrid
{
    Array< std::complex<R>, Pow<q,d>::val > _weights;

public:
    WeightGrid() {}
    ~WeightGrid() {}

    const std::complex<R>&
    operator[] ( unsigned i ) const
    { return _weights[i]; }

    std::complex<R>&
    operator[] ( unsigned i )
    { return _weights[i]; }

    const WeightGrid<R,d,q>&
    operator= ( const WeightGrid<R,d,q>& weightGrid )
    { 
        const unsigned q_to_d = Pow<q,d>::val;
        for( unsigned j=0; j<q_to_d; ++j )
            _weights[j] = weightGrid[j];
        return *this;
    }
};

template<typename R,unsigned d,unsigned q>
class WeightGridList
{
    unsigned _length;
    WeightGrid<R,d,q>* _weightGridList;

public:
    WeightGridList( unsigned length ) 
        : _length(length), _weightGridList(new WeightGrid<R,d,q>[length])
    { }

    WeightGridList( const WeightGridList<R,d,q>& weightGridList )
        : _length(weightGridList.Length()),
          _weightGridList(new WeightGrid<R,d,q>[weightGridList.Length()])
    {
        for( unsigned j=0; j<_length; ++j )
            _weightGridList[j] = weightGridList[j];
    }

    ~WeightGridList() 
    { delete[] _weightGridList; }

    unsigned
    Length() const
    { return _length; }

    const WeightGrid<R,d,q>& 
    operator[] ( unsigned i ) const
    { return _weightGridList[i]; }

    WeightGrid<R,d,q>& 
    operator[] ( unsigned i )
    { return _weightGridList[i]; }

    const WeightGridList<R,d,q>&
    operator=  ( const WeightGridList<R,d,q>& weightGridList )
    { 
        _length = weightGridList.Length();
        for( unsigned j=0; j<_length; ++j )
            _weightGridList[j] = weightGridList[j];
        return *this;
    }
};

} // bfio

#endif // BFIO_DATA_HPP

