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

#endif /* BFIO_DATA_HPP */

