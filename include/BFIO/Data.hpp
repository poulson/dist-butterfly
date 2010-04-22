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
#ifndef BFIO_DATA_HPP
#define BFIO_DATA_HPP 1

#include <complex>
#include "BFIO/Pow.hpp"

namespace BFIO
{
    using namespace std;
    static const double Pi    = 3.141592653589793;
    static const double TwoPi = 6.283185307179586;

    // A d-dimensional point over arbitrary datatype T
    template<typename T,unsigned d>
    class Array
    {
        T _x[d];
    public:
        Array() { }
        Array( T val ) { for( unsigned j=0; j<d; ++j ) _x[j] = val; }
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

    // A d-dimensional coordinate in the frequency domain and the 
    // magnitude of the source located there
    template<typename R,unsigned d>
    struct Source 
    { 
        Array<R,d> p;
        complex<R> magnitude;
    };

    template<typename R,unsigned d,unsigned q>
    class WeightSet
    {
        Array< complex<R>, Pow<q,d>::val > _weight;

    public:
        WeightSet() { }
        ~WeightSet() { }

        const complex<R>&
        operator[] ( const unsigned i ) const
        { return _weight[i]; }

        complex<R>&
        operator[] ( const unsigned i )
        { return _weight[i]; }

        const WeightSet<R,d,q>&
        operator= ( const WeightSet<R,d,q>& weightSet )
        { 
            for( unsigned j=0; j<Pow<q,d>::val; ++j )
                _weight[j] = weightSet[j];
            return *this;
        }
    };

    template<typename R,unsigned d,unsigned q>
    class WeightSetList
    {
        unsigned _length;
        WeightSet<R,d,q>* _weightSetList;

    public:
        WeightSetList( unsigned length ) 
            : _length(length), _weightSetList(new WeightSet<R,d,q>[length])
        { }

        WeightSetList( const WeightSetList<R,d,q>& weightSetList )
            : _length(weightSetList.Length()),
              _weightSetList(new WeightSet<R,d,q>[weightSetList.Length()])
        {
            for( unsigned j=0; j<_length; ++j )
                _weightSetList[j] = weightSetList[j];
        }

        ~WeightSetList() 
        { delete[] _weightSetList; }

        const unsigned
        Length() const
        { return _length; }

        const WeightSet<R,d,q>& 
        operator[] ( unsigned i ) const
        { return _weightSetList[i]; }

        WeightSet<R,d,q>& 
        operator[] ( unsigned i )
        { return _weightSetList[i]; }

        const WeightSetList<R,d,q>&
        operator=  ( const WeightSetList<R,d,q>& weightSetList )
        { 
            _length = weightSetList.Length();
            for( unsigned j=0; j<_length; ++j )
                _weightSetList[j] = weightSetList[j];
            return *this;
        }
    };
}

#endif /* BFIO_DATA_HPP */

