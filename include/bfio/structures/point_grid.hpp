/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of ButterflyFIO and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef BFIO_STRUCTURES_POINT_GRID_HPP
#define BFIO_STRUCTURES_POINT_GRID_HPP

#include <array>
#include <cstring>
#include <vector>

#include "bfio/constants.hpp"

namespace bfio {

using std::array;
using std::memcpy;
using std::size_t;
using std::vector;

template<typename R,size_t d,size_t q>
class PointGrid
{
    // We know the size should be q^d at compile time, but we do not want the
    // data stored on the stack
    vector<array<R,d>> _points;

public:
    PointGrid();
    ~PointGrid();

    const array<R,d>& operator[] ( size_t i ) const;
          array<R,d>& operator[] ( size_t i );

    const PointGrid<R,d,q>& operator= ( const PointGrid<R,d,q>& pointGrid );
};

// Implementations

template<typename R,size_t d,size_t q>
inline
PointGrid<R,d,q>::PointGrid() 
: _points(Pow<q,d>::val) 
{ }

template<typename R,size_t d,size_t q>
inline
PointGrid<R,d,q>::~PointGrid() 
{ }

template<typename R,size_t d,size_t q>
inline const array<R,d>&
PointGrid<R,d,q>::operator[] ( size_t i ) const
{ return _points[i]; }

template<typename R,size_t d,size_t q>
inline array<R,d>&
PointGrid<R,d,q>::operator[] ( size_t i )
{ return _points[i]; }

template<typename R,size_t d,size_t q>
inline const PointGrid<R,d,q>&
PointGrid<R,d,q>::operator= ( const PointGrid<R,d,q>& pointGrid )
{
    const size_t q_to_d = Pow<q,d>::val;
    memcpy( &(_points[0][0]), &(pointGrid[0][0]), q_to_d*sizeof(R) );
    return *this;
}

} // bfio

#endif // ifndef BFIO_STRUCTURES_POINT_GRID_HPP
