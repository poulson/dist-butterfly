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
#ifndef BFIO_STRUCTURES_POINT_GRID_HPP
#define BFIO_STRUCTURES_POINT_GRID_HPP 1

#include "bfio/constants.hpp"
#include "bfio/structures/array.hpp"

namespace bfio {

template<typename R,std::size_t d,std::size_t q>
class PointGrid
{
    // We know the size should be q^d at compile time, but we do not want the
    // data stored on the stack
    std::vector< Array<R,d> > _points;

public:
    PointGrid();
    ~PointGrid();

    const Array<R,d>&
    operator[] ( std::size_t i ) const;

    Array<R,d>&
    operator[] ( std::size_t i );

    const PointGrid<R,d,q>&
    operator= ( const PointGrid<R,d,q>& pointGrid );
};

// Implementations

template<typename R,std::size_t d,std::size_t q>
inline
PointGrid<R,d,q>::PointGrid() 
: _points(Pow<q,d>::val) 
{ }

template<typename R,std::size_t d,std::size_t q>
inline
PointGrid<R,d,q>::~PointGrid() 
{ }

template<typename R,std::size_t d,std::size_t q>
inline const Array<R,d>&
PointGrid<R,d,q>::operator[] ( std::size_t i ) const
{ return _points[i]; }

template<typename R,std::size_t d,std::size_t q>
inline Array<R,d>&
PointGrid<R,d,q>::operator[] ( std::size_t i )
{ return _points[i]; }

template<typename R,std::size_t d,std::size_t q>
inline const PointGrid<R,d,q>&
PointGrid<R,d,q>::operator= ( const PointGrid<R,d,q>& pointGrid )
{
    const std::size_t q_to_d = Pow<q,d>::val;
    std::memcpy( &(_points[0][0]), &(pointGrid[0][0]), q_to_d*sizeof(R) );
    return *this;
}

} // bfio

#endif // BFIO_STRUCTURES_POINT_GRID_HPP

