/*
   ButterflyFIO: a distributed-memory fast algorithm for applying FIOs.
   Copyright (C) 2010-2011 Jack Poulson <jack.poulson@gmail.com>
 
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
#ifndef BFIO_STRUCTURES_LOW_RANK_POTENTIAL_HPP
#define BFIO_STRUCTURES_LOW_RANK_POTENTIAL_HPP 1

#include "bfio/structures/array.hpp"
#include "bfio/structures/weight_grid.hpp"

namespace bfio {

template<typename R,std::size_t d,std::size_t q>
struct LRP
{
    Array<R,d> x0;
    WeightGrid<R,d,q> weightGrid;
};

} // bfio

#endif // BFIO_STRUCTURES_LOW_RANK_POTENTIAL_HPP

