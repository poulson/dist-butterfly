/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of ButterflyFIO and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef BFIO_STRUCTURES_LOW_RANK_POTENTIAL_HPP
#define BFIO_STRUCTURES_LOW_RANK_POTENTIAL_HPP

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

#endif // ifndef BFIO_STRUCTURES_LOW_RANK_POTENTIAL_HPP
