/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of DistButterfly and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef DBF_STRUCTURES_LOW_RANK_POTENTIAL_HPP
#define DBF_STRUCTURES_LOW_RANK_POTENTIAL_HPP

#include <array>

#include "dist-butterfly/structures/weight_grid.hpp"

namespace dbf {

template<typename R,std::size_t d,std::size_t q>
struct LRP
{
    std::array<R,d> x0;
    WeightGrid<R,d,q> weightGrid;
};

} // dbf

#endif // ifndef DBF_STRUCTURES_LOW_RANK_POTENTIAL_HPP
