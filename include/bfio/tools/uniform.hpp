/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of ButterflyFIO and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef BFIO_TOOLS_UNIFORM_HPP
#define BFIO_TOOLS_UNIFORM_HPP

#include <cstdlib>

namespace bfio {

// Samples uniformly within (0,1]
template<typename R>
R
Uniform()
{ return (R(rand())+1)/RAND_MAX; }

} // bfio

#endif // ifndef BFIO_TOOLS_UNIFORM_HPP
