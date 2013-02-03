/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of ButterflyFIO and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef BFIO_HPP
#define BFIO_HPP

// Include the configuration-specific preprocessor definitions
#include "bfio/config.h"

// One could probably speed up compile time by including everything in a 
// dependency-aware order
#include "bfio/constants.hpp"
#include "bfio/structures.hpp"
#include "bfio/tools.hpp"
#include "bfio/functors.hpp"

#include "bfio/rfio.hpp"
#include "bfio/lnuft.hpp"
#include "bfio/inuft.hpp"

#endif // ifndef BFIO_HPP 
