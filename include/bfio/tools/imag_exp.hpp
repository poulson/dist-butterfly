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
#ifndef BFIO_TOOLS_IMAG_EXP_HPP
#define BFIO_TOOLS_IMAG_EXP_HPP 1

#include <math.h>

namespace bfio {

template<typename R>
inline std::complex<R>
ImagExp( R alpha );

template<>
inline std::complex<float>
ImagExp( float alpha )
{
#ifdef _GNU_SOURCE
    float sinAlpha, cosAlpha;
    sincosf( alpha, &sinAlpha, &cosAlpha );
    return std::complex<float>( cosAlpha, sinAlpha );
#else
    return std::complex<float>( cos(alpha), sin(alpha) );
#endif
}

template<>
inline std::complex<double>
ImagExp( double alpha )
{
#ifdef _GNU_SOURCE
    double sinAlpha, cosAlpha;
    sincos( alpha, &sinAlpha, &cosAlpha );
    return std::complex<double>( cosAlpha, sinAlpha );
#else
    return std::complex<double>( cos(alpha), sin(alpha) );
#endif
}

} // bfio

#endif // BFIO_TOOLS_IMAG_EXP_HPP

