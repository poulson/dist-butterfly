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
#ifndef BFIO_TOOLS_IMAG_EXP_HPP
#define BFIO_TOOLS_IMAG_EXP_HPP 1

#include <math.h>
#include <vector>

namespace bfio {

// For single-point imaginary exponentials
template<typename R>
inline std::complex<R>
ImagExp( R alpha );

template<>
inline std::complex<float>
ImagExp( float alpha )
{
    // TODO: Add fast sincos support for various architectures
    return std::complex<float>( cos(alpha), sin(alpha) );
}

template<>
inline std::complex<double>
ImagExp( double alpha )
{
    // TODO: Add fast sincos support for various architectures
    return std::complex<double>( cos(alpha), sin(alpha) );
}

// For vector imaginary exponentials
template<typename R>
inline void
ImagExpBatch
( const std::vector<R>& alpha, std::vector< std::complex<R> >& results );

template<>
inline void
ImagExpBatch
( const std::vector< float               >& alpha, 
        std::vector< std::complex<float> >& results )
{
    results.resize( alpha.size() );
    // TODO: Add vectorization support here for various architectures
    for( unsigned j=0; j<alpha.size(); ++j )
    {
        results[j] = std::complex<float>( cos(alpha[j]), sin(alpha[j]) );
    }
}

template<>
inline void
ImagExpBatch
( const std::vector< double               >& alpha, 
        std::vector< std::complex<double> >& results )
{
    results.resize( alpha.size() );
    // TODO: Add vectorization support here for various architectures
    for( unsigned j=0; j<alpha.size(); ++j )
    {
        results[j] = std::complex<double>( cos(alpha[j]), sin(alpha[j]) );
    }
}

} // bfio

#endif // BFIO_TOOLS_IMAG_EXP_HPP

