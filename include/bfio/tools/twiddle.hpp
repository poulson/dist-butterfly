/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of ButterflyFIO and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef BFIO_TOOLS_TWIDDLE_HPP
#define BFIO_TOOLS_TWIDDLE_HPP

namespace bfio {

inline bool
IsPowerOfTwo( std::size_t N )
{ return N && !(N & (N-1)); }

// This is a slight modification of Sean Eron Anderson's 
// 'Find the log2 base 2 of an N-bit integer in O(lg(N)) operations 
//  with multiply and lookup'. It was found at
//    http://graphics.stanford.edu/~seander/bithacks.html
// and is in the public domain.
//
// Note: the rest of ButterflyFIO is now written in terms of std::size_t rather
//       than unsigned in order to be more compatible with the STL. However,
//       it _extremely_ unlikely that the problem size will be larger than the 
//       range of an unsigned.
std::size_t
Log2( unsigned N )
{
    static const unsigned MultiplyDeBruijnBitPosition[32] = 
    {
      0, 9, 1, 10, 13, 21, 2, 29, 11, 14, 16, 18, 22, 25, 3, 30,
      8, 12, 20, 28, 15, 17, 24, 7, 19, 27, 23, 6, 26, 5, 4, 31
    };

    N |= N >> 1;
    N |= N >> 2;
    N |= N >> 4;
    N |= N >> 8;
    N |= N >> 16;
    N = MultiplyDeBruijnBitPosition[(unsigned)(N*0x07C4ACDDU)>>27];
    return static_cast<std::size_t>(N);
}

// This is a modification of Sean Eron Anderson's binary search algorithm
// for counting the trailing zeros of a 32-bit integer. It was found at
//     http://graphics.stanford.edu/~seander/bithacks.html
// and is in the public domain.
//
// The main difference is that I switched the algorithm to count ones and 
// ignore the case where N=2^32-1 rather than N=0
//
// Note: the rest of ButterflyFIO is now written in terms of std::size_t rather
//       than unsigned in order to be more compatible with the STL. However,
//       it _extremely_ unlikely that the problem size will be larger than the 
//       range of an unsigned.
std::size_t
NumberOfTrailingOnes( unsigned N )
{
    unsigned int c;
    if( (N & 0x1)==0 )
    {
        c = 0;
    }
    else
    {
        c = 1; 
        if( (N & 0xffff)==0xffff )
        {
            N >>= 16; 
            c += 16;
        }
        if( (N & 0xff) == 0xff )
        {
            N >>= 8;
            c += 8;
        }
        if( (N & 0xf) == 0xf )
        {
            N >>= 4;
            c += 4;
        }
        if( (N & 0x3) == 0x3 )
        {
            N >>= 2;
            c += 2;
        }
        c -= !(N & 0x1);
    }
    return static_cast<std::size_t>(c);
}

} // bfio

#endif // ifndef BFIO_TOOLS_TWIDDLE_HPP
