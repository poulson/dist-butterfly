/*
   Copyright (c) 2010, Jack Poulson
   All rights reserved.

   This file is part of ButterflyFIO.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are met:

    - Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    - Neither the name of the owner nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
   ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
   LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
   CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
   SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
   INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
   CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
   ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
   POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once
#ifndef BFIO_TWIDDLE_HPP
#define BFIO_TWIDDLE_HPP 1

namespace bfio {

inline bool
IsPowerOfTwo( unsigned N )
{ return N && !(N & (N-1)); }

// This is a slight modification of Sean Eron Anderson's 
// 'Find the log2 base 2 of an N-bit integer in O(lg(N)) operations 
//  with multiply and lookup'. It was found at
//    http://graphics.stanford.edu/~seander/bithacks.html
// and is in the public domain.
inline unsigned
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
    return MultiplyDeBruijnBitPosition[(unsigned)(N*0x07C4ACDDU)>>27];
}

// This is a modification of Sean Eron Anderson's binary search algorithm
// for counting the trailing zeros of a 32-bit integer. It was found at
//     http://graphics.stanford.edu/~seander/bithacks.html
// and is in the public domain.
//
// The main difference is that I switched the algorithm to count ones and 
// ignore the case where N=2^32-1 rather than N=0
inline unsigned
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
    return c;
}

} // bfio

#endif /* BFIO_TWIDDLE_HPP */

