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
#ifndef BFIO_LAGRANGE_HPP
#define BFIO_LAGRANGE_HPP 1

#include <vector>
#include "bfio/structures/data.hpp"

namespace bfio {

template<typename R,unsigned d,unsigned q>
R
Lagrange
( const unsigned t, const Array<R,d>& z )
{
    const unsigned q_to_d = Pow<q,d>::val;

    static bool initialized = false;
    static Array<R,q> chebyNodes;
    static std::vector< Array<unsigned,d> > chebyIndex( q_to_d );

    if( !initialized )
    {
        for( unsigned i=0; i<q; ++i )
            chebyNodes[i] = 0.5*cos(i*Pi/(q-1));
        for( unsigned tp=0; tp<q_to_d; ++tp )
        {
            unsigned qToThej = 1;
            for( unsigned j=0; j<d; ++j )
            {
                unsigned i = (tp/qToThej) % q;
                chebyIndex[tp][j] = i;
                qToThej *= q;
            }
        }
        initialized = true;
    }

    R product = static_cast<R>(1);
    for( unsigned j=0; j<d; ++j )
    {
        unsigned i = chebyIndex[t][j];
        for( unsigned k=0; k<q; ++k )
        {
            if( i != k )
            {
                product *= 
                    (z[j]-chebyNodes[k]) / (chebyNodes[i]-chebyNodes[k]);
            }
        }
    }
    return product;
}

} // bfio

#endif /* BFIO_LAGRANGE_HPP */

