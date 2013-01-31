/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of ButterflyFIO and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef BFIO_FUNCTORS_PHASE_HPP
#define BFIO_FUNCTORS_PHASE_HPP

#include <cstddef>
#include <vector>

#include "bfio/structures/array.hpp"

namespace bfio {

// You will need to derive from this class and override the operator()
template<typename R,std::size_t d>
class Phase
{
public:
    virtual ~Phase();

    virtual Phase<R,d>* Clone() const = 0;

    // Point-wise evaluation of the phase function
    virtual R operator() 
    ( const Array<R,d>& x, const Array<R,d>& p ) const = 0;

    // ButterflyFIO calls BatchEvaluate whenever possible so that, if the user
    // supplies a class that overrides this method with an efficient vectorized
    // implementation, then performance should significantly increase.
    virtual void BatchEvaluate
    ( const std::vector< Array<R,d> >& x,
      const std::vector< Array<R,d> >& p,
            std::vector< R          >& results ) const;
};

// Implementations

template<typename R,std::size_t d>
inline
Phase<R,d>::~Phase() 
{ }

template<typename R,std::size_t d>
void 
Phase<R,d>::BatchEvaluate
( const std::vector< Array<R,d> >& x,
  const std::vector< Array<R,d> >& p,
        std::vector< R          >& results ) const
{
    results.resize( x.size()*p.size() );
    for( std::size_t i=0; i<x.size(); ++i )
        for( std::size_t j=0; j<p.size(); ++j )
            results[i*p.size()+j] = (*this)(x[i],p[j]);
}

} // bfio

#endif // ifndef BFIO_FUNCTORS_PHASE_HPP
