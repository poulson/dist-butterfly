/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of DistButterfly and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef DBF_FUNCTORS_PHASE_HPP
#define DBF_FUNCTORS_PHASE_HPP

#include <array>
#include <cstddef>
#include <vector>

namespace dbf {

using std::array;
using std::size_t;
using std::vector;

// You will need to derive from this class and override the operator()
template<typename R,size_t d>
class Phase
{
public:
    virtual ~Phase();

    virtual Phase<R,d>* Clone() const = 0;

    // Point-wise evaluation of the phase function
    virtual R operator() 
    ( const array<R,d>& x, const array<R,d>& p ) const = 0;

    // DistButterfly calls BatchEvaluate whenever possible so that, if the user
    // supplies a class that overrides this method with an efficient vectorized
    // implementation, then performance should significantly increase.
    virtual void BatchEvaluate
    ( const vector<array<R,d>>& x,
      const vector<array<R,d>>& p,
            vector<R         >& results ) const;
};

// Implementations

template<typename R,size_t d>
inline
Phase<R,d>::~Phase() 
{ }

template<typename R,size_t d>
void 
Phase<R,d>::BatchEvaluate
( const vector<array<R,d>>& x,
  const vector<array<R,d>>& p,
        vector<R         >& results ) const
{
    results.resize( x.size()*p.size() );
    for( size_t i=0; i<x.size(); ++i )
        for( size_t j=0; j<p.size(); ++j )
            results[i*p.size()+j] = (*this)(x[i],p[j]);
}

} // dbf

#endif // ifndef DBF_FUNCTORS_PHASE_HPP
