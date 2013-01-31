/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of ButterflyFIO and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef BFIO_FUNCTORS_AMPLITUDE_HPP
#define BFIO_FUNCTORS_AMPLITUDE_HPP

#include <cstddef>
#include <complex>
#include <vector>

#include "bfio/structures/array.hpp"

namespace bfio {

// You will need to derive from this class and override the operator()
template<typename R,std::size_t d>
class Amplitude
{
    // If a derived class sets this variable to 'true', then evaluation of the
    // amplitude function is circumvented.
    bool _isUnity; 
                   
public:
    Amplitude();
    Amplitude( bool isUnity );
    virtual ~Amplitude();

    virtual Amplitude<R,d>* Clone() const = 0;

    bool IsUnity() const;

    // Point-wise evaluation of the amplitude function
    virtual std::complex<R> operator() 
    ( const Array<R,d>& x, const Array<R,d>& p ) const = 0;

    // ButterflyFIO calls BatchEvaluate whenever possible so that, if the user
    // supplies a class that overrides this method with an efficient vectorized
    // implementation, then performance should significantly increase.
    virtual void BatchEvaluate
    ( const std::vector< Array<R,d>      >& x,
      const std::vector< Array<R,d>      >& p,
            std::vector< std::complex<R> >& results ) const;
};

// Extend the default class and explicitly call it a unit amplitude functor.
template<typename R,std::size_t d>
class UnitAmplitude : public Amplitude<R,d>
{ 
public:
    UnitAmplitude();

    virtual UnitAmplitude<R,d>* Clone() const;

    virtual std::complex<R> operator()
    ( const Array<R,d>& x, const Array<R,d>& p ) const;

    virtual void BatchEvaluate
    ( const std::vector< Array<R,d>      >& x,
      const std::vector< Array<R,d>      >& p,
            std::vector< std::complex<R> >& results ) const;
};

// Implementations

template<typename R,std::size_t d>
Amplitude<R,d>::Amplitude() 
: _isUnity(false) 
{ }

template<typename R,std::size_t d>
inline
Amplitude<R,d>::Amplitude( bool isUnity ) 
: _isUnity(isUnity) 
{ }

template<typename R,std::size_t d>
inline
Amplitude<R,d>::~Amplitude() 
{ }

template<typename R,std::size_t d>
inline bool 
Amplitude<R,d>::IsUnity() const 
{ return _isUnity; }

template<typename R,std::size_t d>
inline void 
Amplitude<R,d>::BatchEvaluate
( const std::vector< Array<R,d>      >& x,
  const std::vector< Array<R,d>      >& p,
        std::vector< std::complex<R> >& results ) const
{
    results.resize( x.size()*p.size() );
    for( std::size_t i=0; i<x.size(); ++i )
        for( std::size_t j=0; j<p.size(); ++j )
            results[i*p.size()+j] = (*this)(x[i],p[j]);
}

template<typename R,std::size_t d>
inline
UnitAmplitude<R,d>::UnitAmplitude() 
: Amplitude<R,d>(true) 
{ }

template<typename R,std::size_t d>
inline UnitAmplitude<R,d>* 
UnitAmplitude<R,d>::Clone() const
{ return new UnitAmplitude<R,d>(*this); }

template<typename R,std::size_t d>
inline std::complex<R>
UnitAmplitude<R,d>::operator()
( const Array<R,d>& x, const Array<R,d>& p ) const
{ return 1; }

template<typename R,std::size_t d>
inline void
UnitAmplitude<R,d>::BatchEvaluate
( const std::vector< Array<R,d>      >& x,
  const std::vector< Array<R,d>      >& p,
        std::vector< std::complex<R> >& results ) const
{
    results.resize( x.size()*p.size() );
    const std::size_t numEvals = results.size();
    for( std::size_t i=0; i<numEvals; ++i )
        results[i] = 1;
}

} // bfio

#endif // ifndef BFIO_FUNCTORS_AMPLITUDE_HPP
