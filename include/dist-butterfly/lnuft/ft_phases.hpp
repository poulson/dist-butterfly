/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of DistButterfly and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef DBF_LNUFT_FT_PHASES_HPP
#define DBF_LNUFT_FT_PHASES_HPP

#include "dist-butterfly/functors/phase.hpp"

namespace dbf {

using std::array;
using std::memset;
using std::size_t;
using std::vector;

namespace lnuft {

// Perform a trivial extension of Phase so that we may explicitly write the 
// lnuft PotentialField in terms of Fourier phase functions
template<typename R,size_t d>
class FTPhase : public Phase<R,d>
{ };

template<typename R,size_t d>
class ForwardFTPhase : public FTPhase<R,d>
{
public:
    // Default constructors are required to initialize const classes.
    ForwardFTPhase();

    virtual ForwardFTPhase<R,d>* Clone() const;

    virtual R
    operator()( const array<R,d>& x, const array<R,d>& p ) const;

    // We can optionally override the batched application for better efficiency
    virtual void
    BatchEvaluate
    ( const vector<array<R,d>>& xPoints,
      const vector<array<R,d>>& pPoints,
            vector<R         >& results ) const;
};

template<typename R,size_t d>
class AdjointFTPhase : public FTPhase<R,d>
{
public:
    // Default constructors are required to initialize const classes.
    AdjointFTPhase();

    virtual AdjointFTPhase<R,d>* Clone() const;

    virtual R
    operator()( const array<R,d>& x, const array<R,d>& p ) const;

    // We can optionally override the batched application for better efficiency
    virtual void
    BatchEvaluate
    ( const vector<array<R,d>>& xPoints,
      const vector<array<R,d>>& pPoints,
            vector<R         >& results ) const;
};

// Implementations

template<typename R,size_t d>
inline
ForwardFTPhase<R,d>::ForwardFTPhase()
{ }

template<typename R,size_t d>
inline
AdjointFTPhase<R,d>::AdjointFTPhase()
{ }

template<typename R,size_t d>
inline ForwardFTPhase<R,d>*
ForwardFTPhase<R,d>::Clone() const
{ return new ForwardFTPhase<R,d>(*this); }

template<typename R,size_t d>
inline AdjointFTPhase<R,d>*
AdjointFTPhase<R,d>::Clone() const
{ return new AdjointFTPhase<R,d>(*this); }

template<typename R,size_t d>
inline R
ForwardFTPhase<R,d>::operator()
( const array<R,d>& x, const array<R,d>& p ) const
{
    R sum = 0;
    const R* RESTRICT xBuffer = &x[0];
    const R* RESTRICT pBuffer = &p[0];
    for( size_t j=0; j<d; ++j )
        sum += xBuffer[j]*pBuffer[j];
    return -TwoPi<R>()*sum;
}

template<typename R,size_t d>
inline R
AdjointFTPhase<R,d>::operator()
( const array<R,d>& x, const array<R,d>& p ) const
{
    R sum = 0;
    const R* RESTRICT xBuffer = &x[0];
    const R* RESTRICT pBuffer = &p[0];
    for( size_t j=0; j<d; ++j )
        sum += xBuffer[j]*pBuffer[j];
    return TwoPi<R>()*sum;
}

template<typename R,size_t d>
inline void
ForwardFTPhase<R,d>::BatchEvaluate
( const vector<array<R,d>>& xPoints,
  const vector<array<R,d>>& pPoints,
        vector<R         >& results ) const
{
    const size_t nxPoints = xPoints.size();
    const size_t npPoints = pPoints.size();
    results.resize( nxPoints*npPoints );

    const R twoPi = TwoPi<R>();
    R* RESTRICT resultsBuffer = &results[0];
    memset( resultsBuffer, 0, nxPoints*npPoints*sizeof(R) );
    const R* RESTRICT xPointsBuffer = &xPoints[0][0];
    const R* RESTRICT pPointsBuffer = &pPoints[0][0];
    for( size_t i=0; i<nxPoints; ++i )
    {
        for( size_t j=0; j<npPoints; ++j )
        {
            for( size_t k=0; k<d; ++k )
            {
                resultsBuffer[i*npPoints+j] += 
                    xPointsBuffer[i*d+k]*pPointsBuffer[j*d+k];
            }
            resultsBuffer[i*npPoints+j] *= -twoPi;
        }
    }
}

template<typename R,size_t d>
inline void
AdjointFTPhase<R,d>::BatchEvaluate
( const vector<array<R,d>>& xPoints,
  const vector<array<R,d>>& pPoints,
        vector<R         >& results ) const
{
    const size_t nxPoints = xPoints.size();
    const size_t npPoints = pPoints.size();
    results.resize( nxPoints*npPoints );

    const R twoPi = TwoPi<R>();
    R* RESTRICT resultsBuffer = &results[0];
    memset( resultsBuffer, 0, nxPoints*npPoints*sizeof(R) );
    const R* RESTRICT xPointsBuffer = &xPoints[0][0];
    const R* RESTRICT pPointsBuffer = &pPoints[0][0];
    for( size_t i=0; i<nxPoints; ++i )
    {
        for( size_t j=0; j<npPoints; ++j )
        {
            for( size_t k=0; k<d; ++k )
            {
                resultsBuffer[i*npPoints+j] += 
                    xPointsBuffer[i*d+k]*pPointsBuffer[j*d+k];
            }
            resultsBuffer[i*npPoints+j] *= twoPi;
        }
    }
}

} // lnuft
} // dbf

#endif // ifndef DBF_LNUFT_FT_PHASES_HPP
