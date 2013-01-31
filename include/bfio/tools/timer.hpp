/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of ButterflyFIO and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef BFIO_TOOLS_TIMER_HPP
#define BFIO_TOOLS_TIMER_HPP

#include <stdexcept>
#include "mpi.h"

namespace bfio {

class Timer
{
    bool _running;
    double _lastStartTime;
    double _totalTime;
    const std::string _name;
public:        
    Timer();
    Timer( const std::string& name );

    void Start();
    void Stop();
    void Reset();

    const std::string& Name() const;
    double TotalTime() const;
};

} // bfio

// Implementations
inline bfio::Timer::Timer()
: _running(false), _totalTime(0), _name("[blank]")
{ }

inline bfio::Timer::Timer( const std::string& name )
: _running(false), _totalTime(0), _name(name)
{ }

inline void 
bfio::Timer::Start()
{ 
#ifndef RELEASE
    if( _running )
	throw std::logic_error("Forgot to stop timer before restarting.");
#endif
    _lastStartTime = MPI_Wtime();
    _running = true;
}

inline void 
bfio::Timer::Stop()
{
#ifndef RELEASE
    if( !_running )
	throw std::logic_error("Tried to stop a timer before starting it.");
#endif
    _totalTime += MPI_Wtime()-_lastStartTime;
    _running = false;
}

inline void 
bfio::Timer::Reset()
{ _totalTime = 0; }

inline const std::string& 
bfio::Timer::Name() const
{ return _name; }

inline double 
bfio::Timer::TotalTime() const
{ 
#ifndef RELEASE
    if( _running )
	throw std::logic_error("Asked for total time while still timing.");
#endif
    return _totalTime; 
}

#endif // ifndef BFIO_TOOLS_TIMER_HPP
