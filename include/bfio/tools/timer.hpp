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
#ifndef BFIO_TOOLS_TIMER_HPP
#define BFIO_TOOLS_TIMER_HPP 1

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
    const double TotalTime() const;
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

inline const double 
bfio::Timer::TotalTime() const
{ 
#ifndef RELEASE
    if( _running )
	throw std::logic_error("Asked for total time while still timing.");
#endif
    return _totalTime; 
}

#endif // BFIO_TOOLS_TIMER_HPP

