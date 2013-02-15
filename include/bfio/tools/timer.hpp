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
    bool running_;
    double lastStartTime_;
    double totalTime_;
    const std::string name_;
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
: running_(false), totalTime_(0), name_("[blank]")
{ }

inline bfio::Timer::Timer( const std::string& name )
: running_(false), totalTime_(0), name_(name)
{ }

inline void 
bfio::Timer::Start()
{ 
#ifndef RELEASE
    if( running_ )
	throw std::logic_error("Forgot to stop timer before restarting.");
#endif
    lastStartTime_ = MPI_Wtime();
    running_ = true;
}

inline void 
bfio::Timer::Stop()
{
#ifndef RELEASE
    if( !running_ )
	throw std::logic_error("Tried to stop a timer before starting it.");
#endif
    totalTime_ += MPI_Wtime()-lastStartTime_;
    running_ = false;
}

inline void 
bfio::Timer::Reset()
{ totalTime_ = 0; }

inline const std::string& 
bfio::Timer::Name() const
{ return name_; }

inline double 
bfio::Timer::TotalTime() const
{ 
#ifndef RELEASE
    if( running_ )
	throw std::logic_error("Asked for total time while still timing.");
#endif
    return totalTime_; 
}

#endif // ifndef BFIO_TOOLS_TIMER_HPP
