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
#include <ctime>
#include <memory>
#include "bfio.hpp"
using namespace std;
using namespace bfio;

void 
Usage()
{
    cout << "Random3DWaves <N> <M> <T> <nT>" << endl;
    cout << "  N: power of 2, the frequency spread in each dimension" << endl;
    cout << "  M: number of random sources to instantiate" << endl;
    cout << "  T: time to simulate to" << endl;
    cout << "  nT: number of timesteps" << endl;
    cout << endl;
}

static const unsigned d = 3;
static const unsigned q = 5;

class Unity : public AmplitudeFunctor<double,d>
{
public:
    complex<double>
    operator() ( const Array<double,d>& x, const Array<double,d>& p ) const
    { return complex<double>(1); }
};
 
class UpWave : public PhaseFunctor<double,d>
{
    double _t;
public:
    UpWave() : _t(0) {}

    void SetTime( const double t ) { _t = t; }
    double GetTime() const { return _t; }

    double
    operator() ( const Array<double,d>& x, const Array<double,d>& p ) const
    { 
        return x[0]*p[0]+x[1]*p[1]+x[2]*p[2] + 
               _t * sqrt(p[0]*p[0]+p[1]*p[1]+p[2]*p[2]);
    }
};

class DownWave : public PhaseFunctor<double,d>
{
    double _t;
public:
    DownWave() : _t(0) {}

    void SetTime( const double t ) { _t = t; }
    double GetTime() const { return _t; }

    double
    operator() ( const Array<double,d>& x, const Array<double,d>& p ) const
    {
        return x[0]*p[0]+x[1]*p[1]+x[2]*p[2] - 
                _t * sqrt(p[0]*p[0]+p[1]*p[1]+p[2]*p[2]);
    }
};

int
main
( int argc, char* argv[] )
{
    int rank, size;
    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );

    if( !IsPowerOfTwo(size) )
    {
        if( rank == 0 )
            cout << "Must run with a power of two number of cores." << endl;
        MPI_Finalize();
        return 0;
    }

    if( argc != 5 )
    {
        if( rank == 0 )
            Usage();
        MPI_Finalize();
        return 0;
    }
    const unsigned N = atoi(argv[1]);
    const unsigned M = atoi(argv[2]);
    const double   T = atof(argv[3]);
    const unsigned nT = atoi(argv[4]);

    if( rank == 0 )
    {
        ostringstream msg;
        msg << "Will distribute " << M << " random sources over the frequency "
            << "domain, which will be split into " << N
            << " boxes in each of the " << d << " dimensions and distributed "
            << "amongst " << size << " processes. The simulation will be over "
            << T << " units of time with " << nT << " timesteps." 
            << endl << endl;
        cout << msg.str();
    }

    try 
    {
        // Compute the box that our process owns
        Array<double,d> myFreqBoxWidths;
        Array<double,d> myFreqBoxOffsets;
        LocalFreqPartitionData
        ( myFreqBoxWidths, myFreqBoxOffsets, MPI_COMM_WORLD );

        // Seed our process
        long seed = time(0);
        srand( seed );

        // Now generate random sources in our frequency box
        unsigned numLocalSources = ( rank<(int)(M%size) ? M/size+1 : M/size );
        vector< Source<double,d> > mySources( numLocalSources );
        for( unsigned i=0; i<numLocalSources; ++i )
        {
            for( unsigned j=0; j<d; ++j )
            {
                mySources[i].p[j] = 
                    myFreqBoxOffsets[j]+Uniform<double>()*myFreqBoxWidths[j];
            }
            mySources[i].magnitude = 200*Uniform<double>()-100;
        }

        // Set up our amplitude and phase functors
        Unity unity;
        UpWave upWave;
        DownWave downWave;

        // Loop over each timestep, computing in parallel, gathering the 
        // results, and then dumping to file
        double deltaT = T/(nT-1);
        unsigned numLocalLRPs = NumLocalBoxes<d>( N, MPI_COMM_WORLD );
        for( unsigned i=0; i<nT; ++i )
        {
            const double t = i*deltaT;
            upWave.SetTime( t );
            downWave.SetTime( t );

            if( rank == 0 )
            {
                cout << "t=" << t << endl;
                cout << "  Starting upWave transform...";
                cout.flush();
            }
            vector< LowRankPotential<double,d,q> > myUpWaveLRPs
            ( numLocalLRPs, LowRankPotential<double,d,q>(unity,upWave,N) );
            FreqToSpatial
            ( mySources, myUpWaveLRPs, MPI_COMM_WORLD );

            if( rank == 0 )
            {
                cout << "done" << endl;
                cout << "  Starting downWave transform...";
            }
            vector< LowRankPotential<double,d,q> > myDownWaveLRPs
            ( numLocalLRPs, LowRankPotential<double,d,q>(unity,downWave,N) );
            FreqToSpatial
            ( mySources, myDownWaveLRPs, MPI_COMM_WORLD );
            if( rank == 0 )
            {
                cout << "done" << endl;
            }

            // TODO: Gather potentials and then dump to VTK file 
        }
    }
    catch( const exception& e )
    {
        ostringstream msg;
        msg << "Caught exception on process " << rank << ":" << endl;
        msg << "   " << e.what() << endl;
        cout << msg.str();
    }

    MPI_Finalize();
    return 0;
}

