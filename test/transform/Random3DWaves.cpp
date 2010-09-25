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

// Define the dimension of the problem and the order of interpolation
static const size_t d = 3;
static const size_t q = 5;

template<typename R>
class Unity : public AmplitudeFunctor<R,d>
{
public:
    complex<R>
    operator() ( const tr1::array<R,d>& x, const tr1::array<R,d>& p ) const
    { return complex<R>(1); }
};
 
template<typename R>
class UpWave : public PhaseFunctor<R,d>
{
    R _t;
public:
    UpWave() : _t(0) {}

    void SetTime( const R t ) { _t = t; }
    R GetTime() const { return _t; }

    R
    operator() ( const tr1::array<R,d>& x, const tr1::array<R,d>& p ) const
    { 
        return x[0]*p[0]+x[1]*p[1]+x[2]*p[2] + 
               _t * sqrt(p[0]*p[0]+p[1]*p[1]+p[2]*p[2]);
    }
};

template<typename R>
class DownWave : public PhaseFunctor<R,d>
{
    R _t;
public:
    DownWave() : _t(0) {}

    void SetTime( const R t ) { _t = t; }
    R GetTime() const { return _t; }

    R
    operator() ( const tr1::array<R,d>& x, const tr1::array<R,d>& p ) const
    {
        return x[0]*p[0]+x[1]*p[1]+x[2]*p[2] - 
                _t * sqrt(p[0]*p[0]+p[1]*p[1]+p[2]*p[2]);
    }
};

int
main
( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );

    int rank, numProcesses;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank( comm, &rank );
    MPI_Comm_size( comm, &numProcesses );

    if( !IsPowerOfTwo(numProcesses) )
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
    const size_t N = atoi(argv[1]);
    const size_t M = atoi(argv[2]);
    const double   T = atof(argv[3]);
    const size_t nT = atoi(argv[4]);

    // Define the spatial and frequency boxes
    Box<double,d> freqBox, spatialBox;
    for( size_t j=0; j<d; ++j )
    {
        freqBox.offsets[j] = -0.5*N;
        freqBox.widths[j] = N;
        spatialBox.offsets[j] = 0;
        spatialBox.widths[j] = 1;
    }

    if( rank == 0 )
    {
        ostringstream msg;
        msg << "Will distribute " << M << " random sources over the frequency "
            << "domain, which will be split into " << N
            << " boxes in each of the " << d << " dimensions and distributed "
            << "amongst " << numProcesses << " processes. The simulation will "
            << "be over " << T << " units of time with " << nT << " timesteps." 
            << endl << endl;
        cout << msg.str();
    }

    try 
    {
        // Compute the box that our process owns
        Box<double,d> myFreqBox;
        LocalFreqPartitionData( freqBox, myFreqBox, comm );

        // Seed our process
        long seed = time(0);
        srand( seed );

        // Now generate random sources in our frequency box
        size_t numLocalSources = ( rank<(int)(M%numProcesses) 
                                     ? M/numProcesses+1 : M/numProcesses );
        vector< Source<double,d> > mySources( numLocalSources );
        for( size_t i=0; i<numLocalSources; ++i )
        {
            for( size_t j=0; j<d; ++j )
            {
                mySources[i].p[j] = 
                    myFreqBox.offsets[j]+Uniform<double>()*myFreqBox.widths[j];
            }
            mySources[i].magnitude = 200*Uniform<double>()-100;
        }

        // Set up our amplitude and phase functors
        Unity<double> unity;
        UpWave<double> upWave;
        DownWave<double> downWave;

        // Create a context, which includes all of the precomputation
        if( rank == 0 )
            cout << "Creating context..." << endl;
        Context<double,d,q> context;

        // Loop over each timestep, computing in parallel, gathering the 
        // results, and then dumping to file
        double deltaT = T/(nT-1);
        for( size_t i=0; i<nT; ++i )
        {
            const double t = i*deltaT;
            upWave.SetTime( t );
            downWave.SetTime( t );

            auto_ptr< const PotentialField<double,d,q> > u;
            if( rank == 0 )
            {
                cout << "t=" << t << endl;
                cout << "  Starting upWave transform...";
                cout.flush();
            }
            u = FreqToSpatial
            ( N, freqBox, spatialBox, unity, upWave, context, mySources, comm );

            auto_ptr< const PotentialField<double,d,q> > v;
            if( rank == 0 )
            {
                cout << "done" << endl;
                cout << "  Starting downWave transform...";
            }
            v = FreqToSpatial
            ( N, freqBox, spatialBox, unity, downWave, context, mySources, 
              comm );
            if( rank == 0 )
                cout << "done" << endl;

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

