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
    int rank, numProcesses;
    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &numProcesses );

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
    const unsigned N = atoi(argv[1]);
    const unsigned M = atoi(argv[2]);
    const double   T = atof(argv[3]);
    const unsigned nT = atoi(argv[4]);

    // Define the spatial and frequency boxes
    Box<double,d> freqBox, spatialBox;
    for( unsigned j=0; j<d; ++j )
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
        LocalFreqPartitionData
        ( freqBox, myFreqBox, MPI_COMM_WORLD );

        // Seed our process
        long seed = time(0);
        srand( seed );

        // Now generate random sources in our frequency box
        unsigned numLocalSources = ( rank<(int)(M%numProcesses) 
                                     ? M/numProcesses+1 : M/numProcesses );
        vector< Source<double,d> > mySources( numLocalSources );
        for( unsigned i=0; i<numLocalSources; ++i )
        {
            for( unsigned j=0; j<d; ++j )
            {
                mySources[i].p[j] = 
                    myFreqBox.offsets[j]+Uniform<double>()*myFreqBox.widths[j];
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
            ( numLocalLRPs, LowRankPotential<double,d,q>(unity,upWave) );
            FreqToSpatial
            ( N, freqBox, spatialBox, mySources, myUpWaveLRPs, MPI_COMM_WORLD );

            if( rank == 0 )
            {
                cout << "done" << endl;
                cout << "  Starting downWave transform...";
            }
            vector< LowRankPotential<double,d,q> > myDownWaveLRPs
            ( numLocalLRPs, LowRankPotential<double,d,q>(unity,downWave) );
            FreqToSpatial
            ( N, freqBox, spatialBox, mySources, myDownWaveLRPs, 
              MPI_COMM_WORLD );
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

