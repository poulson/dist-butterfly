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
    cout << "VariableUpWave <N> <M> <Amp. Alg.> <testAccuracy?>" << endl;
    cout << "  N: power of 2, the frequency spread in each dimension" << endl;
    cout << "  M: number of random sources to instantiate" << endl;
    cout << "  Amp. Alg.: 0 for MiddleSwitch, 1 for Prefactor" << endl;
    cout << "  testAccuracy?: test accuracy iff 1" << endl;
    cout << endl;
}

static const unsigned d = 2;
static const unsigned q = 12;

class Oscillatory : public AmplitudeFunctor<double,d>
{
public:
    Oscillatory( AmplitudeAlgorithm alg )
    : AmplitudeFunctor<double,d>(alg) 
    { }

    complex<double>
    operator() ( const Array<double,d>& x, const Array<double,d>& p ) const
    { 
        return 1. + 
               0.5*sin(Pi*x[0])*sin(4*Pi*x[1])*sin(3*Pi*p[0])*cos(4*Pi*p[1]);
    }
};

class UpWave : public PhaseFunctor<double,d>
{
public:
    double
    operator() ( const Array<double,d>& x, const Array<double,d>& p ) const
    {
        return x[0]*p[0]+x[1]*p[1]+0.5*sqrt(p[0]*p[0]+p[1]*p[1]);
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
    const AmplitudeAlgorithm algorithm = 
        ( atoi(argv[3]) ? Prefactor : MiddleSwitch );
    const bool testAccuracy = atoi(argv[4]);

    // Set the frequency and spatial boxes
    Box<double,d> freqBox, spatialBox;
    for( unsigned j=0; j<d; ++j )
    {
        freqBox.offsets[j] = -0.5*N/8.;
        freqBox.widths[j] = N/8.;
        spatialBox.offsets[j] = 0;
        spatialBox.widths[j] = 1;
    }

    if( rank == 0 )
    {
        ostringstream msg;
        msg << "Will distribute " << M << " random sources over the frequency "
            << "domain, which will be split into " << N
            << " boxes in each of the " << d << " dimensions and distributed "
            << "amongst " << numProcesses << " processes." << endl << endl;
        msg << "We will use the "
            << ( algorithm==Prefactor ? "Prefactor" : "MiddleSwitch" )
            << " amplitude algorithm." << endl << endl; 
        cout << msg.str();
    }

    try 
    {
        // Consistently randomly seed all of the processes' PRNG.
        long seed;
        if( rank == 0 )
            seed = time(0);
        MPI_Bcast( &seed, 1, MPI_LONG, 0, MPI_COMM_WORLD );
        srand( seed );

        // Compute the box that our process owns
        Box<double,d> myFreqBox;
        LocalFreqPartitionData
        ( freqBox, myFreqBox, MPI_COMM_WORLD );

        // Now generate random sources across the domain and store them in 
        // our local list when appropriate
        vector< Source<double,d> > mySources;
        vector< Source<double,d> > globalSources;
        if( testAccuracy )
        {
            globalSources.resize( M );
            for( unsigned i=0; i<M; ++i )
            {
                for( unsigned j=0; j<d; ++j )
                    globalSources[i].p[j] = freqBox.offsets[j] + 
                        freqBox.widths[j]*Uniform<double>(); 
                globalSources[i].magnitude = 200*Uniform<double>()-100; 

                // Check if we should push this source onto our local list
                bool isMine = true;
                for( unsigned j=0; j<d; ++j )
                {
                    double u = globalSources[i].p[j];
                    double start = myFreqBox.offsets[j];
                    double stop = myFreqBox.offsets[j] + myFreqBox.widths[j];
                    if( u < start || u >= stop )
                        isMine = false;
                }
                if( isMine )
                    mySources.push_back( globalSources[i] );
            }
        }
        else
        {
            unsigned numLocalSources = 
                ( rank<(int)(M%numProcesses) 
                  ? M/numProcesses+1 : M/numProcesses );
            mySources.resize( numLocalSources ); 
            for( unsigned i=0; i<numLocalSources; ++i )
            {
                for( unsigned j=0; j<d; ++j )
                {
                    mySources[i].p[j] = myFreqBox.offsets[j]+
                                        Uniform<double>()*myFreqBox.widths[j];
                }
                mySources[i].magnitude = 200*Uniform<double>()-100;
            }
        }

        // Set up our amplitude and phase functors
        Oscillatory oscillatory(algorithm);
        UpWave upWave;

        // Create a vector for storing the results
        unsigned numLocalLRPs = NumLocalBoxes<d>( N, MPI_COMM_WORLD );
        vector< LowRankPotential<double,d,q> > myUpWaveLRPs
        ( numLocalLRPs, LowRankPotential<double,d,q>(oscillatory,upWave) );

        // Run the algorithm
        MPI_Barrier( MPI_COMM_WORLD );
        double startTime = MPI_Wtime();
        FreqToSpatial
        ( N, freqBox, spatialBox, mySources, myUpWaveLRPs, MPI_COMM_WORLD );
        MPI_Barrier( MPI_COMM_WORLD );
        double stopTime = MPI_Wtime();
        if( rank == 0 )
            cout << "Runtime: " << stopTime-startTime << " seconds." << endl;

        if( testAccuracy )
        {
            double myMaxRelError = 0.;
            for( unsigned k=0; k<myUpWaveLRPs.size(); ++k )
            {
                // Retrieve the spatial center of LRP k
                Array<double,d> x0 = myUpWaveLRPs[k].GetSpatialCenter();

                // Find a random point in that box
                Array<double,d> x;
                for( unsigned j=0; j<d; ++j )
                    x[j] = x0[j] + 
                           spatialBox.widths[j]/(2*N)*(2*Uniform<double>()-1.);

                // Evaluate our LRP at x and compare against truth
                complex<double> u = myUpWaveLRPs[k]( x );
                complex<double> uTruth(0.,0.);
                for( unsigned m=0; m<globalSources.size(); ++m )
                {
                    Array<double,d>& p = globalSources[m].p;
                    double alpha = TwoPi * upWave(x,p);
                    uTruth += oscillatory(x,p)*
                              complex<double>(cos(alpha),sin(alpha))*
                              globalSources[m].magnitude;
                }
                double relError = abs(u-uTruth)/max(abs(uTruth),1.);
                myMaxRelError = max( myMaxRelError, relError );
            }
            double maxRelError;
            MPI_Reduce
            ( &myMaxRelError, &maxRelError, 1, MPI_DOUBLE, MPI_MAX, 0, 
              MPI_COMM_WORLD );
            if( rank == 0 )
                cout << "  maxRelError: " << maxRelError << endl;
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

