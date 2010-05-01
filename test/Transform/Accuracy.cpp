/*
  Copyright 2010 Jack Poulson

  This file is part of ButterflyFIO.

  This program is free software: you can redistribute it and/or modify it under
  the terms of the GNU Lesser General Public License as published by the
  Free Software Foundation; either version 3 of the License, or 
  (at your option) any later version.

  This program is distributed in the hope that it will be useful, but 
  WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
*/
#include <ctime>
#include <memory>
#include "BFIO.hpp"
using namespace std;
using namespace BFIO;

void 
Usage()
{
    cout << "Accuracy <N> <M>" << endl;
    cout << "  N: power of 2, the frequency spread in each dimension" << endl;
    cout << "  M: number of random sources to instantiate" << endl;
    cout << endl;
}

static const unsigned d = 3;
static const unsigned q = 5;

class UpWave : public PhaseFunctor<double,d>
{
public:
    inline double
    operator() ( const Array<double,d>& x, const Array<double,d>& p ) const
    {
        return x[0]*p[0]+x[1]*p[1]+x[2]*p[2] + 
               0.5*sqrt(p[0]*p[0]+p[1]*p[1]+p[2]*p[2]); 
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
        cout << "Must run with a power of two number of cores." << endl;
        return 0;
    }

    if( argc != 3 )
    {
        if( rank == 0 )
            Usage();
        MPI_Finalize();
        return 0;
    }
    const unsigned N = atoi(argv[1]);
    const unsigned M = atoi(argv[2]);

    if( rank == 0 )
    {
        cout << "Will distribute " << M << " random sources over the " << 
        endl << "frequency domain, which will be split into " << N << " " << 
        endl << "boxes in each of the " << d << " dimensions and " << 
        endl << "distributed amongst " << size << " processes." << endl << endl;
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
        Array<double,d> myFreqBoxWidths;
        Array<double,d> myFreqBoxOffsets;
        InitialLocalFreqData
        ( myFreqBoxWidths, myFreqBoxOffsets, MPI_COMM_WORLD );

        // Now generate random sources across the domain and store them in 
        // our local list when appropriate
        vector< Source<double,d> > mySources;
        vector< Source<double,d> > globalSources( M );
        for( unsigned i=0; i<M; ++i )
        {
            for( unsigned j=0; j<d; ++j )
                globalSources[i].p[j] = Uniform<double>();  // [0,1]
            globalSources[i].magnitude = 200*Uniform<double>()-100; 

            // Check if we should push this source onto our local list
            bool isMine = true;
            for( unsigned j=0; j<d; ++j )
            {
                double u = globalSources[i].p[j];
                double start = myFreqBoxOffsets[j];
                double stop = myFreqBoxOffsets[j] + myFreqBoxWidths[j];
                if( u < start || u >= stop )
                    isMine = false;
            }
            if( isMine )
                mySources.push_back( globalSources[i] );
        }

        // Create vectors for storing the results and then run the algorithm
        UpWave upWave;
        unsigned numLocalLRPs = NumLocalLRPs<d>( N, MPI_COMM_WORLD );
        vector< LRP<double,d,q> > myUpWaveLRPs
        ( numLocalLRPs, LRP<double,d,q>(upWave,N) );
        Transform( upWave, N, mySources, myUpWaveLRPs, MPI_COMM_WORLD );

        // Evaluate each processes' low rank potentials at their center
        for( int i=0; i<size; ++i )
        {
            if( i == rank )
            {
                cout << "Process " << i << ":" << endl;
                for( unsigned k=0; k<myUpWaveLRPs.size(); ++k )
                {
                    Array<double,d> x0 = myUpWaveLRPs[k].GetSpatialCenter();
                    complex<double> u = myUpWaveLRPs[k]( x0 );

                    complex<double> uTruth(0.,0.);
                    for( unsigned m=0; m<globalSources.size(); ++m )
                    {
                        double alpha = 
                            TwoPi*upWave(x0,globalSources[m].p);
                        uTruth += complex<double>(cos(alpha),sin(alpha))*
                                  globalSources[m].magnitude;
                    }

                    cout << "  x0: ";
                    for( unsigned j=0; j<d; ++j )
                        cout << fixed << x0[j] << " ";
                    cout << endl;
                    cout << fixed << "  u(x0): " << u << endl;
                    cout << fixed << "  uTruth(x0): " << uTruth << endl;
                    cout << scientific << "  relative error: " 
                         << abs((u-uTruth)/uTruth) << endl << endl;
                }
            }
            MPI_Barrier( MPI_COMM_WORLD );
        }
    }
    catch( const char* errorMsg )
    {
        cout << "Caught exception on process " << rank << ":" << endl;
        cout << "  " << errorMsg << endl;
    }

    MPI_Finalize();
    return 0;
}

