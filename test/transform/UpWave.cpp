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
    cout << "UpWave <N> <M> <testAccuracy?>" << endl;
    cout << "  N: power of 2, the frequency spread in each dimension" << endl;
    cout << "  M: number of random sources to instantiate" << endl;
    cout << "  testAccuracy?: test accuracy iff 1" << endl;
    cout << endl;
}

static const unsigned d = 3;
static const unsigned q = 8;

class Unity : public AmplitudeFunctor<double,d>
{
public:
    complex<double>
    operator() ( const Array<double,d>& x, const Array<double,d>& p ) const
    { return complex<double>(1); }
};

class UpWave : public PhaseFunctor<double,d>
{
public:
    double
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
        if( rank == 0 )
            cout << "Must run with a power of two number of cores." << endl;
        MPI_Finalize();
        return 0;
    }

    if( argc != 4 )
    {
        if( rank == 0 )
            Usage();
        MPI_Finalize();
        return 0;
    }
    const unsigned N = atoi(argv[1]);
    const unsigned M = atoi(argv[2]);
    const bool testAccuracy = atoi(argv[3]);

    if( rank == 0 )
    {
        ostringstream msg;
        msg << "Will distribute " << M << " random sources over the frequency "
            << "domain, which will be split into " << N
            << " boxes in each of the " << d << " dimensions and distributed "
            << "amongst " << size << " processes." << endl << endl;
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
        Array<double,d> myFreqBoxWidths;
        Array<double,d> myFreqBoxOffsets;
        LocalFreqPartitionData
        ( myFreqBoxWidths, myFreqBoxOffsets, MPI_COMM_WORLD );

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
        }
        else
        {
            unsigned numLocalSources = 
                ( rank<(int)(M%size) ? M/size+1 : M/size );
            mySources.resize( numLocalSources ); 
            for( unsigned i=0; i<numLocalSources; ++i )
            {
                for( unsigned j=0; j<d; ++j )
                {
                    mySources[i].p[j] = myFreqBoxOffsets[j]+
                                        Uniform<double>()*myFreqBoxWidths[j];
                }
                mySources[i].magnitude = 200*Uniform<double>()-100;
            }
        }

        // Set up our amplitude and phase functors
        Unity unity;
        UpWave upWave;

        // Create a vector for storing the results
        unsigned numLocalLRPs = NumLocalBoxes<d>( N, MPI_COMM_WORLD );
        vector< LowRankPotential<double,d,q> > myUpWaveLRPs
        ( numLocalLRPs, LowRankPotential<double,d,q>(unity,upWave,N) );

        // Run the algorithm
        MPI_Barrier( MPI_COMM_WORLD );
        double startTime = MPI_Wtime();
        FreqToSpatial
        ( mySources, myUpWaveLRPs, MPI_COMM_WORLD );
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
                    x[j] = x0[j] + 1./(2*N)*(2*Uniform<double>()-1.);

                // Evaluate our LRP at x and compare against truth
                complex<double> u = myUpWaveLRPs[k]( x );
                complex<double> uTruth(0.,0.);
                for( unsigned m=0; m<globalSources.size(); ++m )
                {
                    double alpha = 
                        TwoPi*upWave(x,globalSources[m].p);
                    uTruth += complex<double>(cos(alpha),sin(alpha))*
                              globalSources[m].magnitude;
                }
                double relError = abs(u-uTruth)/abs(uTruth);
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

