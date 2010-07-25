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
#include "bfio.hpp"
using namespace std;
using namespace bfio;

void 
Usage()
{
    cout << "GeneralizedRadon <N> <M> <testAccuracy?>" << endl;
    cout << "  N: power of 2, the frequency spread in each dimension" << endl;
    cout << "  M: number of random sources to instantiate" << endl;
    cout << "  testAccuracy?: tests accuracy iff 1" << endl;
    cout << endl;
}

static const unsigned d = 2;
static const unsigned q = 8;

class GenRadon : public PhaseFunctor<double,d>
{
    double c1( const Array<double,d>& x ) const
    { return (2+sin(TwoPi*x[0])*sin(TwoPi*x[1]))/3.; }

    double c2( const Array<double,d>& x ) const
    { return (2+cos(TwoPi*x[0])*cos(TwoPi*x[1]))/3.; }
public:
    double
    operator() ( const Array<double,d>& x, const Array<double,d>& p ) const
    {
        double a = c1(x)*p[0];
        double b = c2(x)*p[1];
        return x[0]*p[0]+x[1]*p[1] + sqrt(a*a+b*b);
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

    const unsigned L = Log2( N );
    const unsigned s = Log2( size );
    if( s > d*L )
    {
        if( rank == 0 )
            cout << "Cannot run with more than N^d processes." << endl;
        MPI_Finalize();
        return 0;
    }

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
                    mySources[i].p[j] = myFreqBoxOffsets[j] + 
                                        Uniform<double>()*myFreqBoxWidths[j];
                }
                mySources[i].magnitude = 200*Uniform<double>()-100;
            }
        }

        // Create vectors for storing the results and then run the algorithm
        GenRadon genRadon;
        unsigned numLocalLRPs = NumLocalBoxes<d>( N, MPI_COMM_WORLD );
        vector< LowRankPotential<double,d,q> > myGenRadonLRPs
        ( numLocalLRPs, LowRankPotential<double,d,q>(genRadon,N) );

        MPI_Barrier( MPI_COMM_WORLD );
        double startTime = MPI_Wtime();
        FreqToSpatial( genRadon, N, mySources, myGenRadonLRPs, MPI_COMM_WORLD );
        MPI_Barrier( MPI_COMM_WORLD );
        double stopTime = MPI_Wtime();
        if( rank == 0 )
            cout << "Runtime: " << stopTime-startTime << " seconds." << endl;

        // Evaluate each processes' low rank potentials at a random point
        if( testAccuracy )
        {
            double myMaxRelError = 0.;
            for( unsigned k=0; k<myGenRadonLRPs.size(); ++k )
            {
                // Retrieve the spatial center of LRP k
                Array<double,d> x0 = 
                    myGenRadonLRPs[k].GetSpatialCenter();

                // Find a random point in that box
                Array<double,d> x;
                for( unsigned j=0; j<d; ++j )
                    x[j] = x0[j] + 1./(2*N)*(2*Uniform<double>()-1.);

                // Evaluate our LRP at x  and compare against truth
                complex<double> u = myGenRadonLRPs[k]( x );
                complex<double> uTruth(0.,0.);
                for( unsigned m=0; m<globalSources.size(); ++m )
                {
                    double alpha = 
                        TwoPi*genRadon(x,globalSources[m].p);
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
                cout << "Maximum relative error: " << maxRelError << endl;
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

