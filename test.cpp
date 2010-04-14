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
#include "BFIO.hpp"
using namespace std;
using namespace BFIO;

void 
Usage()
{
    cout << "test <N>" << endl;
    cout << "  N: power of 2, the frequency spread in each dimension" << endl;
    cout << endl;
}

// Create a functor that performs a dot product in R^2
struct Dot
{ 
    static inline double
    Eval
    ( const Array<double,2>& x, const Array<double,2>& p )
    { return x[0]*p[0] + x[1]*p[1]; }
};

#define d 2
#define q 5

int
main
( int argc, char* argv[] )
{
    int rank, size;
    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );

    if( argc != 2 )
    {
        if( rank == 0 )
            Usage();
        MPI_Finalize();
        return 0;
    }
    const unsigned N = atoi(argv[1]);

    vector< Source<double,d> > mySources;
    vector< LRP<Dot,double,d,q> > myLRPs;

    // See what happens when we only put sources in the bottom-left corner
    if( rank == 0 )
    {
        mySources.resize(2);

        mySources[0].p[0] = 0.1;
        mySources[0].p[1] = 0.12;
        mySources[0].magnitude = 7.8;

        mySources[1].p[0] = 0.0;
        mySources[1].p[1] = 0.0;
        mySources[1].magnitude = 18.;
    }
    else
    {
        mySources.resize(0);
    }
    
    try
    {
        Transform( N, mySources, myLRPs, MPI_COMM_WORLD );

        // Evaluate each processes' low rank potentials at their center
        for( unsigned i=0; i<size; ++i )
        {
            if( i == rank )
            {
                cout << "Process " << i << ":" << endl;
                for( unsigned k=0; k<myLRPs.size(); ++k )
                {
                    Array<double,d> x0 = myLRPs[i].x0;
                    cout << "  x0: " << x0[0] << "," << x0[1] << endl;
                    complex<double> u = myLRPs[i]( x0 );
                    cout << "  u(x0): " << u << endl << endl;
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

