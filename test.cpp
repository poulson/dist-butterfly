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

// Create a functor that performs a dot product in R^2
struct Dot
{ 
    static inline double
    Eval
    ( const Array<double,2>& x, const Array<double,2>& p )
    { return x[0]*p[0] + x[1]*p[1]; }
};

int
main
( int argc, char* argv[] )
{
    int rank;
    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    vector< Source<double,2> > mySources;
    vector< LRP<Dot,double,2,5> > myLRPs;

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
        Transform( 4, mySources, myLRPs, MPI_COMM_WORLD );
    }
    catch( const char* errorMsg )
    {
        cout << "Caught exception on process " << rank << ":" << endl;
        cout << "  " << errorMsg << endl;
    }

    MPI_Finalize();
    return 0;
}

