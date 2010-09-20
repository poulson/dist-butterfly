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
#include "bfio.hpp"
using namespace std;
using namespace bfio;

void 
Usage()
{
    cout << "ConstrainedHTreeWalker <N> <log2Dim[0]> ... <log2Dim[d-1]>" 
         << endl;
    cout << "  N: number of indices of the HTree to iterate over" << endl;
    cout << "  log2Dim[j]: log2 of the number of boxes in dimension j" << endl;
    cout << endl;
}

static const unsigned d = 3;

int
main
( int argc, char* argv[] )
{
    int rank;
    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    if( argc != 2+d )
    {
        if( rank == 0 )
            Usage();
        MPI_Finalize();
        return 0;
    }
    const unsigned N = atoi(argv[1]);
    Array<unsigned,d> log2BoxesPerDim;
    for( unsigned j=0; j<d; ++j )
        log2BoxesPerDim[j] = atoi(argv[2+j]);

    try
    {
        if( rank == 0 )
        {
            ConstrainedHTreeWalker<d> walker( log2BoxesPerDim );
            for( unsigned i=0; i<N; ++i, walker.Walk() )
            {
                Array<unsigned,d> A = walker.State();
                cout << i << ": ";
                for( unsigned j=0; j<d; ++j )
                    cout << A[j] << " ";
                cout << "; flattened=" 
                     << FlattenConstrainedHTreeIndex( A, log2BoxesPerDim ) 
                     << endl;
            }
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

