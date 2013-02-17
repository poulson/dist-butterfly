/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of DistButterfly and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#include "dist-butterfly.hpp"
using namespace std;

namespace {
void 
Usage()
{
    cout << "HTreeWalker <N>\n"
         << "  N: the number of indices of the HTree to iterate over\n" 
         << endl;
}
} // anonymous namespace

static const size_t d = 2;

int
main( int argc, char* argv[] )
{
    int rank;
    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    if( argc != 2 )
    {
        if( rank == 0 )
            Usage();
        MPI_Finalize();
        return 0;
    }
    const size_t N = atoi(argv[1]);

    try
    {
        if( rank == 0 )
        {
            dbf::HTreeWalker<d> walker;
            for( size_t i=0; i<N; ++i, walker.Walk() )
            {
                const array<size_t,d>& A = walker.State();
                const size_t k = dbf::FlattenHTreeIndex( A );
                cout << i << ": ";
                for( size_t j=0; j<d; ++j )
                    cout << A[j] << " ";
                cout << "; flattened=" << k << endl;
            }
        }
    }
    catch( const exception& e )
    {
        ostringstream msg;
        msg << "Caught exception on process " << rank << ":\n"
            << "   " << e.what();
        cout << msg.str() << endl;
    }

    MPI_Finalize();
    return 0;
}
