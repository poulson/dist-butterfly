/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of ButterflyFIO and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#include "bfio.hpp"

namespace {
void 
Usage()
{
    std::cout << "ConstrainedHTreeWalker <N> <log2Dim[0]> ... <log2Dim[d-1]>\n" 
              << "  N: number of indices of the HTree to iterate over\n" 
              << "  log2Dim[j]: log2 of the number of boxes in dimension j\n" 
              << std::endl;
}
} // anonymous namespace

static const std::size_t d = 3;

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
    const std::size_t N = atoi(argv[1]);
    bfio::Array<std::size_t,d> log2BoxesPerDim;
    for( std::size_t j=0; j<d; ++j )
        log2BoxesPerDim[j] = atoi(argv[2+j]);

    try
    {
        if( rank == 0 )
        {
            bfio::ConstrainedHTreeWalker<d> walker( log2BoxesPerDim );
            for( std::size_t i=0; i<N; ++i, walker.Walk() )
            {
                const bfio::Array<std::size_t,d> A = walker.State();
                const size_t k = 
                    bfio::FlattenConstrainedHTreeIndex( A, log2BoxesPerDim );
                std::cout << i << ": ";
                for( std::size_t j=0; j<d; ++j )
                    std::cout << A[j] << " ";
                std::cout << "; flattened=" << k << std::endl;
            }
        }
    }
    catch( const std::exception& e )
    {
        std::ostringstream msg;
        msg << "Caught exception on process " << rank << ":\n"
            << "   " << e.what();
        std::cout << msg.str() << std::endl;
    }

    MPI_Finalize();
    return 0;
}

