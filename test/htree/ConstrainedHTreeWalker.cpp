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

