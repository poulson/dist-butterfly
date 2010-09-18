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
    std::cout << "HTreeWalker <N>" << std::endl
              << "  N: the number of indices of the HTree to iterate over" 
              << std::endl << std::endl;
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

    if( argc != 2 )
    {
        if( rank == 0 )
            Usage();
        MPI_Finalize();
        return 0;
    }
    const std::size_t N = atoi(argv[1]);

    try
    {
        if( rank == 0 )
        {
            bfio::HTreeWalker<d> walker;
            for( std::size_t i=0; i<N; ++i, walker.Walk() )
            {
                const std::tr1::array<std::size_t,d> A = walker.State();
                const size_t k = bfio::FlattenHTreeIndex( A );
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
        msg << "Caught exception on process " << rank << ":" << std::endl
            << "   " << e.what() << std::endl;
        std::cout << msg.str();
    }

    MPI_Finalize();
    return 0;
}

