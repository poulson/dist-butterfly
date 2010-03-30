#include "include/BFIO.h"
using namespace std;
using namespace BFIO;

// Create a functor that performs a dot product in R^d
template<typename R,unsigned d>
struct Dot
{ 
    static inline complex<R>
    Eval
    ( const Array<R,d>& x, const Array<R,d>& p )
    {
        complex<R> z(0.,0.);
        for( unsigned j=0; j<d; ++j )
            z += x[j]*p[j];
        return z;
    }
};

int
main
( int argc, char* argv[] )
{
    int rank;
    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    vector< Source<double,2> > mySources;
    vector< LRP<Dot<double,2>,double,2,5> > myLRPs;

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
    catch( ... )
    {
        cout << "Caught exception on process " << rank << endl;
    }

    MPI_Finalize();
    return 0;
}

