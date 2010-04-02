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

