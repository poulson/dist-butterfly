#include <iostream>
using namespace std;

template<unsigned x,unsigned y>
struct Power
{ enum { value = x * Power<x,y-1>::value }; };

template<unsigned x>
struct Power<x,1>
{ enum { value = x }; };

int
main
( int argc, char* argv[] )
{
    double weights[ Power<2,4>::value ];
    cout << Power<3,2>::value << endl;

    for( unsigned i=0; i<Power<2,4>::value; ++i )
        cout << i << endl;

    return 0;
}

