#include <iostream>
using namespace std;

template<typename F>
void
Foo()
{ cout << F::Eval(3.,4.) << endl; }

struct Mult
{ static inline double Eval(double x,double p) { return x*p; } };

struct DoubleMult
{ static inline double Eval(double x,double p) { return 2.*x*p; } };

int
main
( int argc, char* argv[] )
{
    Foo<Mult>();
    Foo<DoubleMult>();

    return 0;
}

