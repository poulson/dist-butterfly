#include <iostream>
using namespace std;

template<bool b>
struct If
{ static inline void Eval(unsigned j) { cout << j << endl; } };

template<>
struct If<false> 
{ static inline void Eval(unsigned j) {} };

template<unsigned i,unsigned j>
struct LoopDown
{ static inline void Eval() { If<i!=j>::Eval(j); LoopDown<i,j-1>::Eval(); } };

template<unsigned i>
struct LoopDown<i,0>
{ static inline void Eval() { If<i!=0>::Eval(0); } };

int
main
( int argc, char* argv[] )
{
    LoopDown<7,10>::Eval();

    return 0;
}

