# DistButterfly

A distributed-memory implementation of the butterfly algorithm.
In particular, the current implementation performs analytical interpolation
in order to efficiently apply Egorov-like operators with user-defined 
black-box phase functions. Please see 
[A parallel butterfly algorithm](http://arxiv.org/abs/1305.4650) for more 
details.

### Documentation

Coming soon! For now, please see [this example driver](https://github.com/poulson/dist-butterfly/blob/master/test/transform/GenRadon-3d.cpp), which efficiently
applies an analogue of a generalized Radon transform over a 3D volume. 

Building DistButterfly is often as simple as running:

    cd dist-butterfly
    mkdir build
    cd build
    cmake ..
    make
