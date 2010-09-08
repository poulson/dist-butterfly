Introduction
============

``butterflyfio`` is a set of C++ routines for applying Fourier integral
operators (FIOs) in arbitrary-dimensional domains on distributed-memory 
machines. It makes use of a so-called *butterfly algorithm* that achieves the 
near-optimal computational complexity of :math:`\mathcal{O}(N^d \log_2(N))` in 
:math:`d` dimensions. Its only dependencies are BLAS and MPI libraries.

The project is hosted at: http://code.google.com/p/butterflyfio

Alternatively, you can check out the source code with::

  $ hg clone https://butterflyfio.googlecode.com/hg butterflyfio
  $ cd butterflyfio

The goal of the project is to efficiently approximate the application of 
operators of the form

.. math:: f(x) := \int a(x,\xi) e^{2\pi i\Phi(x,\xi)} \hat f(\xi) d\xi,
on large parallel machines, where :math:`\Phi(x,\xi)` is required to be 
real-valued and also linear in its second argument.

The spatial and frequency domains are currently assumed to be the 
:math:`d`-dimensional unit cube :math:`[0,1]^d`.

