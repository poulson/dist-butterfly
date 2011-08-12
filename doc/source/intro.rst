``butterflyfio`` is a set of C++ routines for applying Fourier integral
operators (FIOs) in arbitrary-dimensional domains on distributed-memory 
machines. It makes use of a so-called *butterfly algorithm* that achieves the 
near-optimal computational complexity of :math:`\mathcal{O}(N^d \log_2(N))` in 
:math:`d` dimensions. Its only dependencies are BLAS and MPI libraries.

The goal of the project is to efficiently approximate the application of 
operators of the form

.. math:: f(x) := \int a(x,\xi) e^{2\pi i\Phi(x,\xi)} \hat f(\xi) d\xi,
on large parallel machines, where :math:`a(x,\xi)` is the complex-valued 
*amplitude function* and :math:`\Phi(x,\xi)` is the real-valued 
*phase function*.

The spatial and frequency domains are each allowed to be arbitrary boxes aligned
with :math:`R^d`.

