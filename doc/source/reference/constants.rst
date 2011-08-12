---------
Constants
---------

"""""""""""
:math:`\pi`
"""""""""""

.. c:var:: double Pi

   :math:`\pi` rounded to sixteen digits of accuracy.

""""""""""""
:math:`2\pi`
""""""""""""

.. c:var:: double TwoPi

   :math:`2\pi` rounded to sixteen digits of accuracy.

"""""""""""""""""""""""""
Compile-time exponentials
"""""""""""""""""""""""""

.. cpp:type:: Pow<x,y>

   A recursive template class used for computing unsigned integer exponentials 
   at compile-time.

   .. cpp:member:: unsigned val

      Returns :math:`x^y`.

   **Example usage:**

   .. code-block:: cpp

      // Compute q^d at compile-time and print the result
      const unsigned q = 8;
      const unsigned d = 3;
      const unsigned q_to_d = bfio::Pow<q,d>::val;
      std::cout << "q^d = " << q_to_d << std::endl;

