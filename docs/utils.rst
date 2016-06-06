.. use latex citation for nice bibliography
.. usage: Lorem ipsum :cite:`Latin006`

.. role:: cite

.. raw:: LaTex

   \providecommand*\DUrolecite[1]{\cite{#1}}


.. colored text

.. role:: red

.. raw:: LaTex

    \providecommand*\DUrolered[1]{{\color{red} {#1}}}


.. reference of figures

.. role:: ref

.. role:: label

.. raw::  latex

  \newcommand*{\docutilsroleref}{\ref}
  \newcommand*{\docutilsrolelabel}{\label}
