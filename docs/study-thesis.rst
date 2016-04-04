.. include:: utils.rst

==============================================
Depth-Aware Motion Deblurring in Stereo Images
==============================================

Study Thesis
============

:Author: Franziska Krüger
:Organization: TU Dresden
:Contact: Franziska.Krueger1@tu-dresden.de
:Date: 21/03/2016
:Supervisor: Dr.-Ing. Anita Sellent
:Abstract:
  This study thesis provides a reference implementation of the depth-aware
  motion deblurring algorithm from Xu and Jia. This algorithm deblurs scenes
  with depth variations. For solving this task a spatially-varying blur kernel is needed. 
  This kernel can be obtained with depth information from a blurred stereo image pair.
  But small-size regions lack necessary information for the kernel estimation. So a hierarchically
  approach named region tree is used to overcome this problem.
  The challenges and the results of this algorithm will be presented in this study thesis.


.. raw:: LaTex

    \newpage

.. contents:: Table of Contents
   :depth: 3

.. raw:: LaTex

    \newpage


.. section-numbering::
    :depth: 3

++++++++++++
Introduction
++++++++++++

.. include:: chapters/introduction.rst

.. raw:: LaTex

    \newpage


++++++++++++++
Basics of blur
++++++++++++++

.. include:: chapters/blur-basics.rst

.. raw:: LaTex

    \newpage


+++++++++++++++++++++++++++++
Depth-Aware Motion Deblurring
+++++++++++++++++++++++++++++

.. include:: chapters/depth-aware-deblurring.rst

.. raw:: LaTex

    \newpage


++++++++++
Evaluation
++++++++++

.. include:: chapters/evaluation.rst

.. raw:: LaTex

    \newpage


++++++++++
Conclusion
++++++++++

.. include:: chapters/conclusion.rst

.. raw:: LaTex

    \newpage


.. raw:: LaTex

    \bibliography{references}
    \bibliographystyle{alpha}