OpenRLHF Documentation
=====================

This repository contains the Sphinx documentation for `OpenRLHF <https://github.com/OpenRLHF/OpenRLHF>`_.

Build locally
-------------

.. code-block:: bash

   python -m pip install -r docs/requirements.txt
   make -C docs html

Then open ``docs/build/html/index.html`` in your browser.

Contributing
------------

- Edit sources under ``docs/source/`` (reStructuredText, ``.rst``).
- Keep pages **non-redundant**: prefer linking to the canonical page instead of copying long command blocks.
