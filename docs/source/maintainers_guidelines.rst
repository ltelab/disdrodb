========================
Maintainers guidelines
========================

The section is dedicated to the DISDRODB core developers (maintainers). 


List of the core contributors 
=================================

* Current Release Manager : Ghiggi Gionata
* Testing Team : Ghiggi Gionata

To do  : define roles here 



Versions guidelines
========================

DISDRODB uses  `Semantic <https://semver.org/>`_ Versioning. Each release is associated with a git tag of the form X.Y.Z.

Given a version number in the MAJOR.MINOR.PATCH (eg., X.Y.Z) format, here are the differences in these terms:

- MAJOR version - make breaking/incompatible API changes
- MINOR version - add functionality in a backwards compatible manner. Adding new reader
- PATCH version - make backwards compatible bug fixes


Breaking vs. non-breaking changes
-----------------------------------

Since disdrodb is used by a broad ecosystem of both API consumers and implementers, it needs a strict definition of what changes are “non-breaking” and are therefore allowed in MINOR and PATCH releases.

In the DISDRODB spec, a breaking change is any change that requires either consumers or implementers to modify their code for it to continue to function correctly.

Examples of breaking changes include:

- Adding or removing a required endpoint or field
- Adding or removing a request parameter
- Changing the data type or semantics of an existing field, including clarifying previously-ambiguous requirements

Examples of non-breaking changes include:

- Adding or removing an optional endpoint or field
- Adding or removing enum values
- Modifying documentation or spec language that doesn't affect the behavior of the API directly

One implication of this policy is that clients should be prepared to ignore the presence of unexpected fields in responses and unexpected values for enums. This is necessary to preserve compatibility between PATCH versions within the same MINOR version range, since optional fields and enum values can be added as non-breaking changes.


Ongoing version support
-----------------------------------

DISDRODB major releases move the community forward, focusing on spec stabilization and major feature additions, rather than backwards-compatibility. DISDRODB minor releases will be backwards compatible. We strongly recommend adopting the latest release of DISDRODB into production within 6 months for major releases, and 4 months for minor releases.

The `LTE <https://https://www.epfl.ch/labs/lte/>`_ supports DISDRODB releases for 2 years. Recommended versions are supported and maintained by the `LTE <https://https://www.epfl.ch/labs/lte/>`_  and our community – we provide updated guidance and documentation, track issues, and provide bug fixes and critical updates in the form of hotfixes for these versions. Releases that are 2 years or older are deprecated.

Refer to the list of Recommended Releases to see current releases and more details. To Do : add such list


Documentation pipepline
========================

DISDRODB’s documentation is built using Sphinx. All documentation lives in the docs/ directory of the project repository. 


Manual documentation creation 
-----------------------------

After editing the source files there the documentation can be generated locally:


.. code-block:: bash

	sphinx-build -b html source build


The output of the make command should be checked for warnings and errors. If code has been changed (new functions or classes) then the API documentation files should be regenerated before running the above command:

.. code-block:: bash

	sphinx-apidoc -f -o source/api ..


Automatic (Github) documentation creation 
------------------------------------------


One webhook is defined in the reopsitory to trigger the publication process to readthedoc.io. 

This webhook is linked to the DISDRODB core developper XXX.

.. image:: /static/documentation_pipepline.png

Ghiggi Gionata owns the readthedoc account.  


Package releases pipepline
============================

One github action is defined to trigger the packaging and the upload on pypi.org.

.. image:: /static/package_pipepline.png

The pypi project is shared beween the core contributors



Reviewing process 
============================






Testing processes
============================

To do : define test process


