Contributing guide
===========================

Hi! Thanks for taking the time to contribute to DISDRODB.

You can contribute in many ways :

-  Join the
   `discussion <https://github.com/ltelab/disdrodb/discussions>`__

Before submitting your contribution, please make sure to take a moment
and read through the following guidelines :

-  `Code of
   Conduct <https://github.com/ltelab/disdrodb/blob/main/CODE_OF_CONDUCT.md>`__
-  `Issue Reporting Guidelines <#issue-reporting-guidelines>`__
-  `Pull Request Guidelines <#pull-request-guidelines>`__
-  `Development Setup <#development-setup>`__
-  `Project Structure <#project-structure>`__
-  `Github Flow <#github-flow>`__
-  `Commit Lint <#commit-lint>`__

Issue Reporting Guidelines
--------------------------

-  Always use `issue
   templates <https://github.com/ltelab/disdrodb/issues/new/choose>`__
-  If you don’t find a corresponding issue template please use the
   template to ask a new template

Contribution Guidelines
-----------------------

**We Develop with Github !**

We use github to host code, to track issues and feature requests, as
well as accept pull requests.

We Use `Github
Flow <https://docs.github.com/en/get-started/quickstart/github-flow>`__,
So All Code Changes Happen Through Pull Requests

Pull requests are the best way to propose changes to the codebase (we
use `Github
Flow <https://docs.github.com/en/get-started/quickstart/github-flow>`__).
We actively welcome your pull requests:

First Time Contributors
-----------------------

-  Setting up the development environment
-  Install pre-commit hooks
-  Respect the Code Style guide

Setting up the development environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You will need python to set up the development environment. See
`README.md <https://github.com/ltelab/disdrodb/blob/main/README.md>`__
for further explanations.

Install pre-commit hooks
~~~~~~~~~~~~~~~~~~~~~~~~

After setting up your development environment, install the git
pre-commit hook by executing the following command in the repository’s
root:

::

   pre-commit install

The pre-commit hooks are scripts executed automatically in every commit
to identify simple issues with the code. When an issue is identified
(the pre-commit script exits with non-zero status), the hook aborts the
commit and prints the error. Currently, DISDRODB only tests that the
code to be committed complies with black’s format style. In case that
the commit is aborted, you only need to run black in the entire source
code. This can be done by running ``black .`` or
``pre-commit run --all-files``. The latter is recommended since it
indicates if the commit contained any formatting errors (that are
automatically corrected).

IMPORTANT: Periodically update the black version used in the pre-commit
hook by running:

::

   pre-commit autoupdate

Respect the Code Style guide
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We follow the pep8 and the python-guide writing style

-  `Code Style — The Hitchhiker's Guide to
   Python <https://docs.python-guide.org/writing/style/>`__

To ensure a minimal style consistency, we use
`black <https://black.readthedocs.io/en/stable/>`__ to auto-format to
the source code. The black configuration used in the pysteps project is
defined in the pyproject.toml, and it is automatically detected by
black.

Black can be run using the following command :

::

   pre-commit run --all-file

**Docstrings**

Every module, function, or class must have a docstring that describe its
purpose and how to use it. The docstrings follows the conventions
described in the `PEP
257 <https://www.python.org/dev/peps/pep-0257/#multi-line-docstrings>`__
and the `Numpy’s docstrings
format <https://numpydoc.readthedocs.io/en/latest/format.html>`__.

Here is a summary of the most important rules:

-  Always use triple quotes for doctrings, even if it fits a single
   line.

-  For one-line docstring, end the phrase with a period.

-  Use imperative mood for all docstrings (“””Return some value.”””)
   rather than descriptive mood (“””Returns some value.”””).

Here is an example of a docstring:

::

    def adjust_lag2_corrcoef1(gamma_1, gamma_2):
       """
       A simple adjustment of lag-2 temporal autocorrelation coefficient to
       ensure that the resulting AR(2) process is stationary when the parameters
       are estimated from the Yule-Walker equations.

       Parameters
       ----------
       gamma_1 : float
         Lag-1 temporal autocorrelation coeffient.
       gamma_2 : float
         Lag-2 temporal autocorrelation coeffient.

       Returns
       -------
       out : float
         The adjusted lag-2 correlation coefficient.
       """


If you are using VS code, you can install the  `autoDocstring <https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring>`_ extention to automatically create such preformatted docstring. 

You should configure VS code as follow : 


.. image:: /static/vs_code_settings.png


How to contribute ?
-------------------

Here is a brief overview of the contribution steps that each DISDRODB
must follow.

1. Fork the repository.
2. Create a new branch for each contribution.
3. Work on your changes.
4. Test your changes.
5. Push you changes to your fork repository.
6. Create a new Pull Request in GitHub.


.. image:: /static/collaborative_process.png




Fork the repository
~~~~~~~~~~~~~~~~~~~

Once you have set the development environment, the next step is creating
your local copy of the repository, where you will commit your
modifications. The steps to follow are:

1. Set up Git on your computer

2. Create a GitHub account (if you don’t have one)

3. Fork the repository in your GitHub.

4. Clone a local copy of your fork. For example:

::

   git clone https://github.com/<your-account>/disdrodb.git

Done!, now you have a local copy of disdrodb git repository.

Create a new branch
~~~~~~~~~~~~~~~~~~~

As a collaborator, all the new contributions you want should be made in
a new branch under your forked repository. Working on the master branch
is reserved for Core Contributors only. Core Contributors are developers
that actively work and maintain the repository. They are the only ones
who accept pull requests and push commits directly to the pysteps
repository.

For more information on how to create and work with branches, see
`“Branches in a
Nutshell” <https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell>`__
in the Git documentation

Branch name must be define as follow :

-  Add a reader: ``reader-<institute>-<campaign>``
-  Fix a bug: ``bugfix-<some_key>-<word>``
-  Improve the doc: ``doc-<some_key>-<word>``
-  Add a new feature: ``feature-<some_key>-<word>``
-  Refactor some code: ``refactor-<some_key>-<word>``
-  Optimize some code: ``optimize-<some_key>-<word>``

Work on your changes
~~~~~~~~~~~~~~~~~~~~

Here again, respect the Respect the Code Style guide.



Test of changes
~~~~~~~~~~~~~~~

Evrey changes must be tested !

DISDRDB tests are written using the third-party pytest package. There is usually no need to run all DISDRDB tests, but instead only run the tests related to the component you are working on. All tests are automatically run from the GitHub Pull Request using multiple versions of Python, multiple operating systems, and multiple versions of dependency libraries. If you want to run all DISDRDB tests you will need to install additional dependencies that aren’t needed for regular DISDRDB usage. To install them run:

.. code-block:: bash

	pip install -e .[tests]


DISDRDB tests can be executed by running:

.. code-block:: bash

	pytest disdrodb/tests


You can also run a specific tests by specifying a sub-directory or module:

.. code-block:: bash

	pytest satpy/tests/reader_tests/<reader_name>.py


.. warning:: 
   If you plan to create a new reader, your pull request must include a test file. This file must be name accordingly to the reader name with the test prefix. The test must simulate the reading of data with a small data sample.






Push you changes to your fork repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

During this process, pre-commit hooks will be run. Your commit will be
allowed only if quality requirements are fulfil.

If you encounter errors, Black can be run using the following command :

::

   pre-commit run --all-file

We follow a commit message convention, to have consistent git messages.
The goal is to increase readability and ease of contribution

-  we use
   `commit-lint <https://github.com/conventional-changelog/commitlint>`__

Create a new Pull Request in GitHub.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once your code has been uploaded into your DISTRODB fork, you can create
a Pull request to the DISDRODB main branch.

**Recommendation for the pull request**

-  Add screenshots or GIFs for any UI changes. This will help the person
   reviewing your code to understand what you’ve changed and how it
   works.

   -  *Hint:
      use*\ `Kap <https://getkap.co/>`__\ *or*\ `Licecap <https://www.cockos.com/licecap/>`__\ *to
      record your screen.*

-  If your team uses the particular template provided for pull requests,
   fill it.

-  It’s OK to have multiple small commits as you work on the PR - GitHub
   will automatically squash it before merging.

-  If adding a new feature:

   -  Add accompanying test case.
   -  Provide a convincing reason to add this feature. Ideally, you
      should open a suggestion issue first and have it approved before
      working on it.
   -  Present your issue in the ‘discussion’ part of this repo

-  If fixing bug:

   -  If you are resolving a special issue, add ``(fix #xxxx[,#xxxx])``
      (#xxxx is the issue id) in your PR title for a better release log,
      e.g. ``update entities encoding/decoding (fix #3899)``.
   -  Provide a detailed description of the bug in the PR. Live demo
      preferred.
   -  Add appropriate test coverage if applicable.

.. _section-1:

Code review checklist
---------------------

-  Ask to people to review your code:

   -  a person who knows the domain well and can spot bugs in the
      business logic;
   -  an expert in the technologies you’re using who can help you
      improve the code quality.

-  When you’re done with the changes after a code review, do another
   self review of the code and write a comment to notify the reviewer,
   that the pull request is ready for another iteration.
-  Review all the review comments and make sure they are all addressed
   before the next review iteration.
-  Make sure you don’t have similar issues anywhere else in your pull
   request.
-  If you’re not going to follow any of the code review recommendations,
   please add a comment explaining why.
-  Avoid writing comment like “done” of “fixed” on each code review
   comment. Reviewers assume you’ll do all suggested changes, unless you
   have a reason not to do some of them.
-  Sometimes it’s okay to postpone changes — in this case you’ll need to
   add a ticket number to the pull request and to the code itself.

.. _section-2:

Financial Contribution
----------------------

We also welcome financial contributions. Please contact us directly.

Credits
-------

Thank you to all the people who have already contributed to DISDRDB
repository!
