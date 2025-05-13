Contributors Guidelines
===========================

Hi! Thank you for taking the time to contribute to DISDRODB.

You can contribute in many ways:

- Join the `GitHub discussions <https://github.com/ltelab/disdrodb/discussions>`__
- Report software `issues <#issue-reporting-guidelines>`__
- Help us develop new readers
- Add new data to the DISDRODB Decentralized Data Archive
- Implement new products
- ...
- Any code improvements are welcome!

**We develop with GitHub!**

We use GitHub to host code, to track issues and feature requests, as well as accept Pull Requests.
We use `GitHub flow <https://docs.github.com/en/get-started/quickstart/github-flow>`__.
So all code changes happen through Pull Requests (PRs).


**First-Time Contributors?**

Before contributing, please take a moment to read through the following sections:

- The `Installation for contributors <https://disdrodb.readthedocs.io/en/latest/installation.html#installation-for-contributors>`__ help you to set up the developing environment and the pre-commit hooks.
- The section `Contributing process <#contributing-process>`__ provides you with a brief overview of the steps that each DISDRODB developer must follow to contribute to the repository.
- The `Code review checklist <#code-review-checklist>`__ enable to speed up the code review process.
- The `Code of conduct <https://github.com/ltelab/disdrodb/blob/main/CODE_OF_CONDUCT.md>`__ details the expected behavior of all contributors.

Initiating a discussion about your ideas or proposed implementations is a vital step before starting your contribution!
Engaging with the community early on can provide valuable insights, ensure alignment with the project's goals, and prevent potential overlap with existing work.
Here are some guidelines to facilitate this process:

1. Start with a conversation

   Before you start coding, open a `GitHub Discussion <https://github.com/ltelab/disdrodb/discussions>`__, a `GitHub Feature Request Issue <https://github.com/ltelab/disdrodb/issues/new/choose>`__ or
   just start a discussion in the `DISDRODB Slack Workspace <https://join.slack.com/t/disdrodbworkspace/shared_invite/zt-25l4mvgo7-cfBdXalzlWGd4Pt7H~FqoA>`__.
   These channels of communication provide an opportunity to gather feedback, understand the project's current state, and improve your contributions.

2. Seek guidance and suggestions

   Utilize the community's expertise. Experienced contributors and maintainers can offer guidance, suggest best practices, and help you navigate any complexities you might encounter.

3. Collaborate on the approach

   Discussing your implementation strategy allows for a collaborative approach to problem-solving.
   It ensures that your contribution is in line with the project's design principles and technical direction.

By following these steps, you not only enhance the quality and relevance of your contribution but also become an integral part of the project's collaborative ecosystem.

If you have any questions, please do not hesitate to ask in the `GitHub Discussions <https://github.com/ltelab/disdrodb/discussions>`__ or in the
`DISDRODB Slack Workspace <https://join.slack.com/t/disdrodbworkspace/shared_invite/zt-25l4mvgo7-cfBdXalzlWGd4Pt7H~FqoA>`__.


Issue Reporting Guidelines
-----------------------------

To facilitate and enhance the issue reporting process, it is important to utilize the predefined GitHub Issue Templates.
These templates are designed to ensure you provide all the essential information in your report, allowing for a faster and more effective response from the maintainers.
You can access and use these templates by visiting the `GitHub Issue Templates page here <https://github.com/ltelab/disdrodb/issues/new/choose>`__.

However, if you find that the existing templates don't quite match the specifics of the issue you're encountering, please feel free to suggest a new template.
Your feedback is invaluable in refining our processes and ensuring we address a broader spectrum of concerns.
To do this, simply create a general issue in the repository, clearly stating that you're requesting a new template and include detailed suggestions about what this new template should entail.
This proactive approach helps us continuously evolve and better serve the needs of the project and its contributors.


Contributing process
-----------------------

Here is a brief overview of the steps that each DISDRODB developer must follow to contribute to the repository.

1. Fork the repository.
2. Create a new branch for each contribution.
3. Work on your changes.
4. Test your changes.
5. Push your local changes to your fork repository.
6. Create a new Pull Request in GitHub.


.. image:: https://github.com/ltelab/disdrodb/blob/main/docs/source/static/collaborative_process.png?raw=true


1. Fork the repository and install the development environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you do not have a GitHub account yet, please create one `here <https://github.com/join>`__.
If you do not have yet Git installed on your computer, please install it following `these instructions <https://github.com/git-guides/install-git>`__.
Then, please follow the guidelines in the `Installation for contributors <https://disdrodb.readthedocs.io/en/latest/installation.html#installation-for-contributors>`__ section
to create the local copy of the disdrodb repository, set up the developing environment and the pre-commit hooks.

Once you have a local copy of the disdrodb repository on your machine, you're ready to contribute!


2. Create a new branch
~~~~~~~~~~~~~~~~~~~~~~~

Each contribution should be made in a separate new branch of your forked repository.
For example, if you plan to contribute with new readers, please create a branch for every single reader.
Working on the main branch is reserved for *Core Contributors* only.
Core Contributors are developers that actively work and maintain the repository.
They are the only ones who accept Pull Requests and push commits directly to the DISDRODB repository.

For more information on how to create and work with branches, see
`“Branches in a Nutshell” <https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell>`__
in the Git documentation.

Please define the name of your branch based on the scope of the contribution. Try to strictly stick to the following guidelines:

-  If you add a reader, use : ``reader-<data_source*>-<campaign>``
-  If you fix a bug: ``bugfix-<some_key>-<word>``
-  If you improve the documentation: ``doc-<some_key>-<word>``
-  If you add a new feature: ``feature-<some_key>-<word>``
-  If you refactor some code: ``refactor-<some_key>-<word>``
-  If you optimize some code: ``optimize-<some_key>-<word>``


\* Guidelines for the ``<data_source>``
- 	We use the institution name when campaign data spans more than 1 country (i.e. ARM, GPM)
- 	We use the country name when all campaigns (or sensor networks) are inside a given country.



3. Work on your changes
~~~~~~~~~~~~~~~~~~~~~~~~~~

When you are working on your changes, please stick with the repository's coding style and documentation rules.

**Code Style**

We follow the `PEP 8 <https://pep8.org/>`__ style guide for python code.
Another relevant style guide can be found in the `The Hitchhiker's Guide to Python <https://docs.python-guide.org/writing/style/>`__.

To ensure a minimal style consistency, we use `black <https://black.readthedocs.io/en/stable/>`__ to auto-format the source code.
The *black* configuration used in the DISDRODB project is
defined in the `pyproject.toml <https://github.com/ltelab/disdrodb/blob/main/pyproject.toml>`__.


**Code Documentation**

Every module, function, or class must have a docstring that describes its purpose and how to use it.
The docstrings follows the conventions described in the `PEP 257 <https://www.python.org/dev/peps/pep-0257/#multi-line-docstrings>`__
and the `Numpy's docstrings format <https://numpydoc.readthedocs.io/en/latest/format.html>`__.

Here is a summary of the most important rules:

-  Always use triple quotes for doctrings, even if it fits a single
   line.

-  For one-line docstring, end the phrase with a period.

-  Use imperative mood for all docstrings (``“””Return some value.”””``)
   rather than descriptive mood (``“””Returns some value.”””``).

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
         Lag-1 temporal autocorrelation coefficient.
       gamma_2 : float
         Lag-2 temporal autocorrelation coefficient.

       Returns
       -------
       out : float
         The adjusted lag-2 correlation coefficient.
       """


If you are using VS code, you can install the  `autoDocstring <https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring>`_
extension to automatically create such preformatted docstring.

You should configure VS code as follow :

.. image:: https://github.com/ltelab/disdrodb/blob/main/docs/source/static/vs_code_settings.png?raw=true
   :alt: VS Code Settings
   :align: center

The convention we adopt for our docstrings is the numpydoc string convention.


.. _code_quality_control:

4. Code quality control
~~~~~~~~~~~~~~~~~~~~~~~~~~

Pre-commit hooks are automated scripts that run during each commit to detect basic code quality issues.
If a hook identifies an issue (signified by the pre-commit script exiting with a non-zero status), it halts the commit process and displays the error messages.

Currently, DISDRODB tests that the code to be committed complies with `black's  <https://github.com/psf/black>`__ format style,
the `ruff <https://github.com/charliermarsh/ruff>`__ linter and the `codespell <https://github.com/codespell-project/codespell>`__ spelling checker.

+-----------------------------------------------------------------------------------------------+------------------------------------------------------------------+------------+-------+
|  Tool                                                                                         | Aim                                                              | pre-commit | CI/CD |
+===============================================================================================+==================================================================+============+=======+
| `Black <https://black.readthedocs.io/en/stable/>`__                                           | Python code formatter                                            | 👍         | 👍    |
+-----------------------------------------------------------------------------------------------+------------------------------------------------------------------+------------+-------+
| `Ruff  <https://github.com/charliermarsh/ruff>`__                                             | Python linter                                                    | 👍         | 👍    |
+-----------------------------------------------------------------------------------------------+------------------------------------------------------------------+------------+-------+
| `Codespell  <https://github.com/codespell-project/codespell>`__                               | Spelling checker                                                 | 👍         | 👍    |
+-----------------------------------------------------------------------------------------------+------------------------------------------------------------------+------------+-------+

The versions of the software used in the pre-commit hooks is specified in the `.pre-commit-config.yaml <https://github.com/ltelab/disdrodb/blob/main/.pre-commit-config.yaml>`__ file.
This file serves as a configuration guide, ensuring that the hooks are executed with the correct versions of each tool, thereby maintaining consistency and reliability in the code quality checks.


If a commit is blocked due to these checks, you can manually correct the issues by running locally the appropriate tool: ``black .`` for Black, ``ruff check .`` for Ruff, or ``codespell`` for Codespell.
Alternatively, you can use the ``pre-commit run --all-files`` command to attempt automatic corrections of all formatting errors across all files.

The Continuous Integration (CI) tools integrated within GitHub employ the same pre-commit hooks to consistently uphold code quality for every Pull Request.

In addition to the pre-commit hooks, the Continuous Integration (CI) setup on GitHub incorporates an extended suite of tools.
These tools, which are not installable on a local setup, perform advanced code quality analyses and reviews after each update to a Pull Request.

Refer to the table below for a comprehensive summary of all CI tools employed to assess the code quality of a Pull Request.

+----------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------+
| Tool                                               | Aim                                                                                                                               |
+====================================================+===================================================================================================================================+
| `pre-commit.ci <https://pre-commit.ci/>`__         | Run pre-commit (as defined in `.pre-commit-config.yaml <https://github.com/ltelab/disdrodb/blob/main/.pre-commit-config.yaml>`_ ) |
+----------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------+
| `CodeBeat <https://codebeat.co/>`__                | Automated code review and analysis tools                                                                                          |
+----------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------+
| `CodeScene <https://codescene.com/>`__             | Automated code review and analysis tools                                                                                          |
+----------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------+
| `CodeFactor <https://www.codefactor.io/>`__        | Automated code review and analysis tools                                                                                          |
+----------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------+
| `Codacy <https://www.codacy.com/>`__               | Automated code review and analysis tools                                                                                          |
+----------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------+


5. Code testing with pytest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DISDRODB tests are written using the third-party `pytest <https://docs.pytest.org>`_ package. Every code change must be tested!

The tests are organized within the ``/disdrodb/tests`` directory and are structured to comprehensively assess various aspects of the code.

These tests are integral to the development process and are automatically triggered on GitHub upon any new commits or updates to a Pull Request.
The Continuous Integration (CI) on GitHub runs tests and analyzes code coverage using multiple versions of Python,
multiple operating systems, and multiple versions of dependency libraries. This is done to ensure that the code works in a variety of environments.

The following tools are used:

+-----------------------------------------------------------------------------------------------+------------------------------------------------------------------+
|  Tool                                                                                         | Aim                                                              |
+===============================================================================================+==================================================================+
| `Pytest  <https://docs.pytest.org>`__                                                         | Execute unit tests and functional tests                          |
+-----------------------------------------------------------------------------------------------+------------------------------------------------------------------+
| `Coverage <https://coverage.readthedocs.io/>`__                                               | Measure the code coverage of the project's unit tests            |
+-----------------------------------------------------------------------------------------------+------------------------------------------------------------------+
| `CodeCov    <https://about.codecov.io/>`__                                                    | Uses Coverage to track and analyze code coverage over time.      |
+-----------------------------------------------------------------------------------------------+------------------------------------------------------------------+
| `Coveralls    <https://coveralls.io/>`__                                                      | Uses Coverage to track and analyze code coverage over time.      |
+-----------------------------------------------------------------------------------------------+------------------------------------------------------------------+


For contributors interested in running the tests locally:

1. Ensure you have the `development environment <https://disdrodb.readthedocs.io/en/latest/installation.html#installation-for-contributors>`__ correctly set up.
2. Navigate to the disdrodb root directory.
3. Execute the following command to run the entire test suite:

.. code-block:: bash

	pytest

For more focused testing or during specific feature development, you may run subsets of tests.
This can be done by specifying either a sub-directory or a particular test module.

Run tests in a specific sub-directory:

.. code-block:: bash

    pytest disdrodb/tests/<test_subdirectory>/

Run a particular test module:

.. code-block:: bash

    pytest disdrodb/tests/<test_subdirectory>/test_<module_name>.py

These options provide flexibility, allowing you to efficiently target and validate specific components of the disdrodb software.

.. note::
   Each test module must be prefixed with ``test_`` to be recognized and selected by pytest.
   This naming pattern is a standard convention in pytest and helps in the automatic discovery of test files.


6. Push your changes to your fork repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

During this process, pre-commit hooks will be run. Your commit will be
allowed only if quality requirements are fulfilled.

If you encounter errors, you can attempt to fix the formatting errors with the following command:

.. code-block:: bash

   pre-commit run --all-files


7. Create a new Pull Request in GitHub.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once your code has been uploaded into your DISDRODB fork, you can create a Pull Request (PR) to the DISDRODB main branch.

Recommendation for the Pull Requests:

-  Please fill it out accurately the Pull Request template.
-  It is perfectly fine to make many small commits as you work on a Pull Request. GitHub will automatically squash all the commits before merging the Pull Request.
-  If adding a new feature:

   -  Provide a convincing reason to add the new feature. Ideally, propose your idea through a `Feature Request Issue <https://github.com/ltelab/disdrodb/issues/new/choose>`__ and obtain approval before starting work on it. Alternatively, you can present your ideas in the `GitHub Discussions <https://github.com/ltelab/disdrodb/discussions>`__ or in the `DISDRODB Slack Workspace <https://join.slack.com/t/disdrodbworkspace/shared_invite/zt-25l4mvgo7-cfBdXalzlWGd4Pt7H~FqoA>`__.
   -  Implement unit tests to verify the functionality of the new feature. This ensures that your addition works as intended and maintains the quality of the codebase.

-  If fixing bug:

   -  Provide a comprehensive description of the bug within your Pull Request. This aids reviewers in understanding the issue and the impact of your fix.
   -  If your Pull Request addresses a specific issue, add ``(fix #xxxx)`` in your PR title to link the PR to the issue and enhance the clarity of release logs. For example, the title of a PR fixing issue ``#3899`` would be ``<your PR title> (fix #3899)``.
   -  If applicable, ensure that your fix includes appropriate tests. Adding tests for your bug fix helps prevent future regressions and maintains the stability of the software.


Code review checklist
---------------------

-  Once your Pull Request is ready, ask the maintainers to review your code.
-  When you are done with the changes suggested by the reviewers, do another  self review of the code and write a comment to notify the reviewer,
   that the Pull Request is ready for another iteration.
-  Resolve all the review comments, making sure they are all addressed before another review iteration.
-  If you are not going to follow a code review recommendations, please add a comment explaining why you think the reviewer suggestion is not relevant.
-  Avoid writing comment like “done” of “fixed” on each code review comment.
   Reviewers assume you will do all suggested changes, unless you have a reason not to do some of them.


Credits
-------

Thank you to all the people who have already contributed to DISDRODB repository!

If you have contributed data and/or code to disdrodb, add your name to the `AUTHORS.md <https://github.com/ltelab/disdrodb/blob/main/AUTHORS.md>`__ file.
