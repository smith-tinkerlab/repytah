
# Contributing code

This project is a community effort.

## How to contribute

The preferred way to contribute to `repytah` is to fork the [main repository](https://github.com/smith-tinkerlab/repytah) on GitHub:

1. Fork the [project repository](https://github.com/smith-tinkerlab/repytah):
   click on the 'Fork' button near the top of the page. This creates
   a copy of the code under your account on the GitHub server.

2. Clone this copy to your local disk:

   ```bash
   git clone git@github.com:smith-tinkerlab/repytah.git
   cd repytah 
   ```

3. (Optional) If you want to remove any previously installed repytah:

   ```bash
   pip uninstall repytah
   ```

4. Install your local copy with testing dependencies:

   ```bash
   pip install -e repytah[test]
   ```

5. Create a branch to hold your changes:

   ```bash
   git checkout -b my-feature
   ```

   and start making changes. Never work in the `main` branch!

6. Work on this copy on your computer using Git to do the version
   control. When you're done editing, do:

   ```bash
   git add modified_files
   git commit
   ```

   to record your changes in Git, then push them to GitHub with:

   ```bash
   git push -u origin my-feature
   ```

7. Finally, go to the web page of the your fork of the repytah repo, and click 'Pull request' to send your changes to the maintainers for review. This will send an email to the committers. (If any of the above seems like magic to you, then look up the [Git documentation](http://git-scm.com/documentation) on the web.)

It is recommended to check that your contribution complies with the
following rules before submitting a pull request:

- All public methods should have informative docstrings with sample
   usage presented.

You can also check for common programming errors with the following
tools:

- Code with good test coverage (at least 80%), check with:

   ```bash
   pytest
   ```

- No PEP8 warnings, check with:

   ```bash
   pip install pep8
   pep8 path/to/module.py
   ```

- AutoPEP8 can help you fix some of the easy redundant errors:

   ```bash
   pip install autopep8
   autopep8 path/to/pep8.py
   ```

## Filing bugs

We use Github issues to track all bugs and feature requests; feel free to
open an issue if you have found a bug or wish to see a feature implemented.

It is recommended to check that your issue complies with the
following rules before submitting:

- Verify that your issue is not being currently addressed by other
   [issues](https://github.com/smith-tinkerlab/repytah/issues)
   or [pull requests](https://github.com/smith-tinkerlab/repytah/pulls).

- Please ensure all code snippets and error messages are formatted in
   appropriate code blocks.
   See [Creating and highlighting code blocks](https://help.github.com/articles/creating-and-highlighting-code-blocks).

- Please include your operating system type and version number, as well
   as your Python, numpy, and scipy versions. This information
   can be found by running the following code snippet:

  ```python
  import platform; print(platform.platform())
  import sys; print("Python", sys.version)
  import numpy; print("NumPy", numpy.__version__)
  import scipy; print("SciPy", scipy.__version__)
  import repytah; print("repytah", repytah.__version__)
  ```

## Documentation

You can edit the documentation using any text editor and then generate
the HTML output by typing ``make html`` from the docs/ directory.
The resulting HTML files will be placed in _build/html/ and are viewable
in a web browser. See the README file in the doc/ directory for more information.

For building the documentation, you will need
[sphinx](https://www.sphinx-doc.org/) and
[matplotlib](https://matplotlib.org/stable/index.html).

## Note

This document was borrowed from [scikit-learn](http://scikit-learn.org/) and [librosa](https://github.com/librosa/librosa).
