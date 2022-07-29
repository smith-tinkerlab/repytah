# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from pathlib import Path
import sphinx

# This enables builds from outside the docs directory
srcpath = os.path.abspath(Path(os.path.dirname(__file__)) / '..')
sys.path.insert(0, srcpath)
# for modules to document with autodoc in another directory
sys.path.insert(0, os.path.abspath('../repytah'))

# -- Project information -----------------------------------------------------

project = u"repytah"
copyright = u"2021, repytah development team"
author = u"Katherine M Kinnaird, Eleanor Donaher, Lizette Carpenter, Jordan Moody, Denise Nava, Sasha Yeutseyeva, Chenhui Jia, Marium Tapal, Betty Wang, Thu Tran, Zoie Zhao"


# -- General configuration ---------------------------------------------------

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.

from importlib.machinery import SourceFileLoader

repytah_version = SourceFileLoader(
    "repytah.version", os.path.abspath(Path(srcpath) / 'repytah' / 'version.py')
).load_module()

# The short X.Y version.
version = repytah_version.version
# The full version, including alpha/beta/rc tags.
release = repytah_version.version

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.
import sphinx_rtd_theme
extensions = ["sphinx.ext.autodoc", 
              "sphinx.ext.coverage",
              "sphinx.ext.napoleon", 
              "nbsphinx", 
              "sphinx_rtd_theme",
              "sphinx.ext.imgconverter"
]

## Include Python objects as they appear in source files
autodoc_member_order = 'bysource'
autodoc_default_options = {
    'members': True
}

source_suffix = ".rst"


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# List of imported modules
autodoc_mock_imports = ["numpy", "scipy", "matplotlib", "pandas", "cv2"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_css_files = [
    "custom.css"
]

html_logo = "../branding/repytah_logo.svg"

html_theme_options = {
    "logo_only": True,
}
