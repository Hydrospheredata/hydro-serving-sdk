# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
from recommonmark.transform import AutoStructify

# sys.path.insert(0, os.path.abspath('.'))

project = 'hydrosdk'
copyright = '2020, Hydrosphere.io'
author = 'Hydrosphere.io'

# The full version, including alpha/beta/rc tags
release = '2.3.0'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'sphinx.ext.intersphinx',
              'recommonmark',
              'sphinx.ext.coverage',
              'sphinx_rtd_theme'
              ]

# Tell Sphinx to use recommonmark to parse md
source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/{.major}'.format(
        sys.version_info), None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None)
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

html_theme = "sphinx_rtd_theme"
html_logo = '_static/logo_white.svg'
html_favicon = '_static/favicon.svg'

html_theme_options = {
    "github_url": "https://github.com/Hydrospheredata",
    "twitter_url": "https://twitter.com/Hydrospheredata",
    "external_links": [
        {"name": "gitter", "url": "https://gitter.im/Hydrospheredata/hydro-serving"},
    ]
}

github_url = "https://github.com/Hydrospheredata/hydro-serving-sdk"

# TODO
# google_analytics_id

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


def setup(app):
    # app.add_config_value('recommonmark_config', {
    #         # 'url_resolver': lambda url: "/Users/ygavrilin/Projects/hydro-docs/build/html/" + url,
    #         'auto_toc_tree_section': 'Contents',
    #         'enable_eval_rst': True,
    #         }, True)
    app.add_transform(AutoStructify)
