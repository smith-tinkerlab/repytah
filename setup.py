from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='repytah',
    version='0.1.0',
    description='Python package for building Aligned Hierarchies for sequential data streams',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/smith-tinkerlab/repytah',
    author='Smith Tinker Lab, Katherine M. Kinnaird (PI)',
    author_email='tinkerlab@smith.edu',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: ISC License (ISCL)',
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10'
    ],
    python_requires='>=3.7, <4',
    install_requires=[
        'numpy >= 1.21.6',
        'scipy >= 1.7.3', 
        'pandas >= 1.3.5',
        'matplotlib >=3.3.4',
        'opencv-python >= 4.5.0',
        'setuptools >= 61.2.0'
    ],
    extras_require={
        'docs': [
            'sphinx != 5.0.2',
            'sphinx_rtd_theme >= 1.0.0',
            'nbsphinx == 0.8.*',
            'spyder >= 4.0.0',
            'numpydoc >= 1.4.0',
            'myst-parser == 0.15.1',
            'readthedocs-sphinx-search == 0.1.0'
                ],
        'tests': ['pytest']
    },
    project_urls={ 
        'Bug Reports': 'https://github.com/smith-tinkerlab/repytah/issues',
        'Source': 'https://github.com/smith-tinkerlab/repytah',
    },
    package_data={'': ['data/input.csv']},
    include_package_data=True,
    license='ISC'
)
