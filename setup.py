from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='repytah',
    version='0.1.0dev',
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
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.6, <4',
    install_requires=[
        'numpy >= 1.15.0',
        'scipy >= 1.0.0', 
        'pandas >= 1.0.0',
        'matplotlib >= 3.3.0'
    ],
    extras_require={
        'docs': [
            'sphinx != 1.3.1',
            'sphinx_rtd_theme >= 0.3.1',
            'nbsphinx == 0.8.*',
            'spyder >= 4.0.0',
            'numpydoc >= 0.9.0',
            'myst-parser == 0.15.1',
            'readthedocs-sphinx-search == 0.1.0'
                ],
        'tests': ['pytest']
    },
    project_urls={ 
        'Bug Reports': 'https://github.com/smith-tinkerlab/repytah/issues',
        'Source': 'https://github.com/smith-tinkerlab/repytah',
    },
    package_data={'repytah': ['input.csv']},
    include_package_data=True,
    license='MIT'
)