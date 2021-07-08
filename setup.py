from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='repytah',
    version='0.1.0dev',
    description='Python package for building Aligned Hierarchies for music-based data streams',  # Optional
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/smith-tinkerlab/repytah',
    author='Smith Tinker Lab, Katherine M. Kinnaird (PI)',
    author_email='tinkerlab@smith.edu',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
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
        'pandas' >= '1.0.0',
        'matplotlib >= 3.3.0'
    ],
    extras_require={
        'docs': [
            'sphinx != 1.3.1',
            'nbsphinx',
            'sphinx_rtd_theme==0.5.*',
            'sphinx-multiversion >= 0.2.3',
            'sphinx-gallery >= 0.7',
            'spinxcontrib-svg2pdfconverter',
            'presets'
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