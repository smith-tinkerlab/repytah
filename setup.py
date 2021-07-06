from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='repytah',  
    version='0.0.0a0',
    description='repytah: Python package for building Aligned Hierarchies for music-based data streams.',  # Optional
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/smith-tinkerlab/ah',
    author='Katherine M. Kinnaird, Smith Tinkerlab',
    author_email='kkinnaird@smith.edu',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
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
        'pandas',
        'matplotlib'
    ],
    extras_require={
        'docs': [
            'sphinx != 1.3.1',
            'sphinx_rtd_theme==0.5.*',
            'sphinx-multiversion >= 0.2.3',
            'sphinx-gallery >= 0.7',
            'spinxcontrib-svg2pdfconverter',
            'presets'
                ],
        'tests': ['pytest']
    },
    project_urls={ 
        'Bug Reports': 'https://github.com/smith-tinkerlab/ah/issues',
        'Source': 'https://github.com/smith-tinkerlab/ah',
    },
    package_data={'repytah': ['input.csv']},
    include_package_data=True,
    license_file = 'LICENSE.md',
    license='MIT'
)