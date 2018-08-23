from setuptools import setup
from magsqs import __version__, __email__, __author__

setup(
    name='magsqs',
    version=__version__,
    author=__author__,
    author_email=__email__,
    description='Generator for non collinear and collinear magnetic moments',
    license='GPLv3',
    keyword='ncl collinear material science magnetism magnetic',
    url='https://github.com/dnoeger/magsqs',
    py_modules=['magsqs'],
    entry_points={
        'console_scripts': ['magsqs=magsqs:main']
    },
    install_requires = [
        'pymatgen',
        'sympy',
        'numpy',
        'matplotlib',
        'docopt'
    ]
)
