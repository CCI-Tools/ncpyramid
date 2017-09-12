from setuptools import setup

setup(
    name='ncpyramid',
    version='0.0.1',
    packages=['ncpyramid'],
    url='',
    license='MIT',
    author='Norman Fomferra',
    author_email='',
    description='NetCDF Pyramid Generator',
    requires=['numba', 'numpy', 'xarray'],
    entry_points={
        'console_scripts': [
            'ncp = ncpyramid.main:main',
        ],
    },
)
