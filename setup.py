from setuptools import setup

with open("README.md", "r") as fh:
    readme = fh.read()

setup(name='WRFdownscalingML',
    version='0.0.1',
    url='https://https://github.com/rjsampa/WRFdownscalingML',
    license='MIT License',
    author='Rafael Sampaio',
    long_description=readme,
    long_description_content_type="text/markdown",
    author_email='samprafael@gmail.com',
    keywords='Pacote',
    description=u'Framework para downscaling de outputs do WRF',
    packages=['WRFdownscalingML'],
    install_requires=['numpy','pandas','matplotlib','netCDF4'],)
