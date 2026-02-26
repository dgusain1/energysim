from setuptools import setup

import os

def readme():
    readme_path = os.path.join(os.path.dirname(__file__), '..', 'README.rst')
    if os.path.exists(readme_path):
        with open(readme_path, encoding='utf-8') as f:
            return f.read()
    return 'Python package for cosimulation of multi-energy systems'


setup(
      name='energysim',
      version='2.1.9.1',
      description='Python package for cosimulation of multi-energy systems',
      long_description=readme(),
      long_description_content_type="text/x-rst",
      url='https://github.com/dgusain1/energysim',
      author='Digvijay Gusain',
      author_email='digvijay.gusain29@gmail.com',
      license='MIT',
      classifiers=[
              'License :: OSI Approved :: MIT License',
              'Programming Language :: Python :: 3',
              'Programming Language :: Python :: 3.8',
              'Programming Language :: Python :: 3.9',
              'Programming Language :: Python :: 3.10',
              'Programming Language :: Python :: 3.11',
              'Programming Language :: Python :: 3.12',
              'Programming Language :: Python :: 3.13'],
      packages=['energysim'],
      include_package_data=True,
      install_requires=['numpy', 'pandas', 'tqdm', 'tables', 'networkx'],
      extras_require={
          'fmu': ['fmpy>=0.3'],
          'powerflow': ['pandapower'],
          'pypsa': ['pypsa'],
          'plotting': ['matplotlib'],
          'all': ['fmpy>=0.3', 'pandapower', 'pypsa', 'matplotlib'],
      },
      python_requires='>=3.8',
      )
