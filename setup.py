from setuptools import setup

def readme():
    with open('README.md', encoding='utf-8') as f:
        README = f.read()
    return README


setup(
      name='energysim',
      version='1.0.1',
      description='Python package for cosimulation of multi-energy systems',
      long_description=readme(),
      long_description_content_type="text/markdown",
      url='https://github.com/dgusain1/energysim',
      author='Digvijay Gusain',
      author_email='d.gusain@tudelft.nl',
      license='MIT',
      classifiers=[
              'License :: OSI Approved :: MIT License',
              'Programming Language :: Python :: 3',
              'Programming Language :: Python :: 3.6'],
      packages=['energysim'],
      include_package_data=True,
      install_requires=['numpy', 'pandas', 'tqdm', 'fmpy', 'pandapower', 'pypsa', 'networkx', 'matplotlib']     
      )
