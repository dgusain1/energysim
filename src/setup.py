from setuptools import setup

def readme():
    with open('README.rst', encoding='utf-8') as f:
        README = f.read()
    return README


setup(
      name='energysim',
      version='2.1.8',
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
              'Programming Language :: Python :: 3.6'],
      packages=['energysim'],
      include_package_data=True,
      install_requires=['numpy', 'pandas', 'tqdm',
                        'fmpy==0.2.14', 'pandapower',
                        'pypsa', 'networkx', 'matplotlib',
                        'tables', 'h5py']
      )
