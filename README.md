# energysim

![energysim](https://i.imgur.com/9Z73YNM.png)

**energysim** (previously, FMUWorld) is a python-based tool that was developed specifically to simplify energy system based cosimulations. Currently, it allows users to combine:

  - Dynamic Models packaged as Functional Mockup Units (FMUs)
  - Pandapower Networks packaged as pickle files
  - PyPSA models (BETA) packaged as Excel workbook
  - csv data files

## Installation
energysim can be installed using PyPI index command:
```
pip install energysim
```

It uses the following packages to work:
  - [FMPy](https://github.com/CATIA-Systems/FMPy)
  - [Numpy](https://pypi.org/project/numpy/)
  - [Pandas](https://pypi.org/project/pandas/)
  - [Matplotlib](https://pypi.org/project/matplotlib/)
  - [Pandapower](https://pypi.org/project/pandapower/)
  - [PyPSA](https://pypi.org/project/pypsa/)

## Usage
energysim simplifies cosimulation setup by conceptualising the setup in an intuitive manner. The whole system is defined within a `World()` canvas. Within `World()`, users can specify different simulators by using commands such as `add_fmu()`, `add_powerflow()`, `add_csv()`, `add_signal()`. Apart from this, users can also specify initialisation options using the `options()` command, connections between simulators using the `add_connections()` command. Finally, the system can be simulated using the `simulate()` command. This method provides a results dataframe which can then be used for analysis of the multi-energy system simulation. Apart from simple simulations, **energysim** alo incudes a method to perform sensitivity analysis on selected parameters within the cosimulation environment.

A brief example is shown below:

```
#import package
from FMUWorld import World
#create world instance
my_world = World(stop_time = 1000, 
                logging = True, 
                exchange = 2,
                interpolate_results = False)

simLoc1 = os.path.join(working_dir,chp+'.fmu') #fmu location
simLoc2 = os.path.join(working_dir,elec+'.p') #pandapower network location

#add fmu
my_world.add_fmu(name = 'chp', 
                loc = simLoc1, 
                step_size = 2, 
                outputs = ['rampeQfuel.y.signal',
                            'Alternateur.Welec'])
my_world.add_powerflow(name = 'elec', 
                loc = simLoc2, 
                step_size = 1e-3, 
                inputs = ['gen1.P']
                outputs = ['gen1.f',
                           'gen2.f',
                           'gen3.f'])

#define connections
connections = 
    {'chp.Alternateur.Welec':'elec.gen1.P'}

#add connections
my_world.add_connections(connections)

#simulate
results = my_world.simulate()
```

More information is provided on the documentation page.

## Citing
Please cite the following paper if you use **energysim**:
Gusain, D, Cvetković, M & Palensky, P 2019, Energy flexibility analysis using FMUWorld. in 2019 IEEE Milan PowerTech., 8810433, IEEE, 2019 IEEE Milan PowerTech, PowerTech 2019, Milan, Italy, 23/06/19. https://doi.org/10.1109/PTC.2019.8810433
