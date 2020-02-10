# POMDP_Policy-Iteration
Uses Finite State Controller (FSC) architecture to solve the infinite horizon reward maximization problem for POMDPs.

This solver is built of top of the [AIToolbox](https://github.com/Svalorzen/AI-Toolbox) by [Svalrozen](https://github.com/Svalorzen)

```
  cd AI-Toolbox-master
  mkdir build
  cd build/
  cmake ..
  make
```

``cmake`` can be called with a series of flags in order to customize the output, f building everything is not desirable. 
The following flags are available:

```
  CMAKE_BUILD_TYPE # Defines the build type
  MAKE_ALL         # Builds all there is to build in the project
  MAKE_LIB         # Builds the whole core C++ libraries (MDP, POMDP, etc..)
  MAKE_MDP         # Builds only the core C++ MDP library
  MAKE_FMDP        # Builds only the core C++ Factored/Multi-Agent and MDP libraries
  MAKE_POMDP       # Builds only the core C++ POMDP and MDP libraries
  MAKE_TESTS       # Builds the library's tests for the compiled core libraries
  MAKE_EXAMPLES    # Builds the library's examples using the compiled core libraries
  MAKE_PYTHON      # Builds Python bindings for the compiled core libraries
  PYTHON_VERSION   # Selects the Python version you want (2 or 3). If not
                   # specified, we try to guess based on your default interpreter.
```

These flags can be combined as needed. For example:

```
  # Will build MDP and MDP Python 3 bindings
  cmake -DCMAKE_BUILD_TYPE=Debug -DMAKE_MDP=1 -DMAKE_POMDP=1 -DMAKE_PYTHON=1 -DPYTHON_VERSION=3 ..
```

For the sake of the POMDP solver, it is necessary to set ``MAKE_MDP`` and ``MAKE_POMDP`` to True
  
To solve any problem:

```
   cd src/problem-name
   script.sh [1] [2] [3] [4]
```

   * ``[1]`` problem-name
   * ``[2]`` Re-make AIToolbox (0/1)
   * ``[3]`` Recompile problem file (0/1)
   * ``[4]`` Run for multiple epochs (0/1)
