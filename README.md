[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://obeznosov-lanl.github.io/hFlux/) 

# hFlux

hFlux -- lightweight toolkit for tokamak simulation code diagnostics.

<img src="https://github.com/obeznosov-LANL/hFlux/blob/main/docs/src/assets/readme.png" width="100%" title="Poincare plots">

# Key features
* Poloidal flux function.
* Continuous, smooth and divergence free magnetic field reconstructionk.
* Built-in divergence cleaning, input fields do not require to be divergence free.
* Second order differentials of flux function (for example $\nabla B$, $\nabla \times B$).
* Calculation of magnetic axis.
* Safety-factor for arbitrary flux surfaces.
* Field line calculations, supporting Poincare plot data output.
* C/C++ interface, Python and Julia Wrappers.

## Requirements

- **CMake ≥ 3.10**  
- A C++ compiler supporting **C++23**

You can verify your CMake version by running `cmake --version`. Make sure your compiler (e.g., GCC, Clang, MSVC) supports C++23.


## Build instructions
```console
~$ cmake -Bbuild
~$ cd build; make
```
Now you can access library through Julia interface. Coherent C, C++ and python interfaces are in the works


# Release
O4754 hFlux was approved for Open-Source Assertion
