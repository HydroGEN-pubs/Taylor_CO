# Taylor_CO
Scripts and workflow for the Taylor catchment (CO) Parflow-CLM runs machine learning emulator

This repository contains the code required to reproduce the results in Leonarduzzi et al. (submitted to Frontiers in Water)

1) `Taylor_parflow` allows to create inputs, create forcing scenarios (increasing temperature and decreasing precipitation), and run Parflow-CLM for the catchment Taylor in Colorado.  

2) `Taylor_ml` trains and evaluates 2D and 3D Convolutional Neural Network on the PF-CLM simulations (assumes the runs in 1) were created).  

Instructions are provided in a `README.md` each folder.

The pre-requisites are:
- you have installed [ParFlow-CLM](https://github.com/parflow/parflow)
- you have installed [hydrogen package] (REF?)
