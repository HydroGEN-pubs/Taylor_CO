import sys
import shutil
import calendar
import pathlib
import os

"""
Reference: github link to put here to the repository

This script runs parflow-clm for one water year.
It can be run as
python Taylor_clm.py year_run root_path
where:  YEAR_RUN is the water year for which you want to run (e.g., 1988)
		ROOTH_PATH is the path to where you placed Taylor_parflow (e.g., /home/user/Desktop/Taylor_CO/Taylor_parflow/)
--> python Taylor_clm.py 1988 /home/user/Desktop/Taylor_CO/Taylor_parflow/

It assumes
- you already created the forcing for the water year 
(either historical, subsetting from UCRB with subset_UCRB.py or modified with PF_generate_forcing_scenarios.py)
	--> placed in the current path in a subfolder called NLDAS
- all static inputs are available (are provided in the github repository or created with subset_UCRB.py) in ../../static_Taylor
- the initial pressure file initial_pressure.pfb is available in the current folder (can also be generated with subset_UCRB.py)
- you're running for one water year, hourly timesteps
- drv_clmin.dat is available in the current folder, corrected (within subset_UCRB.py)
- you followed the folder structure reccomended on github, README.md (otherwise change all the paths to point in the proper folders/files)
- you will be running with 9 processors (otherwise change P,Q, and R) Note: this must be consistent with what you had creating the forcing files, if it isn't,
	set distribute_forcing to True
"""

##water years for which to run simulation
year_run = int(sys.argv[1])

#path to Taylor_parflow
root_directory = sys.argv[2]

## Processors topology (needs to be consistent with what was done when creating the forcing files)
P=3
Q=3
R=1
#if this is not 3-3-1 or is different to what you used when creating the forcing files, set this to True and it will re-distribute
distribute_forcing = False


##path where the inputs are saved:
path_inputs = f'{root_directory}/static_Taylor/'

## folder where the forcing data is saved
met_path = f'NLDAS/'

tcl_precision = 17

if calendar.isleap(year_run):
	no_day = 366
else:
	no_day=365

#
# Import the ParFlow TCL package
#
from parflow import Run
Taylor_clm = Run(f'Taylor_{year_run}', __file__)

Taylor_clm.FileVersion = 4

#Processors topology (determines the distribution of forcing files, change this if not running on 3*3*1 = 9 processors)
Taylor_clm.Process.Topology.P = P
Taylor_clm.Process.Topology.Q = Q
Taylor_clm.Process.Topology.R = R

nproc = Taylor_clm.Process.Topology.P*Taylor_clm.Process.Topology.Q*Taylor_clm.Process.Topology.R

path_slope_x = f'{path_inputs}slope_x.pfb'
path_slope_y = f'{path_inputs}slope_y.pfb'
path_drv_vegm = f'{path_inputs}drv_vegm_v2.Taylor.dat'
path_drv_vegp = f'{path_inputs}drv_vegp.dat'
path_indicator = f'{path_inputs}Taylor.IndicatorFile_v2.pfb'
path_pfsol = f'{path_inputs}Taylor.pfsol'

ip = 'initial_pressure.pfb'
indicator = f'Taylor.IndicatorFile_v2.pfb'
solidfile = "Taylor.pfsol"

current_path = os.getcwd()

shutil.copy(f'{path_slope_x}', current_path)
shutil.copy(f'{path_slope_y}', current_path)
shutil.copy(f'{path_drv_vegm}', current_path)
shutil.copy(f'{path_drv_vegp}', current_path)
shutil.copy(f'{path_pfsol}', current_path)
shutil.copy(f'{path_indicator}', current_path)

istep = 0
clmstep = istep+1

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------
Taylor_clm.ComputationalGrid.Lower.X = 0.0
Taylor_clm.ComputationalGrid.Lower.Y = 0.0
Taylor_clm.ComputationalGrid.Lower.Z = 0.0

Taylor_clm.ComputationalGrid.NX = 46
Taylor_clm.ComputationalGrid.NY = 48
Taylor_clm.ComputationalGrid.NZ = 5

Taylor_clm.ComputationalGrid.DX = 1000.0
Taylor_clm.ComputationalGrid.DY = 1000.0
Taylor_clm.ComputationalGrid.DZ = 200.0

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------
Taylor_clm.GeomInput.Names = 'domaininput soilinput indi_input'
Taylor_clm.GeomInput.Names = 'domaininput indi_input'

Taylor_clm.GeomInput.domaininput.GeomName = 'domain'

Taylor_clm.GeomInput.domaininput.InputType = 'SolidFile'
Taylor_clm.GeomInput.domaininput.GeomNames = 'domain'
Taylor_clm.GeomInput.domaininput.FileName = solidfile

#pfset Geom.domain.Patches             "land top  bottom"
Taylor_clm.Geom.domain.Patches = 'top bottom side'


#-----------------------------------------------------------------------------
# Indicator Geometry Input
#-----------------------------------------------------------------------------

Taylor_clm.GeomInput.indi_input.InputType = 'IndicatorField'
Taylor_clm.GeomInput.indi_input.GeomNames = 's1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 g1 g2 g3 g4 g5 g6 g7 g8 b1 b2'
Taylor_clm.Geom.indi_input.FileName = indicator

Taylor_clm.GeomInput.s1.Value = 1
Taylor_clm.GeomInput.s2.Value = 2
Taylor_clm.GeomInput.s3.Value = 3
Taylor_clm.GeomInput.s4.Value = 4
Taylor_clm.GeomInput.s5.Value = 5
Taylor_clm.GeomInput.s6.Value = 6
Taylor_clm.GeomInput.s7.Value = 7
Taylor_clm.GeomInput.s8.Value = 8
Taylor_clm.GeomInput.s9.Value = 9
Taylor_clm.GeomInput.s10.Value = 10
Taylor_clm.GeomInput.s11.Value = 11
Taylor_clm.GeomInput.s12.Value = 12
Taylor_clm.GeomInput.s13.Value = 13

Taylor_clm.GeomInput.g1.Value = 21
Taylor_clm.GeomInput.g2.Value = 22
Taylor_clm.GeomInput.g3.Value = 23
Taylor_clm.GeomInput.g4.Value = 24
Taylor_clm.GeomInput.g5.Value = 25
Taylor_clm.GeomInput.g6.Value = 26
Taylor_clm.GeomInput.g7.Value = 27
Taylor_clm.GeomInput.g8.Value = 28
Taylor_clm.GeomInput.b1.Value = 19
Taylor_clm.GeomInput.b2.Value = 20

#--------------------------------------------
# variable dz assignments
#------------------------------------------
Taylor_clm.Solver.Nonlinear.VariableDz = True
Taylor_clm.dzScale.GeomNames = 'domain'
Taylor_clm.dzScale.Type = 'nzList'
Taylor_clm.dzScale.nzListNumber = 5

# 5 layers, starts at 0 for the bottom to 5 at the top
# note this is opposite Noah/WRF
# layers are 0.1 m, 0.3 m, 0.6 m, 1.0 m, 100 m
Taylor_clm.Cell._0.dzScale.Value = 0.5
#pfset Cell.0.dzScale.Value 0.001
# 100 m * .01 = 1m 
Taylor_clm.Cell._1.dzScale.Value = 0.005
# 100 m * .006 = 0.6 m 
Taylor_clm.Cell._2.dzScale.Value = 0.003
# 100 m * 0.003 = 0.3 m 
Taylor_clm.Cell._3.dzScale.Value = 0.0015
# 100 m * 0.001 = 0.1m = 10 cm which is default top Noah layer
Taylor_clm.Cell._4.dzScale.Value = 0.0005

#-----------------------------------------------------------------------------
# Permeability (values in m/hr)
#-----------------------------------------------------------------------------
Taylor_clm.Geom.Perm.Names = 'domain s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 g1 g2 g3 g4 g5 g6 g7 g8 b1 b2'

# Values in m/hour

Taylor_clm.Geom.domain.Perm.Type = 'Constant'
Taylor_clm.Geom.domain.Perm.Value = 0.02

Taylor_clm.Geom.s1.Perm.Type = 'Constant'
Taylor_clm.Geom.s1.Perm.Value = 0.269022595

Taylor_clm.Geom.s2.Perm.Type = 'Constant'
Taylor_clm.Geom.s2.Perm.Value = 0.043630356

Taylor_clm.Geom.s3.Perm.Type = 'Constant'
Taylor_clm.Geom.s3.Perm.Value = 0.015841225

Taylor_clm.Geom.s4.Perm.Type = 'Constant'
Taylor_clm.Geom.s4.Perm.Value = 0.007582087

Taylor_clm.Geom.s5.Perm.Type = 'Constant'
Taylor_clm.Geom.s5.Perm.Value = 0.01818816

Taylor_clm.Geom.s6.Perm.Type = 'Constant'
Taylor_clm.Geom.s6.Perm.Value = 0.005009435

Taylor_clm.Geom.s7.Perm.Type = 'Constant'
Taylor_clm.Geom.s7.Perm.Value = 0.005492736

Taylor_clm.Geom.s8.Perm.Type = 'Constant'
Taylor_clm.Geom.s8.Perm.Value = 0.004675077

Taylor_clm.Geom.s9.Perm.Type = 'Constant'
Taylor_clm.Geom.s9.Perm.Value = 0.003386794

Taylor_clm.Geom.s10.Perm.Type = 'Constant'
Taylor_clm.Geom.s10.Perm.Value = 0.004783973

Taylor_clm.Geom.s11.Perm.Type = 'Constant'
Taylor_clm.Geom.s11.Perm.Value = 0.003979136

Taylor_clm.Geom.s12.Perm.Type = 'Constant'
Taylor_clm.Geom.s12.Perm.Value = 0.006162952

Taylor_clm.Geom.s13.Perm.Type = 'Constant'
Taylor_clm.Geom.s13.Perm.Value = 0.005009435



Taylor_clm.Geom.b1.Perm.Type = 'Constant'
Taylor_clm.Geom.b1.Perm.Value = 0.005

Taylor_clm.Geom.b2.Perm.Type = 'Constant'
Taylor_clm.Geom.b2.Perm.Value = 0.01

Taylor_clm.Geom.g1.Perm.Type = 'Constant'
Taylor_clm.Geom.g1.Perm.Value = 0.02

Taylor_clm.Geom.g2.Perm.Type = 'Constant'
Taylor_clm.Geom.g2.Perm.Value = 0.03

Taylor_clm.Geom.g3.Perm.Type = 'Constant'
Taylor_clm.Geom.g3.Perm.Value = 0.04

Taylor_clm.Geom.g4.Perm.Type = 'Constant'
Taylor_clm.Geom.g4.Perm.Value = 0.05

Taylor_clm.Geom.g5.Perm.Type = 'Constant'
Taylor_clm.Geom.g5.Perm.Value = 0.06

Taylor_clm.Geom.g6.Perm.Type = 'Constant'
Taylor_clm.Geom.g6.Perm.Value = 0.08

Taylor_clm.Geom.g7.Perm.Type = 'Constant'
Taylor_clm.Geom.g7.Perm.Value = 0.1

Taylor_clm.Geom.g8.Perm.Type = 'Constant'
Taylor_clm.Geom.g8.Perm.Value = 0.2


Taylor_clm.Perm.TensorType = 'TensorByGeom'
Taylor_clm.Geom.Perm.TensorByGeom.Names = 'domain'
Taylor_clm.Geom.domain.Perm.TensorValX = 1.0
Taylor_clm.Geom.domain.Perm.TensorValY = 1.0
Taylor_clm.Geom.domain.Perm.TensorValZ = 1.0


#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

Taylor_clm.SpecificStorage.Type = 'Constant'
Taylor_clm.SpecificStorage.GeomNames = 'domain'
Taylor_clm.Geom.domain.SpecificStorage.Value = 1.0e-4

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

Taylor_clm.Phase.Names = 'water'

Taylor_clm.Phase.water.Density.Type = 'Constant'
Taylor_clm.Phase.water.Density.Value = 1.0

Taylor_clm.Phase.water.Viscosity.Type = 'Constant'
Taylor_clm.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------

Taylor_clm.Contaminants.Names = ''

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------

Taylor_clm.Geom.Retardation.GeomNames = ''

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

Taylor_clm.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

#
Taylor_clm.TimingInfo.BaseUnit = 1.0
Taylor_clm.TimingInfo.StartCount = istep
Taylor_clm.TimingInfo.StartTime = istep
Taylor_clm.TimingInfo.StopTime = 24*no_day
Taylor_clm.TimingInfo.DumpInterval = 1.0
Taylor_clm.TimeStep.Type = 'Constant'
Taylor_clm.TimeStep.Value = 1.0


#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------
Taylor_clm.Geom.Porosity.GeomNames = 'domain s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 g1 g2 g3 g4 g5 g6 g7 g8'
#pfset Geom.Porosity.GeomNames           "domain"

Taylor_clm.Geom.domain.Porosity.Type = 'Constant'
Taylor_clm.Geom.domain.Porosity.Value = 0.33

Taylor_clm.Geom.s1.Porosity.Type = 'Constant'
Taylor_clm.Geom.s1.Porosity.Value = 0.375

Taylor_clm.Geom.s2.Porosity.Type = 'Constant'
Taylor_clm.Geom.s2.Porosity.Value = 0.39

Taylor_clm.Geom.s3.Porosity.Type = 'Constant'
Taylor_clm.Geom.s3.Porosity.Value = 0.387

Taylor_clm.Geom.s4.Porosity.Type = 'Constant'
Taylor_clm.Geom.s4.Porosity.Value = 0.439

Taylor_clm.Geom.s5.Porosity.Type = 'Constant'
Taylor_clm.Geom.s5.Porosity.Value = 0.489

Taylor_clm.Geom.s6.Porosity.Type = 'Constant'
Taylor_clm.Geom.s6.Porosity.Value = 0.399

Taylor_clm.Geom.s7.Porosity.Type = 'Constant'
Taylor_clm.Geom.s7.Porosity.Value = 0.384

Taylor_clm.Geom.s8.Porosity.Type = 'Constant'
Taylor_clm.Geom.s8.Porosity.Value = 0.482

Taylor_clm.Geom.s9.Porosity.Type = 'Constant'
Taylor_clm.Geom.s9.Porosity.Value = 0.442

Taylor_clm.Geom.s10.Porosity.Type = 'Constant'
Taylor_clm.Geom.s10.Porosity.Value = 0.385

Taylor_clm.Geom.s11.Porosity.Type = 'Constant'
Taylor_clm.Geom.s11.Porosity.Value = 0.481

Taylor_clm.Geom.s12.Porosity.Type = 'Constant'
Taylor_clm.Geom.s12.Porosity.Value = 0.459

Taylor_clm.Geom.s13.Porosity.Type = 'Constant'
Taylor_clm.Geom.s13.Porosity.Value = 0.399

Taylor_clm.Geom.g1.Porosity.Type = 'Constant'
Taylor_clm.Geom.g1.Porosity.Value = 0.33

Taylor_clm.Geom.g2.Porosity.Type = 'Constant'
Taylor_clm.Geom.g2.Porosity.Value = 0.33

Taylor_clm.Geom.g3.Porosity.Type = 'Constant'
Taylor_clm.Geom.g3.Porosity.Value = 0.33

Taylor_clm.Geom.g4.Porosity.Type = 'Constant'
Taylor_clm.Geom.g4.Porosity.Value = 0.33

Taylor_clm.Geom.g5.Porosity.Type = 'Constant'
Taylor_clm.Geom.g5.Porosity.Value = 0.33

Taylor_clm.Geom.g6.Porosity.Type = 'Constant'
Taylor_clm.Geom.g6.Porosity.Value = 0.33

Taylor_clm.Geom.g7.Porosity.Type = 'Constant'
Taylor_clm.Geom.g7.Porosity.Value = 0.33

Taylor_clm.Geom.g8.Porosity.Type = 'Constant'
Taylor_clm.Geom.g8.Porosity.Value = 0.33


#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------

Taylor_clm.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

Taylor_clm.Phase.RelPerm.Type = 'VanGenuchten'
Taylor_clm.Phase.RelPerm.GeomNames = 'domain s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13'

Taylor_clm.Geom.domain.RelPerm.Alpha = 1.
Taylor_clm.Geom.domain.RelPerm.N = 3.
Taylor_clm.Geom.domain.RelPerm.NumSamplePoints = 20000
Taylor_clm.Geom.domain.RelPerm.MinPressureHead = -300
Taylor_clm.Geom.domain.RelPerm.InterpolationMethod = 'Linear'


Taylor_clm.Geom.s1.RelPerm.Alpha = 3.548
Taylor_clm.Geom.s1.RelPerm.N = 4.162
Taylor_clm.Geom.s1.RelPerm.NumSamplePoints = 20000
Taylor_clm.Geom.s1.RelPerm.MinPressureHead = -300
Taylor_clm.Geom.s1.RelPerm.InterpolationMethod = 'Linear'

Taylor_clm.Geom.s2.RelPerm.Alpha = 3.467
Taylor_clm.Geom.s2.RelPerm.N = 2.738
Taylor_clm.Geom.s2.RelPerm.NumSamplePoints = 20000
Taylor_clm.Geom.s2.RelPerm.MinPressureHead = -300
Taylor_clm.Geom.s2.RelPerm.InterpolationMethod = 'Linear'

Taylor_clm.Geom.s3.RelPerm.Alpha = 2.692
Taylor_clm.Geom.s3.RelPerm.N = 2.445
Taylor_clm.Geom.s3.RelPerm.NumSamplePoints = 20000
Taylor_clm.Geom.s3.RelPerm.MinPressureHead = -300
Taylor_clm.Geom.s3.RelPerm.InterpolationMethod = 'Linear'

Taylor_clm.Geom.s4.RelPerm.Alpha = 0.501
Taylor_clm.Geom.s4.RelPerm.N = 2.659
Taylor_clm.Geom.s4.RelPerm.NumSamplePoints = 20000
Taylor_clm.Geom.s4.RelPerm.MinPressureHead = -300
Taylor_clm.Geom.s4.RelPerm.InterpolationMethod = 'Linear'

Taylor_clm.Geom.s5.RelPerm.Alpha = 0.661
Taylor_clm.Geom.s5.RelPerm.N = 2.659
Taylor_clm.Geom.s5.RelPerm.NumSamplePoints = 20000
Taylor_clm.Geom.s5.RelPerm.MinPressureHead = -300
Taylor_clm.Geom.s5.RelPerm.InterpolationMethod = 'Linear'

Taylor_clm.Geom.s6.RelPerm.Alpha = 1.122
Taylor_clm.Geom.s6.RelPerm.N = 2.479
Taylor_clm.Geom.s6.RelPerm.NumSamplePoints = 20000
Taylor_clm.Geom.s6.RelPerm.MinPressureHead = -300
Taylor_clm.Geom.s6.RelPerm.InterpolationMethod = 'Linear'

Taylor_clm.Geom.s7.RelPerm.Alpha = 2.089
Taylor_clm.Geom.s7.RelPerm.N = 2.318
Taylor_clm.Geom.s7.RelPerm.NumSamplePoints = 20000
Taylor_clm.Geom.s7.RelPerm.MinPressureHead = -300
Taylor_clm.Geom.s7.RelPerm.InterpolationMethod = 'Linear'

Taylor_clm.Geom.s8.RelPerm.Alpha = 0.832
Taylor_clm.Geom.s8.RelPerm.N = 2.514
Taylor_clm.Geom.s8.RelPerm.NumSamplePoints = 20000
Taylor_clm.Geom.s8.RelPerm.MinPressureHead = -300
Taylor_clm.Geom.s8.RelPerm.InterpolationMethod = 'Linear'

Taylor_clm.Geom.s9.RelPerm.Alpha = 1.585
Taylor_clm.Geom.s9.RelPerm.N = 2.413
Taylor_clm.Geom.s9.RelPerm.NumSamplePoints = 20000
Taylor_clm.Geom.s9.RelPerm.MinPressureHead = -300
Taylor_clm.Geom.s9.RelPerm.InterpolationMethod = 'Linear'

Taylor_clm.Geom.s10.RelPerm.Alpha = 3.311
#pfset Geom.s10.RelPerm.Alpha        2.
Taylor_clm.Geom.s10.RelPerm.N = 2.202
Taylor_clm.Geom.s10.RelPerm.NumSamplePoints = 20000
Taylor_clm.Geom.s10.RelPerm.MinPressureHead = -300
Taylor_clm.Geom.s10.RelPerm.InterpolationMethod = 'Linear'

Taylor_clm.Geom.s11.RelPerm.Alpha = 1.622
Taylor_clm.Geom.s11.RelPerm.N = 2.318
Taylor_clm.Geom.s11.RelPerm.NumSamplePoints = 20000
Taylor_clm.Geom.s11.RelPerm.MinPressureHead = -300
Taylor_clm.Geom.s11.RelPerm.InterpolationMethod = 'Linear'

Taylor_clm.Geom.s12.RelPerm.Alpha = 1.514
Taylor_clm.Geom.s12.RelPerm.N = 2.259
Taylor_clm.Geom.s12.RelPerm.NumSamplePoints = 20000
Taylor_clm.Geom.s12.RelPerm.MinPressureHead = -300
Taylor_clm.Geom.s12.RelPerm.InterpolationMethod = 'Linear'

Taylor_clm.Geom.s13.RelPerm.Alpha = 1.122
Taylor_clm.Geom.s13.RelPerm.N = 2.479
Taylor_clm.Geom.s13.RelPerm.NumSamplePoints = 20000
Taylor_clm.Geom.s13.RelPerm.MinPressureHead = -300
Taylor_clm.Geom.s13.RelPerm.InterpolationMethod = 'Linear'


#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

Taylor_clm.Phase.Saturation.Type = 'VanGenuchten'
Taylor_clm.Phase.Saturation.GeomNames = 'domain s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13'
#
# @RMM added very low Sres to help with dry / large evap
#
Taylor_clm.Geom.domain.Saturation.Alpha = 1.
Taylor_clm.Geom.domain.Saturation.N = 3.
#pfset Geom.domain.Saturation.SRes         0.1
Taylor_clm.Geom.domain.Saturation.SRes = 0.001
Taylor_clm.Geom.domain.Saturation.SSat = 1.0

Taylor_clm.Geom.s1.Saturation.Alpha = 3.548
Taylor_clm.Geom.s1.Saturation.N = 4.162
Taylor_clm.Geom.s1.Saturation.SRes = 0.0001
#pfset Geom.s1.Saturation.SRes         0.1
Taylor_clm.Geom.s1.Saturation.SSat = 1.0

Taylor_clm.Geom.s2.Saturation.Alpha = 3.467
#pfset Geom.s2.Saturation.Alpha        2.5
Taylor_clm.Geom.s2.Saturation.N = 2.738
Taylor_clm.Geom.s2.Saturation.SRes = 0.0001
#pfset Geom.s2.Saturation.SRes         0.1
Taylor_clm.Geom.s2.Saturation.SSat = 1.0

Taylor_clm.Geom.s3.Saturation.Alpha = 2.692
Taylor_clm.Geom.s3.Saturation.N = 2.445
Taylor_clm.Geom.s3.Saturation.SRes = 0.0001
#pfset Geom.s3.Saturation.SRes         0.1
Taylor_clm.Geom.s3.Saturation.SSat = 1.0

Taylor_clm.Geom.s4.Saturation.Alpha = 0.501
Taylor_clm.Geom.s4.Saturation.N = 2.659
#pfset Geom.s4.Saturation.SRes         0.0001
Taylor_clm.Geom.s4.Saturation.SRes = 0.1
Taylor_clm.Geom.s4.Saturation.SSat = 1.0

Taylor_clm.Geom.s5.Saturation.Alpha = 0.661
Taylor_clm.Geom.s5.Saturation.N = 2.659
Taylor_clm.Geom.s5.Saturation.SRes = 0.0001
#pfset Geom.s5.Saturation.SRes         0.1
Taylor_clm.Geom.s5.Saturation.SSat = 1.0

Taylor_clm.Geom.s6.Saturation.Alpha = 1.122
Taylor_clm.Geom.s6.Saturation.N = 2.479
Taylor_clm.Geom.s6.Saturation.SRes = 0.0001
#pfset Geom.s6.Saturation.SRes         0.1
Taylor_clm.Geom.s6.Saturation.SSat = 1.0

Taylor_clm.Geom.s7.Saturation.Alpha = 2.089
Taylor_clm.Geom.s7.Saturation.N = 2.318
Taylor_clm.Geom.s7.Saturation.SRes = 0.0001
#pfset Geom.s7.Saturation.SRes         0.1
Taylor_clm.Geom.s7.Saturation.SSat = 1.0

Taylor_clm.Geom.s8.Saturation.Alpha = 0.832
Taylor_clm.Geom.s8.Saturation.N = 2.514
Taylor_clm.Geom.s8.Saturation.SRes = 0.0001
#pfset Geom.s8.Saturation.SRes         0.1
Taylor_clm.Geom.s8.Saturation.SSat = 1.0

Taylor_clm.Geom.s9.Saturation.Alpha = 1.585
Taylor_clm.Geom.s9.Saturation.N = 2.413
Taylor_clm.Geom.s9.Saturation.SRes = 0.0001
#pfset Geom.s9.Saturation.SRes         0.1
Taylor_clm.Geom.s9.Saturation.SSat = 1.0

Taylor_clm.Geom.s10.Saturation.Alpha = 3.311
#pfset Geom.s10.Saturation.Alpha        2.
Taylor_clm.Geom.s10.Saturation.N = 2.202
Taylor_clm.Geom.s10.Saturation.SRes = 0.0001
#pfset Geom.s10.Saturation.SRes         0.1
Taylor_clm.Geom.s10.Saturation.SSat = 1.0

Taylor_clm.Geom.s11.Saturation.Alpha = 1.622
Taylor_clm.Geom.s11.Saturation.N = 2.318
Taylor_clm.Geom.s11.Saturation.SRes = 0.0001
#pfset Geom.s11.Saturation.SRes         0.1
Taylor_clm.Geom.s11.Saturation.SSat = 1.0

Taylor_clm.Geom.s12.Saturation.Alpha = 1.514
Taylor_clm.Geom.s12.Saturation.N = 2.259
Taylor_clm.Geom.s12.Saturation.SRes = 0.0001
#pfset Geom.s12.Saturation.SRes         0.1
Taylor_clm.Geom.s12.Saturation.SSat = 1.0

Taylor_clm.Geom.s13.Saturation.Alpha = 1.122
Taylor_clm.Geom.s13.Saturation.N = 2.479
Taylor_clm.Geom.s13.Saturation.SRes = 0.0001
#pfset Geom.s13.Saturation.SRes         0.1
Taylor_clm.Geom.s13.Saturation.SSat = 1.0

#----------------------------------------------------------------------------
# Mobility
#----------------------------------------------------------------------------
Taylor_clm.Phase.water.Mobility.Type = 'Constant'
Taylor_clm.Phase.water.Mobility.Value = 1.0

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------
Taylor_clm.Wells.Names = ''

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------
Taylor_clm.Cycle.Names = 'constant'
Taylor_clm.Cycle.constant.Names = 'alltime'
Taylor_clm.Cycle.constant.alltime.Length = 10000000
Taylor_clm.Cycle.constant.Repeat = -1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------
#pfset BCPressure.PatchNames             "land top  bottom"
Taylor_clm.BCPressure.PatchNames = 'top bottom side'

#no flow boundaries for the land borders and the bottom
Taylor_clm.Patch.side.BCPressure.Type = 'FluxConst'
Taylor_clm.Patch.side.BCPressure.Cycle = 'constant'
Taylor_clm.Patch.side.BCPressure.alltime.Value = 0.0

Taylor_clm.Patch.bottom.BCPressure.Type = 'FluxConst'
Taylor_clm.Patch.bottom.BCPressure.Cycle = 'constant'
Taylor_clm.Patch.bottom.BCPressure.alltime.Value = 0.0

Taylor_clm.Patch.top.BCPressure.Type = 'OverlandFlow'
Taylor_clm.Patch.top.BCPressure.Cycle = 'constant'
Taylor_clm.Patch.top.BCPressure.alltime.Value = 0.0000


Taylor_clm.Solver.EvapTransFile = False

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------

Taylor_clm.TopoSlopesX.Type = 'PFBFile'
Taylor_clm.TopoSlopesX.GeomNames = 'domain'

Taylor_clm.TopoSlopesX.FileName = 'slope_x.pfb'


#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------

Taylor_clm.TopoSlopesY.Type = 'PFBFile'
Taylor_clm.TopoSlopesY.GeomNames = 'domain'

Taylor_clm.TopoSlopesY.FileName = 'slope_y.pfb'

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

Taylor_clm.Geom.domain.ICPressure.RefGeom = 'domain'
Taylor_clm.Geom.domain.ICPressure.RefPatch = 'bottom'
#pfset Geom.domain.ICPressure.RefPatch                   top
Taylor_clm.ICPressure.Type = 'PFBFile'
Taylor_clm.ICPressure.GeomNames = 'domain'
Taylor_clm.Geom.domain.ICPressure.FileName = ip

#---------
##  Distribute inputs
#---------

Taylor_clm.dist('slope_x.pfb')#, 'P'=Taylor_clm.Process.Topology.P, 'Q'=Taylor_clm.Process.Topology.Q)

Taylor_clm.dist('slope_y.pfb')#, 'P'=Taylor_clm.Process.Topology.P, 'Q'=Taylor_clm.Process.Topology.Q)

if distribute_forcing:
	for filename_forcing in os.listdir(met_path):
		print(filename_forcing)
		if filename_forcing[-3:]=='pfb':
			Taylor_clm.dist(f'{met_path}{filename_forcing}')#,  'P'=Taylor_clm.Process.Topology.P, 'Q'=Taylor_clm.Process.Topology.Q)
	print('Done distributing forcing')

Taylor_clm.dist(indicator)#, 'P'=Taylor_clm.Process.Topology.P, 'Q'=Taylor_clm.Process.Topology.Q)
print('Done distributing indicator')
Taylor_clm.dist(ip)#, 'P'=Taylor_clm.Process.Topology.P, 'Q'=Taylor_clm.Process.Topology.Q)

print('Done distributing variables')
#---------------------------------------------------------
# Mannings coefficient
#---------------------------------------------------------

Taylor_clm.Mannings.Type = 'Constant'
Taylor_clm.Mannings.GeomNames = 'domain'
#Taylor_clm.Mannings.Geom.domain.Value = 2.e-6
Taylor_clm.Mannings.Geom.domain.Value = 0.0000024
#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

Taylor_clm.PhaseSources.water.Type = 'Constant'
Taylor_clm.PhaseSources.water.GeomNames = 'domain'
Taylor_clm.PhaseSources.water.Geom.domain.Value = 0.0

#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

Taylor_clm.KnownSolution = 'NoKnownSolution'

#----------------------------------------------------------------
# CLM Settings:
# ------------------------------------------------------------
Taylor_clm.Solver.LSM = 'CLM'
Taylor_clm.Solver.CLM.CLMFileDir = './'
Taylor_clm.Solver.CLM.Print1dOut = False
Taylor_clm.Solver.BinaryOutDir = False
Taylor_clm.Solver.CLM.CLMDumpInterval = 1

Taylor_clm.Solver.CLM.MetForcing = '3D'
Taylor_clm.Solver.CLM.MetFileName = 'NLDAS'
Taylor_clm.Solver.CLM.MetFilePath = met_path
Taylor_clm.Solver.CLM.MetFileNT = 24
#pfset Solver.CLM.MetFileNT                            1
Taylor_clm.Solver.CLM.IstepStart = clmstep

Taylor_clm.Solver.CLM.EvapBeta = 'Linear'
Taylor_clm.Solver.CLM.VegWaterStress = 'Saturation'
Taylor_clm.Solver.CLM.ResSat = 0.2
Taylor_clm.Solver.CLM.WiltingPoint = 0.2
Taylor_clm.Solver.CLM.FieldCapacity = 1.00
Taylor_clm.Solver.CLM.IrrigationType = 'none'

Taylor_clm.Solver.CLM.RootZoneNZ = 4
Taylor_clm.Solver.CLM.SoiLayer = 4
Taylor_clm.Solver.CLM.ReuseCount = 1
Taylor_clm.Solver.CLM.WriteLogs = False
## writing only last daily restarts.  This will be at Midnight GMT and 
## starts at timestep 18, then intervals of 24 thereafter
Taylor_clm.Solver.CLM.WriteLastRST = True
#pfset Solver.CLM.WriteLastRST                       False 
Taylor_clm.Solver.CLM.DailyRST = True
Taylor_clm.Solver.CLM.SingleFile = True

#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------
Taylor_clm.Solver = 'Richards'
Taylor_clm.Solver.MaxIter = 100000

Taylor_clm.Solver.TerrainFollowingGrid = True
#pfset Solver.TerrainFollowingGrid.SlopeUpwindFormulation   Upwind


Taylor_clm.Solver.Nonlinear.MaxIter = 2000
Taylor_clm.Solver.Nonlinear.ResidualTol = 1e-5
Taylor_clm.Solver.Nonlinear.EtaChoice = 'EtaConstant'
Taylor_clm.Solver.Nonlinear.EtaValue = 1e-3
Taylor_clm.Solver.Nonlinear.UseJacobian = True

Taylor_clm.Solver.Nonlinear.DerivativeEpsilon = 1e-16
Taylor_clm.Solver.Nonlinear.StepTol = 1e-25

Taylor_clm.Solver.Linear.KrylovDimension = 500
Taylor_clm.Solver.Linear.MaxRestarts = 8
Taylor_clm.Solver.MaxConvergenceFailures = 5

Taylor_clm.Solver.Linear.Preconditioner = 'PFMGOctree'
Taylor_clm.Solver.Linear.Preconditioner.PCMatrixType = 'FullJacobian'
Taylor_clm.Solver.Linear.Preconditioner.PFMGOctree.NumPreRelax = 3
Taylor_clm.Solver.Linear.Preconditioner.PFMGOctree.NumPostRelax = 3
Taylor_clm.Solver.Nonlinear.UseJacobian = True

#pfset Solver.Linear.Preconditioner.PCMatrixType     FullJacobian
Taylor_clm.Solver.WriteSiloPressure = False
Taylor_clm.Solver.PrintSubsurfData = True
Taylor_clm.Solver.PrintMask = True
Taylor_clm.Solver.PrintVelocities = False
Taylor_clm.Solver.PrintSaturation = False
Taylor_clm.Solver.PrintPressure = True
#Writing output (no binary except Pressure, all silo):
Taylor_clm.Solver.PrintSubsurfData = True
#pfset Solver.PrintLSMSink                        True 
Taylor_clm.Solver.PrintSaturation = True
Taylor_clm.Solver.WriteCLMBinary = False
Taylor_clm.Solver.PrintCLM = True
Taylor_clm.Solver.PrintSlopes = True

Taylor_clm.Solver.WriteSiloSpecificStorage = False
Taylor_clm.Solver.WriteSiloMannings = False
Taylor_clm.Solver.WriteSiloMask = False
Taylor_clm.Solver.WriteSiloSlopes = False
Taylor_clm.Solver.WriteSiloSubsurfData = False
Taylor_clm.Solver.WriteSiloPressure = False
Taylor_clm.Solver.WriteSiloSaturation = False
Taylor_clm.Solver.WriteSiloEvapTrans = False
Taylor_clm.Solver.WriteSiloEvapTransSum = False
Taylor_clm.Solver.WriteSiloOverlandSum = False
Taylor_clm.Solver.WriteSiloCLM = False
#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------

Taylor_clm.run()
