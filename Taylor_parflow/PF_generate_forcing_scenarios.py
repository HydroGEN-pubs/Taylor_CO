import numpy as np
import sys
import calendar
import shutil
import glob
import os
import random
from parflow.tools.io import read_pfb, write_pfb

"""
Reference: github link to put here to the repository

This script generates sets of forcing for the selected years based on historical forcing, changing precipitation and temperature homogeneously in space/time

What needs to be done is:
- set $water_years$ to the list of water years of interest

It assumes
- you have the forcing data for Taylor (subset from UCRB with subset_UCRB.py)
- you will be running with 9 processors (otherwise change P,Q, and R)
- you want to use the same forcing scenarios as in Leonarduzzi et al. Otherwise set them to be randomly chosen (manual_selection_scenarios=False) and chose the number of
	scenarios (num_scenarios), or manually choose the corrections (manual_selection_scenarios=True and manually change rainfall_corrections and temperature_corrections)
- you followed the folder structure reccomended on github, README.md (otherwise change all the paths to point in the proper folders/files)
"""

water_years = [1988,1990,2000,2002,2021,2013,2015,2016,2018] #any year(s) between 1983-2018 for which you created subsetted forcing for Taylor

#Processors topology (determines the distribution of forcing files, change this if not running on 3*3*1 = 9 processors)
P = 3
Q = 3
R = 1


manual_selection_scenarios = True #set this to true if you want to choose your own corrections of precipitation and temperature
num_scenarios = 10
def generate_random_value(value_min,value_max):
      rand = random.random()
      return round(rand*(value_max-value_min)+value_min,2)

if manual_selection_scenarios: #setting correcting factors of scenarios manually: currently to reproduce Leonarduzzi et al. (D1-D12)
	rainfall_corrections    = np.array((0.80,0.58,0.63,0.52,0.83,0.93,0.81,0.81,0.90,0.65,0.65,0.58)) #indicate here each value of precipitation correction (multiplicative factor)
	temperature_corrections = np.array((0.50,0.44,0.51,0.56,0.62,0.68,0.45,0.51,0.25,0.65,0.80,0.60)) #indicate here each value of precipitation correction (additive factor)
	num_scenarios = rainfall_corrections.shape[0]
else:
	rainfall_corrections=np.zeros((num_scenarios))
	temperature_corrections=np.zeros((num_scenarios))
	for i in range(num_scenarios):
		rainfall_corrections[i] = generate_random_value(0.5,1)
		temperature_corrections[i] = generate_random_value(0,1)


#path where the historical forcing is saved (where you saved the subset forcing from UCRB to Taylor)
path_forcing_historical = 'historical_forcing/'

#path where to save forcing for the forcing scenarios, will create a folder for each forcing set
root_save = 'forcing_scenarios'

if not os.path.exists(root_save):
      os.makedirs(root_save)

#creating the folders for each forcing scenario and copying there the corresponding historical forcing
for i in range(num_scenarios):
	print(f'scenario {i+1} of {num_scenarios}')
	print(f'Correction rain: {rainfall_corrections[i]}')
	print(f'Correction temperature: {temperature_corrections[i]}')
	#creating folder where to store the forcing for the scenario
	id_temp = ("{:.2f}".format(temperature_corrections[i])).replace('.', '')
	id_rain = ("{:.2f}".format(rainfall_corrections[i])).replace('.', '')
	for year in water_years:
		print(f'copying year: {year}')
		path_save = f'{root_save}/temp{id_temp}_prec{id_rain}/{year}/NLDAS/'
		curr_path_met = f'{path_forcing_historical}/{year}/NLDAS'
		os.makedirs(path_save)
		for filename in glob.glob(os.path.join(curr_path_met, '*.*')):
			shutil.copy(filename, path_save)
		#copy the correct drv_clmin to the scenario folder
		shutil.copy(f'{path_forcing_historical}/{year}/drv_clmin.dat', f'{root_save}/temp{id_temp}_prec{id_rain}/{year}/')
		#copy the intitial pressure to the scenario folder
		shutil.copy(f'{path_forcing_historical}/{year}/initial_pressure.pfb', f'{root_save}/temp{id_temp}_prec{id_rain}/{year}/')
		#copy the running script to the scenario folder
		shutil.copy(f'{path_forcing_historical}/{year}/Taylor_parflow.py', f'{root_save}/temp{id_temp}_prec{id_rain}/{year}/')
		print("Done copying forcing")

#Correcting precipitation and forcing for each scenario
for year in water_years:
	print(f'Correcting year: {year}')
	path_met = f'{path_forcing_historical}/{year}/NLDAS/'
	if calendar.isleap(year):
		N_hours = 366*24
	else:
		N_hours = 365*24
	
	for i in range(1,N_hours,24):
		t0_forc = str(int(i)).rjust(6, '0')
		t1_forc = str(int(i+23)).rjust(6, '0')
		
		APCP = read_pfb(f'{path_met}NLDAS.APCP.{t0_forc}_to_{t1_forc}.pfb')
		Temp = read_pfb(f'{path_met}NLDAS.Temp.{t0_forc}_to_{t1_forc}.pfb')
		for scen in range(num_scenarios):
			id_temp = ("{:.2f}".format(temperature_corrections[scen])).replace('.', '')
			id_rain = ("{:.2f}".format(rainfall_corrections[scen])).replace('.', '')
			if i==1:
				print(f'Correcting scenario {scen+1} out of {num_scenarios}')
				print(f'temp{id_temp}_prec{id_rain}')
			path_save = f'{root_save}/temp{id_temp}_prec{id_rain}/{year}/NLDAS/'
			APCP1 = np.copy(APCP)*rainfall_corrections[scen]
			write_pfb(f'{path_save}NLDAS.APCP.{t0_forc}_to_{t1_forc}.pfb',APCP1,dx=1000,dy=1000,p=P,q=Q,r=R)
			Temp1 = np.copy(Temp)+temperature_corrections[scen]
			write_pfb(f'{path_save}NLDAS.Temp.{t0_forc}_to_{t1_forc}.pfb',Temp1,dx=1000,dy=1000,p=P,q=Q,r=R)
	
	
	
	
	
	
	
	
	