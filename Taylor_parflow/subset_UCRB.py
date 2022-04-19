import numpy as np
from parflow.tools.io import read_pfb, write_pfb, read_clm
from PIL import Image
import calendar
import os
import shutil

"""

Reference: github link to put here to the repository

This script subsets 
 - the forcing data and the initial pressure from UCRB for the selected water year
(UCRB data in: https://datacommons.cyverse.org/browse/iplant/home/shared/commons_repo/curated/Tran_UpperCO_simulation_Sep2020).
 - all static inputs (i.e., slopes, indicator file, mask, and landcover)
Prepares the drv_clmin.dat for the selected year and copies the parflow-clm script in the run folder

What needs to be done is:
- set $water_years$ to the list of water years of interest

It assumes
- you downloaded the forcing data for UCRB (see README.md for more guidance)
- you downloaded the initial pressure for UCRB (see README.md for more guidance)
- you downloaded the static inputs for UCRB (see README.md for more guidance)
- you will be running with 9 processors (otherwise change P,Q, and R)
- you followed the folder structure reccomended on github, README.md (otherwise change all the paths to point in the proper folders/files)

"""

water_years = [1988,1990,2000,2002,2021,2013,2015,2016,2018]

# -----------------
# TOPOLOGY
# -----------------
#Processors topology (determines the distribution of forcing files, change this if not running on 3*3*1 = 9 processors)
P = 3
Q = 3
R = 1     

#path to where the mask is stored (provided in inputs in the github repository)
path_mask = 'UCRB/9110000_WatershedMask.tif'

# -----------------
# PATHS WHERE THE UCRB STATIC INPUTS WHERE DOWNLOADED (datacommons.cyverse repository)
# -----------------
path_UCRB_vegm = 'UCRB/static_UCRB/drv_vegm_v2.UC.dat' #drv_vegm file
path_UCRB_slopex = 'UCRB/static_UCRB/UpperCO.slopex.rivth1500.pfb' #slope in x
path_UCRB_slopey = 'UCRB/static_UCRB/UpperCO.slopey.rivth1500.pfb' #slope in y
path_UCRB_indicator = 'UCRB/static_UCRB/UpperCO_IndicatorFile_v2.pfb' #indicator file (subsurface regions)


# -----------------
# PATHS WHERE THE Taylor STATIC INPUTS WILL BE SAVED
# -----------------
path_static_out = 'static_Taylor/' #will be created if it doesn't exist
path_Taylor_mask = 'static_Taylor/mask.pfb'
path_Taylor_vegm = 'static_Taylor/drv_vegm_v2.Taylor.dat'
path_Taylor_slopex = 'static_Taylor/slope_x.pfb'
path_Taylor_slopey = 'static_Taylor/slope_y.pfb'
path_Taylor_indicator = 'static_Taylor/Taylor.IndicatorFile_v2.pfb'

# -----------------
# MASK
# -----------------
#reading the mask of Taylor watershed and defining bounding box
mask_Taylor_in_UCRB = np.array(Image.open(f'{path_mask}'))
mask_watershed = np.flip(mask_Taylor_in_UCRB,0) #y-axis in PF is S to N
where = np.array(np.where(mask_watershed))
x1, y1 = np.amin(where, axis=1)
x2, y2 = np.amax(where, axis=1)
#to make sure x2 and y2 are included in the subsetting
x2+=1
y2+=1


# -----------------
# SUBSETTING STATIC INPUTS (only needed once)
# -----------------

#check/create static input directory
if not os.path.exists(path_static_out):
        os.makedirs(path_static_out)

#subsetting the mask
mask_taylor = mask_watershed[x1:x2, y1:y2]
mask_taylor[mask_taylor>0]=1
write_pfb(path_Taylor_mask,mask_taylor.astype('float64'),dx=1000,dy=1000,dz=200,dist=False)

#subsetting the indicator file (describing subsurface regions)
indicator = read_pfb(path_UCRB_indicator)[:,x1:x2, y1:y2]
write_pfb(path_Taylor_indicator,indicator,dx=1000,dy=1000,dz=200,dist=False)

#subsetting the slopes
slope_x = np.squeeze(read_pfb(path_UCRB_slopex))[x1:x2, y1:y2]
write_pfb(path_Taylor_slopex,slope_x,dx=1000,dy=1000,dz=200,dist=False)

slope_y = np.squeeze(read_pfb(path_UCRB_slopey))[x1:x2, y1:y2]
write_pfb(path_Taylor_slopey,slope_y,dx=1000,dy=1000,dz=200,dist=False)

#subsetting the drv_vegm file
vegm_UCRB = read_clm(path_UCRB_vegm,type='vegm')
vegm_subset = vegm_UCRB[x1:x2, y1:y2, :]
#CREATING/OPENING OUTPUT FILE
out_file = open(path_Taylor_vegm, "w")

#WRITING FIRST TWO LINES WHICH ARE FIXED
out_file.write('x y lat lon sand clay color fractional coverage of grid, by vegetation class (Must/Should Add to 1.0)\n')
out_file.write('  (Deg) (Deg) (%/100)  index 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18\n')

for yy in range(vegm_subset.shape[0]):
    if yy%5 ==0:
        print(f'Out of {vegm_subset.shape[0]} cells in y, we are now at: {yy}')
    for xx in range(vegm_subset.shape[1]):
        #WRITING THE vegm ENTRIES FOR THE CURRENT POINT
        out_file.write(f'{xx+1} {yy+1} ' \
        #this is the latitude, every domain cell is a new line in the latlon file
        f'{vegm_subset[yy,xx,0]:.2f} ' \
        #this is the longitude, every domain cell is a new line in the latlon file
        f'{vegm_subset[yy,xx,1]:.2f} ' \
        # %/100 of sand
        f'{vegm_subset[yy,xx,2]:.2f} ' \
        # %/100 of clay
        f'{vegm_subset[yy,xx,3]:.2f} ' \
        # soil color class
        f'{int(vegm_subset[yy,xx,4])} ' \
        # landcover indices
        f'{int(vegm_subset[yy,xx,5])} ' \
        f'{int(vegm_subset[yy,xx,6])} ' \
        f'{int(vegm_subset[yy,xx,7])} ' \
        f'{int(vegm_subset[yy,xx,8])} ' \
        f'{int(vegm_subset[yy,xx,9])} ' \
        f'{int(vegm_subset[yy,xx,10])} ' \
        f'{int(vegm_subset[yy,xx,11])} ' \
        f'{int(vegm_subset[yy,xx,12])} ' \
        f'{int(vegm_subset[yy,xx,13])} ' \
        f'{int(vegm_subset[yy,xx,14])} ' \
        f'{int(vegm_subset[yy,xx,15])} ' \
        f'{int(vegm_subset[yy,xx,16])} ' \
        f'{int(vegm_subset[yy,xx,17])} ' \
        f'{int(vegm_subset[yy,xx,18])} ' \
        f'{int(vegm_subset[yy,xx,19])} ' \
        f'{int(vegm_subset[yy,xx,20])} ' \
        f'{int(vegm_subset[yy,xx,21])} ' \
        f'{int(vegm_subset[yy,xx,22])}\n')
out_file.close()

# -----------------
# SUBSETTING FORCING, INITIAL PRESSURE, and CORRECTING DRV_CLMIN.dat FOR EACH WATER YEAR
# -----------------

for selected_water_year in water_years:
    
    # -----------------
    # PATHS WHERE THE UCRB FORCING AND INITIAL PRESSURE WHERE DOWNLOADED (datacommons.cyverse repository) AND FOR RUNNING HISTORICAL SIMULATIONS
    # -----------------
    #path where you'll be running parflow with historical forcing
    path_running = f'historical_forcing/{selected_water_year}/'
    #All paths assume the same data structure as in the UCRB repository is mantained. The relevative path to where this is saved should be provided here:
    path_forcing = f'UCRB/{selected_water_year}/NLDAS/'
    #Path where you want to save the forcing for the subset
    path_forcing_out = f'{path_running}/NLDAS/'
    #path where the pressure at t=0 for the selected water year is stored
    path_pressure = f'UCRB/{selected_water_year}/'
    #path where the provided drv_clmin.dat is stored
    path_drv_clmin_in = 'static_Taylor/drv_clmin.dat'
    
    # -----------------
    # CREATE OUTPUT FOLDERS IF THEY DON't EXIST
    # -----------------
    if not os.path.exists(path_running):
        os.makedirs(path_running)
    if not os.path.exists(path_forcing_out):
        os.makedirs(path_forcing_out)
    

    # -----------------
    # CORRECT drv_clmin file for chosen water year
    # -----------------
    a_file = open(f'{path_drv_clmin_in}', "r")
    list_of_lines = a_file.readlines()

    list_of_lines[35]=f'syr {selected_water_year-1} Starting Year\n'
    list_of_lines[42]=f'eyr {selected_water_year} Ending Year\n'

    list_of_lines[29]=f'startcode      2\n'
    list_of_lines[46]=f'clm_ic         2\n'

    a_file = open(f'{path_running}drv_clmin.dat', "w")
    a_file.writelines(list_of_lines)
    a_file.close()

    # -----------------
    # SUBSETTING DYNAMIC VARIABLES (forcing and initial pressure)
    # -----------------

    #checking number of hours in the year
    if calendar.isleap(selected_water_year):
        N_hours = 8784+1
    else:
        N_hours = 8760+1

    #function to read, subset and write pfbs
    def read_subset_write_pfb(filename_in,filename_out,x1,x2,y1,y2,P,Q,R):
        variable = read_pfb(filename_in)[:,x1:x2, y1:y2]
        write_pfb(filename_out,variable,dx=1000,dy=1000,p=P,q=Q,r=R)
        del variable

    for i in range(1,N_hours,24):
        t0_forcing = str(int(i)).rjust(6, '0') 
        t1_forcing = str(int(i+23)).rjust(6, '0') 
        forcing_end_filename = f'.{t0_forcing}_to_{t1_forcing}.pfb'
        read_subset_write_pfb(f'{path_forcing}NLDAS.APCP{forcing_end_filename}',f'{path_forcing_out}NLDAS.APCP{forcing_end_filename}',x1,x2,y1,y2,P,Q,R)
        read_subset_write_pfb(f'{path_forcing}NLDAS.Temp{forcing_end_filename}',f'{path_forcing_out}NLDAS.Temp{forcing_end_filename}',x1,x2,y1,y2,P,Q,R)
        read_subset_write_pfb(f'{path_forcing}NLDAS.UGRD{forcing_end_filename}',f'{path_forcing_out}NLDAS.UGRD{forcing_end_filename}',x1,x2,y1,y2,P,Q,R)
        read_subset_write_pfb(f'{path_forcing}NLDAS.VGRD{forcing_end_filename}',f'{path_forcing_out}NLDAS.VGRD{forcing_end_filename}',x1,x2,y1,y2,P,Q,R)
        read_subset_write_pfb(f'{path_forcing}NLDAS.Press{forcing_end_filename}',f'{path_forcing_out}NLDAS.Press{forcing_end_filename}',x1,x2,y1,y2,P,Q,R)
        read_subset_write_pfb(f'{path_forcing}NLDAS.SPFH{forcing_end_filename}',f'{path_forcing_out}NLDAS.SPFH{forcing_end_filename}',x1,x2,y1,y2,P,Q,R)
        read_subset_write_pfb(f'{path_forcing}NLDAS.DSWR{forcing_end_filename}',f'{path_forcing_out}NLDAS.DSWR{forcing_end_filename}',x1,x2,y1,y2,P,Q,R)
        read_subset_write_pfb(f'{path_forcing}NLDAS.DLWR{forcing_end_filename}',f'{path_forcing_out}NLDAS.DLWR{forcing_end_filename}',x1,x2,y1,y2,P,Q,R)

    #reading initial pressure and subsetting it
    read_subset_write_pfb(f'{path_pressure}UC_clm_{selected_water_year}.out.press.00000.pfb',f'{path_running}initial_pressure.pfb',x1,x2,y1,y2,P,Q,R)

    #copying the run script
    shutil.copy(f'Taylor_clm.py', f'{path_running}')

