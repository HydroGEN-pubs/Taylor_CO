# -----------------
# IMPORTS
# -----------------
import os
from copy import copy
from datetime import date
import datetime
import importlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import pickle

from parflow import Run
from parflow.tools.settings import set_working_directory
from parflow.tools.io import read_pfb
from hydrogen.transform import float32_clamp_scaling
from utils import FORCING_VARIABLES, CLM_OUTPUT_VARIABLES, hours_in_year, compute_stats, plot
import pickle


"""
Reference: github link to put here to the repository

This script trains a 2D CNN as outlined in Leonarduzzi et al.
The different options in the first section (`USER CHOICES/SETTINGS`) must be filled in with the specific choices or paths (Section 'FUNCTIONS').
The required changes are:
- PATH_TO_REPOSITORY: where the git repository was cloned (name of repository must be incuded, currently added)

It assumes
- ParFlow-CLM runs exist for all years and scenarios used (if not, see folder Taylor_parflow and instructions provided there)
- that the CNN model is available ('RMM_NN_2D_B1.py' available on github in the `models` subfolder)
- that the mask for the Taylor catchment is available (created with the subsetting from UCRB, in `Taylor_parflow/static_Taylor/mask.pfb`, see Taylor_parflow git folder)
"""


# -----------------
# USER CHOICES/SETTINGS
# -----------------

PATH_TO_REPOSITORY = '/home/user/Desktop/git_repo/'+'Taylor_CO/'

#these are the path where the run with historical forcing and with the different forcing scenarios are stored. 
# This assumes that the folder organisation for 
#     forcing scenarios is (consistent with PF_generate_forcing_scenarios.py):
#           f'ForcingScenarios/temp{correction_temperature}_prec{correction_precipitation}/{year_run}/'
#      historical forcing is:
#           f'historical_forcing/{year_run}'
# if organize otherwise, you'll need to adapt the code to point to the proper locations
INPUT_FOLDER = {
   'historical_forcing' : f'{PATH_TO_REPOSITORY}/Taylor_parflow/historical_forcing/',
   'forcing_scenarios' : f'{PATH_TO_REPOSITORY}/Taylor_parflow/ForcingScenarios/'
}

#change this to point wherever you want your output folder to be located (will be created if it doesn't exist)
OUTPUT_FOLDER = f'2D_CNN_1/'

#path where the models are saved (must contain `RMM_NN_2D_B1.py`)
PATH_MODELS = f'{PATH_TO_REPOSITORY}/Taylor_ml/models/'

#set this to true if you want to save pickle files of the losses in training and validation over the epochs
SAVE_PICKLE_LOSSES = True 

#set this to True if you want to save pickle files of the soil moisture of ParFlow-CLM (referred to as "obs.") and the CNN (referred to as "pred.")
SAVE_PICKLE_SOILMOISTURE = True

PATH_MASK = f'{PATH_TO_REPOSITORY}/Taylor_parflow/static_Taylor/mask.pfb' #path where the mask of the Taylor catchment is stored (available in the github repository)
TRAIN = True #if you want to train the model
EVALUATE = True #if you want to carry out testing (can be done only once the model has been trained)
FIRST_DAY_YEAR=0 #first day to consider. FIRST_DAY_YEAR days at the beginning of each year will be removed
USE_PREVIOUS_PREDICTION = True #If true recursive testing (i.e. output of ML used as input for next prediction), if False always using ParFlow input to predict next step
PLOTS_TRAINING = True #if True, saving the plot of the losses in training & validation as a function of the epochs
PLOTS_TESTING = False #if True, saving snapshots of labels and predictions

YEARS_TRAIN = [1988, 1990, 2015, 2016, 2018] #years for training of the CNN
YEARS_VAL = [2000, 2012, 2013] #years for validation (early stopping) of the CNN
YEARS_TESTING = [2002] #years for testing/evaluation

RUNNAME_PARFLOW = 'Taylor_{year}' #the runname used in your ParFlow-CLM runs (year will be replaced with the actual number)

SHUFFLING_DAYS = 'across_years' # False (no shuffling), 'across_years' (shuffling of all training data), 'in_year' (shuffling within each year&scenario)

CUT_TO_364 = True # set this to True if you want only the first 364 days to be considered (i.e., you throw away 30th of Sept.) -->helpful to control batch size

RANDOM_SEED = None #set to a number >0 if you want to choose the random seed (to get same results when repeating same experiment)

INPUT_DYNAMIC = ('qflx_infl', 'qflx_tran_veg', 'soil_moisture')
INPUT_STATIC = ('slope_x', 'slope_y','computed_porosity', 'computed_permeability_x')
OUTPUT_FEATURE = 'soil_moisture'

TEMPORAL_RESOLUTION = 24 #number of hours for resolution of data, if 24 is daily
BATCH_SIZE = 100 #batch size

TRAINING_SCENARIOS = ['FB', 'temp050_prec080', 'temp044_prec058', 'temp051_prec063', 'temp056_prec052', 'temp062_prec083', 'temp068_prec093', 'temp045_prec081', 'temp051_prec081', 'temp_025_prec090', 'temp065_prec065', 'temp080_prec065']  #list of scenarios to do the training on, for historical, use ['FB']

TESTING_SCENARIOS = ['temp060_prec058']

#SCALING: transforming values to 0:1 or -1:1 ranges
# <feature_name> => (<input_range>, <output_range>) mapping
# NOTE: Inputs and outputs are clamped at the boundaries of these limits, so select these ranges very carefully!
# Reverse scaling is done automatically

SCALING = {
    'soil_moisture': ((0.002553,0.482),(0,1)),
    'qflx_tran_veg': ((0.0,0.00006),(0,1)),
    'qflx_infl': ((-0.00005,0.00049259),(-1,1)),
    'computed_porosity': ((0.33,0.482),(0,1)),
    'slope_x': ((-0.405, 0.457),(-1,1)),
    'slope_y': ((-0.405, 0.457),(-1,1)),
    'computed_permeability_x': ((0.004675077,0.06),(0,1))
}





# -----------------
# FUNCTIONS
# -----------------
def train(inputs_dynamic, inputs_static, output_feature, input_folder, output_folder, years, filename_mask,
          run_name_format, path_models, resolution_hours=24, batch_size=None, learning_rate=10e-5, n_epoch=20, scaling={},
          plots=False, random_seed=None, first_day_to_consider=None, shuffle_days = False, cut_to_364=False,years_val=None,training_scenarios=['FB'], n_epoch_loss = 100):
   """
   Run Training on multiple years for a domain run
   :param inputs_dynamic: Any of the 8 CLM forcing variables, 13 CLM output variables,
     or soil_moisture / pressure / saturation / wtd
     Coming soon - surface_storage/subsurface_storage/et/overland_flow
   :param inputs_static: Any 3D property of DataAccessor
     Note that only the surface-level value (or shape ny, nx) is considered for now
     Example properties include slope_x/slope_y/computed_permeability_<x/y/z>/elevation/computed_porosity
   :param output_feature: Any one of inputs_dynamic/inputs_static
   :param input_folder: Folder with input data
   :param output_folder: Folder to save output
   :param years: Year(s) for which to run training
   :param filename_mask: Path the mask .pfb file for domain
   :param path_models: path where the models are defined
   :param run_name_format: Run name format to use to locate .pfidb file in input_folder
     '{year}', if present in run_name_format, is expanded automatically
   :param resolution_hours: No. of hours to group together for training
     For dynamic inputs, the mean value across resolution_hours is computed for training purposes.
   :param batch_size: Size of each batch to train. If None, all samples are trained in a single batch.
   :param learning_rate: Learning Rate
   :param n_epoch: No. of epochs
   :param scaling: A dictionary with mapping <feature_name>: (<input_range>, <output_range>)
   :param plots: boolean; Whether to save plot .png files while training
   :param random_seed: Random seed to use for torch; If None, no random seed is pushed
   :param first_day_to_consider: Day of the year to start your training from for each year (e.g., if 10, the first 9 days of the year will be removed)
   :param shuffle_days: if shuffling the days or keep them in chronological order (if "in_year" shuffling will be only within each year, if "across_years" shuffling will be among all timesteps/years considered, if False, no shuffling)
   :param cut_to_364: whether to cut the last day of the year (to avoid having leep years one day longer)
   :param years_val: Year(s) for which to do validation (will determine early stopping)
   :param training_scenarios: scenarios to use in training (this is specific to where you stored your runs, will determine paths)
   :param n_epoch_loss: number of epochs over which to average the loss for early stopping (stop when mean over last n_epoch_loss is larger than antecedent n_epoch_loss)

   :return: None
   """

   DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
   print(f'training on: {DEVICE}')
   
   #if chosen, setting the random seed (to allow to reproduce same results)
   if random_seed is not None:
     torch.manual_seed(random_seed)
   
   #number of inputs, sum of dynamic and static inputs
   n_channel = len(inputs_dynamic) + len(inputs_static)
   
   #identifying where in the inputs the label variable is
   label_index = (inputs_static + inputs_dynamic).index(output_feature)
   
   #loading the mask
   mask = read_pfb(filename_mask).squeeze()  # 3D -> 2D
   mask = mask > 0  # boolean 2D mask
   
   n_timesteps = sum(hours_in_year(year) for year in years)
   n_timesteps = n_timesteps // resolution_hours

   ny, nx = mask.shape
   
   #setting up CNN model
   sys.path.append(f'{path_models}')
   from RMM_NN_2D_B1 import RMM_NN
   model = RMM_NN(grid_size=[ny, nx], channels=n_channel, verbose=True)
   
   model.to(DEVICE) #using GPU if available
   model.verbose = False
   model.use_dropout = True
   
   def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

   print(f'number of model parameters: {count_parameters(model)}')
   
   #creating the transformation for the variables selected
   transforms = {k: float32_clamp_scaling(src_range=v[0], dst_range=v[1]) for k, v in scaling.items()}

   X = np.zeros((0, n_channel, ny, nx))
   y = np.zeros((0, ny * nx))
   
   def unison_shuffled_copies(a, b):
      assert a.shape[0] == b.shape[0]
      p = np.random.permutation(a.shape[0])
      return a[p,:,:,:], b[p,:]
   
   #reading one year at a time
   for year in years:
      print(year)
      for scen in training_scenarios:
         if scen == 'FB':
            curr_path_input = input_folder['historical_forcing']
            input_folder_ = f'{curr_path_input}/{year}'
         else:
            curr_path_input = input_folder['forcing_scenarios']
            input_folder_ = f'{curr_path_input}{scen}/{year}'
            
         print('Reading input:')
         print(input_folder_)
         _X, _y = create_Xy(input_folder_, year, run_name_format, label_index, inputs_static, inputs_dynamic,
                     output_feature, mask=mask, transforms=transforms)
         
         if cut_to_364:
            print('I am only using first 364 days in training')
            _X = _X[0:364,:,:,:]
            _y = _y[0:364,:]
         if shuffle_days=='in_year':
            _X, _y = unison_shuffled_copies(_X,_y)
            print('I am shuffling every day within each year')
         if first_day_to_consider is not None:
            _X=_X[first_day_to_consider:,:,:,:]  
            _y=_y[first_day_to_consider:,:]
         X = np.concatenate([X, _X], axis=0)
         y = np.concatenate([y, _y], axis=0)
   
   if shuffle_days=='across_years':
      X, y = unison_shuffled_copies(X,y)
      print('I am shuffling every day across years')
   print("Reading validation inputs/outputs")
   
   #reading the validation years (if not None)
   if years_val is not None:
      x_val = np.zeros((0, n_channel, ny, nx))
      y_val = np.zeros((0, ny * nx))
      for year in years_val:
         print("A scenario was indicated")
         for scen in training_scenarios:
            if scen == 'FB':
               curr_path_input = input_folder['historical_forcing']
               input_folder_ = f'{curr_path_input}/{year}'
            else:
               curr_path_input = input_folder['forcing_scenarios']
               input_folder_ = f'{curr_path_input}{scen}/{year}'
            
            _X, _y = create_Xy(input_folder_, year, run_name_format, label_index, inputs_static, inputs_dynamic,
                              output_feature, mask=mask, transforms=transforms)
            
            if cut_to_364:
               print('I am only using first 364 days in training')
               _X = _X[0:364,:,:,:]
               _y = _y[0:364,:]
            if shuffle_days=='in_year':
               _X, _y = unison_shuffled_copies(_X,_y)
               print('I am shuffling every day within each year')
            if first_day_to_consider is not None:
               _X=_X[first_day_to_consider:,:,:,:]  
               _y=_y[first_day_to_consider:,:]
            x_val = np.concatenate([x_val, _X], axis=0)
            y_val = np.concatenate([y_val, _y], axis=0)
   
   
   X = torch.from_numpy(X).type(torch.FloatTensor)
   y = torch.from_numpy(y).type(torch.FloatTensor)
   
   
   x_val  = torch.from_numpy(x_val).type(torch.FloatTensor)
   y_val = torch.from_numpy(y_val).type(torch.FloatTensor)
   #if running into GPU memory issues, this could be done differently (within the loop)
   x_val = x_val.type(torch.FloatTensor).to(DEVICE)
   y_val = y_val.type(torch.FloatTensor).to(DEVICE)
   
   flattened_mask = mask.flatten()
   losses = np.empty(n_epoch)
   losses_val=np.empty(n_epoch)
   optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
   scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,learning_rate,total_steps=n_epoch)

   if batch_size is None: #if no batch_size, considering just 1 batch
      batch_size = X.shape[0]
   n_iteration_per_epoch = (X.shape[0] + (batch_size - 1)) // batch_size
   
   # Attach useful information to the model object as attributes
   model.inputs_dynamic = inputs_dynamic
   model.inputs_static = inputs_static
   model.output_feature = output_feature
   model.filename_mask = filename_mask
   model.run_name_format = run_name_format
   model.scaling = scaling
   
   os.makedirs(output_folder, exist_ok=True)


   for epoch in range(n_epoch): #looping through epochs
      #splitting inputs and labels into batches
      Xs = np.array_split(X, n_iteration_per_epoch)
      ys = np.array_split(y, n_iteration_per_epoch)
      loss_train = 0
      for _X, _y in zip(Xs, ys): #looping through batches
         #sending them to GPU (if available)-->done here to avoid filling GPU memory
         _X = _X.type(torch.FloatTensor).to(DEVICE) 
         _y = _y.type(torch.FloatTensor).to(DEVICE)
         model.train() #setting the model to training mode
         optimizer.zero_grad()
         # Forward pass
         y_pred = model(_X)
      
         # For calculating loss, set y (actual/predicted) values outside the mask to 0
         _y[:, ~flattened_mask] = 0
         y_pred[:, ~flattened_mask] = 0
      
         loss = torch.nn.SmoothL1Loss()(y_pred, _y)
         #loss_train += np.copy(loss.detach().numpy())
         loss_train+=np.copy(loss.data.cpu().numpy())
         
         loss.backward()
         optimizer.step()
      losses[epoch] = loss_train / n_iteration_per_epoch #storing the training loss
      
      if years_val is not None:
         model.eval() #setting the model to evaluation mode
         y_pred_val = model(x_val)
         #computing the losses over validation set
         loss_val = torch.nn.SmoothL1Loss()(y_pred_val, y_val)
         losses_val[epoch] = np.copy(loss_val.data.cpu().numpy()) 
         
         print("Epoch: %3d, Train loss: %5.3e, Val loss: %5.3e" % (epoch, loss,loss_val), end='\r')
         #checking if the performances in validation are still improving or not
         if epoch>n_epoch_loss*2-1: #still improving
            if np.mean(losses_val[epoch-n_epoch_loss:epoch+1])<np.mean(losses_val[epoch-n_epoch_loss*2:epoch-n_epoch_loss]):
               #still improving performances in validation
               #save the model (next time they might not improve anymore and you should keep these results)
               torch.save(model, f'{output_folder}/model.pik')
               torch.save(model.state_dict(), f'{output_folder}/model.pth')
            else: #performances have gotten worse
               print(f'Breaking loop here: {epoch}')
               print(f'The mean loss between epochs {epoch-n_epoch_loss} and {epoch+1} ({np.mean(losses_val[epoch-n_epoch_loss:epoch+1])}) is greater than between epochs {epoch-n_epoch_loss*2} and {epoch-n_epoch_loss} ({np.mean(losses_val[epoch-n_epoch_loss*2:epoch-n_epoch_loss])})')
               losses = losses[0:epoch+1]
               losses_val = losses_val[0:epoch+1]
               break #exit the training
      else:
         print("Epoch: %3d, loss: %5.3e" % (epoch, loss), end='\r')
      scheduler.step()
      
   
   if plots:
      # plotting the losses during training as a function of the epoch number
      os.makedirs(f'{output_folder}/images', exist_ok=True)
      
      fig, axs = plt.subplots(1, 1)
      axs.plot(losses,label='training')
      axs.plot(losses_val,label='validation')
      axs.set_title('Loss')
      axs.set_xlabel('epoch')
      axs.set_ylabel('Loss')
      axs.set_yscale('log')
      plt.legend()
      plt.savefig(f'{output_folder}/images/losses.png')
      plt.close(fig)

   #saving pickles of losses in training and validation
   if SAVE_PICKLE_LOSSES:
      pickle.dump(losses, open( f'{output_folder}/Losses.p', "wb" ) )
      pickle.dump(losses_val, open( f'{output_folder}/LossesVal.p', "wb" ) )

def evaluate(model, input_folder, output_folder, years, plots=False, first_day_to_consider=None, evaluating_scenario=['FB'], use_previous_prediction=True):
   """
   Evaluate Model
   :param model: Model object, with weights pre-loaded
   :param input_folder: Folder with input data
   :param output_folder: Folder to save output
   :param years: Year(s) for which to run evaluation
   :param plots: boolean; whether to save .png files during evaluation
   :param first_day_to_consider: Day of the year to start your training from for each year (e.g., if 10, the first 9 days of the year will be removed)
   :param evaluating_scenario: scenarios to use in testing (this is specific to where you stored your runs, will determine paths)
   :param use_previous_prediction: if set to true, CNN model used recursively in prediction (using in t the soil moisture predicted at t-1 instead of parflow-s sm at t)
   :return: None
   """

   print(f'Evaluating on: {evaluating_scenario}')
   # Retrieve useful information from the model object's attributes
   inputs_dynamic = model.inputs_dynamic
   inputs_static = model.inputs_static
   output_feature = model.output_feature
   filename_mask = model.filename_mask
   run_name_format = model.run_name_format
   scaling = model.scaling
   
   transforms = {k: float32_clamp_scaling(src_range=v[0], dst_range=v[1]) for k, v in scaling.items()}
   reverse_transforms = {k: float32_clamp_scaling(src_range=v[1], dst_range=v[0]) for k, v in scaling.items()}
   print(transforms)
   print(transforms['soil_moisture'])
   
   label_index = (inputs_static + inputs_dynamic).index(output_feature)
   
   mask = read_pfb(filename_mask).squeeze()  # 3D -> 2D
   mask = mask > 0  # boolean 2D mask
   ny, nx = mask.shape

   flattened_mask = mask.flatten()
   
   model.use_dropout = False
   for parameter in model.parameters():
      parameter.requires_grad = False
   
   for year_i, year in enumerate(years):
      print(year)
      
      for scen in evaluating_scenario:
         print(scen)
         if scen == 'FB':
            curr_path_input = input_folder['historical_forcing']
            input_folder_ = f'{curr_path_input}/{year}'
         else:
            curr_path_input = input_folder['forcing_scenarios']
            input_folder_ = f'{curr_path_input}{scen}/{year}'
            
         X, y = create_Xy(input_folder_, year, run_name_format, label_index, inputs_static, inputs_dynamic,
                              output_feature, mask=mask, transforms=transforms)
         if first_day_to_consider is not None:
            X=X[first_day_to_consider:,:,:,:]  
            y=y[first_day_to_consider:,:]
         
         #beginning and end of evaluation, change start_day to start at a different day (e.g., to start at peak sm)
         start_day_ = 0
         end_day_ = X.shape[0]
         #matrix for storing observations of sm and predictions of sm
         out_obs = np.empty((end_day_-start_day_,ny,nx))
         out_pred = np.empty((end_day_-start_day_,ny,nx))
      
         # prediction from the model - initialize to the true output to begin with
         y_pred = np.copy(X[start_day_, label_index, :, :])
         create_files = 0
         
         for day_i in range(start_day_, end_day_):
            _X = X[day_i, :, :, :]  # take the input matrix for this day
            if use_previous_prediction:
               _X[label_index, ...] = y_pred  # but replace feature value with the last prediction that we have
            y_hat = y[day_i, :]
            __X = torch.from_numpy(_X[np.newaxis, ...]).type(torch.FloatTensor) # add time-axis in front of input
            __X = __X.type(torch.FloatTensor).to(DEVICE)
            _y_pred = model(__X).data.cpu().numpy().squeeze(axis=0) # remove time-axis from front of output
            pred_ = model(__X).squeeze()
            refer_ = torch.from_numpy(y_hat).to(DEVICE)
            pred_[~flattened_mask] = 0
            refer_[~flattened_mask] = 0
            loss = torch.nn.SmoothL1Loss()(pred_, refer_)
            
            y_hat = y_hat.reshape(ny, nx)
            _y_pred = _y_pred.reshape(ny, nx)
            # set predicted values outside the mask to 0
            _y_pred[~mask] = 0

            if output_feature in reverse_transforms:
               y_pred_unscaled = reverse_transforms[output_feature](y_pred)
               y_hat_unscaled = reverse_transforms[output_feature](y_hat)
            else:
               y_pred_unscaled = y_pred
               y_hat_unscaled = y_hat
            
            out_obs[day_i,:,:] = np.copy(y_hat_unscaled)
            out_pred[day_i,:,:] = np.copy(y_pred_unscaled)
            
            if day_i==start_day_: #first step in loop, create txt files to store performances
               y_initial = np.copy(y_hat_unscaled) #save the sm at t=0 for later
            
            file_path_ml = Path(f'{output_folder}/StatisticsEval_{scen}_{year}.txt')
            file_path_persist0 = Path(f'{output_folder}/StatisticsEvalPersist0_{scen}_{year}.txt')
            
            if create_files!=0:#file_path.is_file():
               file_stats_ml = open(file_path_ml, 'a')
               file_stats_persist0 = open(file_path_persist0,'a')
            else:
               file_stats_ml = open(file_path_ml, 'a')
               file_stats_persist0 = open(file_path_persist0,'a')
               file_stats_ml.write('year,day_input,RMSD,NSE,KGE\n')
               file_stats_persist0.write('year,day_input,RMSD,NSE,KGE\n')
               create_files=1
            #statistics of the CNN model prediction
            RMSD, NSE, KGE = compute_stats(y_hat_unscaled, y_pred_unscaled,mask)
            #statistics of the persistent case (assuming sm never changes)
            RMSDmean, NSEmean, KGEmean = compute_stats(y_hat_unscaled, y_initial,mask)
            
            if first_day_to_consider is None:
               curr_day_save = day_i
            else:
               curr_day_save = day_i+first_day_to_consider
            file_stats_ml.write(f'{year},{curr_day_save},{RMSD},{NSE},{KGE}\n')
            file_stats_persist0.write(f'{year},{curr_day_save},{RMSDmean},{NSEmean},{KGEmean}\n')
            
            if plots:
               plot(y_hat_unscaled, y_pred_unscaled, filepath=f'{output_folder}/images/{scen}_{year}_{day_i}.png')
         #saving pickle files of observed and predicted soil moisture
         if SAVE_PICKLE_SOILMOISTURE:
            pickle.dump(out_obs, open( f'{output_folder}/Observed_SM_{scen}_{year}.p', "wb" ) )
            pickle.dump(out_pred, open( f'{output_folder}/Predicted_SM_{scen}_{year}.p', "wb" ) )

def read_dynamic_inputs(t, data, which):
   """
   Read dynamic (changing in time) inputs for one timestep
   :param t: timestep (0-indexed) for which to read inputs
   :param data: A Run.data_accessor object
      A shallow copy of this object is made inside this function, making this function thread-safe.
   :param which: A str or iterable of strings, representing properties to read.
      One or more of:
         pressure
         saturation
         wtd
         DSWR/DLWR/APCP/Temp/UGRD/VGRD/Press/SPFH (i.e. any of the forcing variables)
         eflx_lh_tot/eflx_lwrad_out (i.e. any of the CLM output variables)
   :return: An ndarray of input variable values.
   """
   _valid_properties = ('pressure', 'saturation', 'wtd', 'soil_moisture') + FORCING_VARIABLES + CLM_OUTPUT_VARIABLES
   if isinstance(which, str):
      which = [which]
   for w in which:
      assert w in _valid_properties, f'Unrecognized property {w}'

   data = copy(data)
   run = data._run

   data.forcing_time = t
   data.time = t
   return_array = np.empty((len(which), run.ComputationalGrid.NY, run.ComputationalGrid.NX))

   for i, w in enumerate(which):
      if w in FORCING_VARIABLES:
         _array = data.clm_forcing(w)
      elif w in CLM_OUTPUT_VARIABLES:
         _array = data.clm_output(w, layer=-1)
      else:
         if w == 'soil_moisture':
            if  getattr(data, 'saturation').shape[0] < 1:
               print(t)
            _array = getattr(data, 'saturation') * getattr(data, 'computed_porosity')
         else:
            _array = getattr(data, w)

      # Convert 3D data to 2D by taking surface-level values
      if _array.ndim == 3:
         _array = _array[-1, ...]

      return_array[i, ...] = _array

   return return_array


def create_Xy(input_folder, year, run_name_format, label_index, inputs_static, inputs_dynamic, output_feature,
              t_start=0, t_end=None, mask=None, resolution_hours=24, transforms={}):

   #transforming inputs and labels, for the variables for which a transformation was indicated
   def _transform_Xy(X, y, output_feature):
      for i, feature in enumerate(inputs_dynamic, start=len(inputs_static)):
         if feature in transforms:
            X[i, :, :] = transforms[feature](X[i, :, :])
      
      if output_feature in transforms:
         y[:] = transforms[output_feature](y[:])

      return X, y

   if t_end is None:
      t_end = hours_in_year(year)

   n_hours = t_end - t_start
   n_timesteps = n_hours // resolution_hours

   runname = run_name_format.format(year=year)

   # setting path of the pfidb file
   path_PFdatabase = f'{input_folder}/{runname}.pfidb'

   # load the PF metadata and put it the run data structure
   run = Run.from_definition(path_PFdatabase)
   run.set_name(runname)

   # getting dimensions of domain
   nx = run.ComputationalGrid.NX
   ny = run.ComputationalGrid.NY

   data = run.data_accessor
   set_working_directory(f'{input_folder}/')

   X = np.empty((n_timesteps, len(inputs_static), ny, nx))

   for i, feature in enumerate(inputs_static):
      feature_values = getattr(data, feature)
      assert feature_values.ndim == 3, f'Expected 3D values for feature {feature}'

      feature_values = feature_values[-1, :, :]  # Take surface-level feature
      if feature in transforms:
            feature_values = transforms[feature](feature_values)
         
      X[:, i, :, :] = np.broadcast_to(feature_values, (n_timesteps, ny, nx))

   # For dynamic inputs, start at t_start = 1
   if t_start == 0:
      t_start = 1

   input_dynamic = np.zeros((n_hours, len(inputs_dynamic), ny, nx))
   future_map = {}  # mapping from Future objects to the timesteps that they will populate
   with ThreadPoolExecutor(max_workers=1) as executor:
      # The following loop creates N 'Future' objects (units of execution that may or may not have completed)
      # submits them all, and runs quickly.
      for i in range(t_start, t_end):
         future = executor.submit(read_dynamic_inputs, i, data, inputs_dynamic)
         future_map[future] = i

      for future in tqdm(as_completed(future_map), total=n_hours):
         # This block is entered once for every completed Future
         i = future_map[future]
         input_dynamic[i-t_start, :, :, :] = future.result()

   # Taking mean of all dynamic inputs over RESOLUTION_HOURS
   input_dynamic = input_dynamic.reshape(-1, resolution_hours, len(inputs_dynamic), ny, nx).mean(axis=1)

   X = np.concatenate([X, input_dynamic], axis=1)
   y = np.copy(X[:, label_index, :, :].reshape((n_timesteps, -1)))
      
   future_map = {}
   # Change max_workers to the maximum number of threads you want to use.
   with ThreadPoolExecutor(max_workers=1) as executor:
      # The following loop creates N 'Future' objects (units of execution that may or may not have completed)
      # submits them all, and runs quickly.
      for i in range(n_timesteps):
         future = executor.submit(_transform_Xy, X[i, :, :, :], y[i, :], output_feature)
         future_map[future] = i

      for future in tqdm(as_completed(future_map), total=n_timesteps):
         # This block is entered once for every completed Future
         i = future_map[future]
         X[i, :, :, :], y[i, :] = future.result()

   if mask is not None:
      X[:, :, ~mask] = 0
      y[:, ~mask.flatten()] = 0

   return X[:-1, ...], y[1:, ...]


# -----------------
# MAIN
# -----------------
if __name__ == '__main__':
   from random import randint
   from time import sleep

   if not os.path.exists(OUTPUT_FOLDER):
      os.makedirs(OUTPUT_FOLDER)
   
   file_years_train = open(f'{OUTPUT_FOLDER}/YearsTraining.txt','a') #storing which years were used for training
   for year in YEARS_TRAIN:
      file_years_train.write(f'{year}\n')
   
   if not os.path.exists(f'{OUTPUT_FOLDER}/images'):
      os.makedirs(f'{OUTPUT_FOLDER}/images')
   
   if TRAIN:
      print("*********** STARTING TRAINING ***********")
      train(
         inputs_dynamic=INPUT_DYNAMIC,
         inputs_static=INPUT_STATIC, 
         output_feature=OUTPUT_FEATURE,
         input_folder=INPUT_FOLDER,
         shuffle_days = SHUFFLING_DAYS,
         output_folder=OUTPUT_FOLDER,
         years=YEARS_TRAIN,
         filename_mask=f'{PATH_MASK}',
         run_name_format=RUNNAME_PARFLOW,
         path_models = PATH_MODELS,
         resolution_hours=TEMPORAL_RESOLUTION, #number of hours for resolution of data, if 24 is daily
         batch_size=BATCH_SIZE,
         learning_rate=10e-4,
         n_epoch=5000, #max number of epochs for the training (will be stopped earlier if early stopping criterion satisfied)
         scaling=SCALING,
         plots=PLOTS_TRAINING, #saving the plot of losses
         random_seed=RANDOM_SEED, #option to either impose a random seed (random_seed>0) or let it be chosen randomly (random_seed=0)
         cut_to_364 = CUT_TO_364, #option that cuts December 31st for leap years (so that all have 364 days)
         first_day_to_consider = FIRST_DAY_YEAR, #option to remove first FIRST_DAY_YEAR days of the year
         training_scenarios = TRAINING_SCENARIOS,
         years_val = YEARS_VAL #years used for validation (for early stopping of training)
     )

   if EVALUATE:
      DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
      print(f'evaluating on: {DEVICE}')
      print("*********** STARTING EVALUATION ***********")
      model = torch.load(f'{OUTPUT_FOLDER}/model.pik', map_location=DEVICE)
      model.load_state_dict(torch.load(f'{OUTPUT_FOLDER}/model.pth', map_location=DEVICE))
      
      print(f'Evaluating on: {TESTING_SCENARIOS}')
      
      evaluate(
         model,
         input_folder=INPUT_FOLDER,
         output_folder=OUTPUT_FOLDER,
         years=YEARS_TESTING,
         evaluating_scenario = TESTING_SCENARIOS,
         use_previous_prediction = USE_PREVIOUS_PREDICTION, #if true, using prediction of previous step for next prediction
         plots=PLOTS_TESTING,
         first_day_to_consider = FIRST_DAY_YEAR
      )
