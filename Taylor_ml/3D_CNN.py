# -----------------
# IMPORTS
# -----------------
import os
from copy import copy
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

This script trains a 3D CNN as outlined in Leonarduzzi et al.
The different options in the first section (`USER CHOICES/SETTINGS`) must be filled in with the specific choices or paths (Section 'FUNCTIONS').
The required changes are:
- INPUT_FOLDER: where the path to the historical forcing runs and the forcing scenarios are set (if you follow the suggested folder structure, just need to change path to Taylor_parflow)
- OUTPUT_FOLDER: absolute/relative path where you want your ML model and all outputs to be saved

It assumes
- ParFlow-CLM runs exist for all years and scenarios used
- that the CNN model is available ('RMM_NN_2D_B1.py' available on github in the `models` subfolder)
- that the mask for the Taylor catchment is available (`mask.pfb` available on github in the 'inputs' subfolder)
"""


# -----------------
# USER CHOICES/SETTINGS
# -----------------

PATH_TO_REPOSITORY = '/home/user/Desktop/git_repo/'

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
OUTPUT_FOLDER = f'3D_CNN_1/'

#path where the models are saved (must contain `RMM_NN_3D_A.py`)
PATH_MODELS = f'{PATH_TO_REPOSITORY}/Taylor_ml/models/'

#set this to true if you want to save pickle files of the losses in training and validation over the epochs
SAVE_PICKLE_LOSSES = True 

#set this to True if you want to save pickle files of the soil moisture of ParFlow-CLM (referred to as "obs.") and the CNN (referred to as "pred.")
SAVE_PICKLE_SOILMOISTURE = True

PATH_MASK = f'{PATH_TO_REPOSITORY}/Taylor_ml/inputs/mask.pfb' #path where the mask of the Taylor catchment is stored (available in the github repository)
TRAIN = True #if you want to train the model
EVALUATE = True #if you want to carry out testing (can be done only once the model has been trained)
FIRST_DAY_YEAR=0 #first day to consider. FIRST_DAY_YEAR days at the beginning of each year will be removed
#DEL USE_PREVIOUS_PREDICTION = True #If true recursive testing (i.e. output of ML used as input for next prediction), if False always using ParFlow input to predict next step
PLOTS_TRAINING = True #if True, saving the plot of the losses in training & validation as a function of the epochs
PLOTS_TESTING = False #if True, saving snapshots of labels and predictions

YEARS_TRAIN = [1988, 1990, 2015, 2016, 2018] #years for training of the CNN
YEARS_VAL = [2000, 2012, 2013] #years for validation (early stopping) of the CNN
YEARS_TESTING = [2002] #years for testing/evaluation

RUNNAME_PARFLOW = 'Taylor_{year}' #the runname used in your ParFlow-CLM runs (year will be replaced with the actual number)

CUT_TO_364 = True # set this to True if you want only the first 364 days to be considered (i.e., you throw away 30th of Sept.) -->helpful to control batch size

RANDOM_SEED = None #set to a number >0 if you want to choose the random seed (to get same results when repeating same experiment)

INPUT_DYNAMIC = ('qflx_infl', 'qflx_tran_veg', 'soil_moisture')
INPUT_STATIC = ('slope_x', 'slope_y','computed_porosity', 'computed_permeability_x')
OUTPUT_FEATURE = 'soil_moisture'

TEMPORAL_RESOLUTION = 24 #number of hours for resolution of data, if 24 is daily
BATCH_SIZE = 364 #batch size

TRAINING_SCENARIOS = ['FB','temp050_prec080','temp044_prec058','temp051_prec063','temp056_prec052','temp062_prec083','temp068_prec093','temp045_prec081','temp051_prec081','temp_025_prec090','temp065_prec065','temp080_prec065'] #list of scenarios to do the training on, for historical, use ['FB']

TESTING_SCENARIOS = ['temp060_prec058']

LEARNING_RATE = 10e-4
NUM_EPOCHS = 5000

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
          plots=False, random_seed=None, first_day_to_consider=None, cut_to_364=False,years_val=None,training_scenarios=None,n_epoch_loss=100):
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
   :param cut_to_364: whether to cut the last day of the year (to avoid having leep years one day longer)
   :param years_val: Year(s) for which to do validation (will determine early stopping)
   :param training_scenarios: scenarios to use in training (this is specific to where you stored your runs, will determine paths)
   :param n_epoch_loss: number of epochs over which to average the loss for early stopping (stop when mean over last n_epoch_loss is larger than antecedent n_epoch_loss)

   :return: None
   """
   DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
   print(f'training on: {DEVICE}')
      
   if random_seed is not None:
     torch.manual_seed(random_seed)
   
   n_channel = len(inputs_dynamic) + len(inputs_static)
   label_index = (inputs_static + inputs_dynamic).index(output_feature)
   
   mask = read_pfb(filename_mask).squeeze()  # 3D -> 2D
   mask = mask > 0  # boolean 2D mask
   
   n_timesteps = sum(hours_in_year(year) for year in years)
   n_timesteps = n_timesteps // resolution_hours

   ny, nx = mask.shape
   
   print(f'first_day: {first_day_to_consider}')
   if first_day_to_consider>1:
      N = 364-first_day_to_consider
      print(f"Cutting to {N} days")
   elif cut_to_364:
      N=364

   #setting up CNN model
   sys.path.append(f'{path_models}')
   from RMM_NN_3D_A import RMM_NN
   model = RMM_NN(grid_size=[N, ny, nx], channels=n_channel, verbose=True)
   
   model.to(DEVICE) #using GPU if available
   model.verbose = False
   model.use_dropout = True

   def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

   print(f'number of model parameters: {count_parameters(model)}')
   
   transforms = {k: float32_clamp_scaling(src_range=v[0], dst_range=v[1]) for k, v in scaling.items()}
   
   if training_scenarios is None:
      n_batches = len(years)
      n_batches_val = len(years_val)
   else:
      n_batches = len(years)*len(training_scenarios)
      n_batches_val = len(years_val)*len(training_scenarios)
   
   X = np.zeros((n_batches, n_channel, N, ny, nx)) #nchannel,N,ny,nx
   y = np.zeros((n_batches, N * ny * nx))

   curr_batch = 0
   #reading inputs/labels for training
   for year in years:
      print(year)
      for scen in training_scenarios:
         print(f'The scenario is: {scen}')
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
         if N<364:
            print(f'I am only using last {N} days in training')
            _X  = _X [_X .shape[0]-N:,:,:,:]
            _y = _y[_y.shape[0]-N:,:]

         #reshaping for 3D CNN input/label shapes
         for ch in range(_X.shape[1]):
            X[curr_batch,ch,:,:,:] = _X[:,ch,:,:]
            if ch==label_index:
               X[curr_batch,ch,:,:,:]=np.broadcast_to(X[curr_batch, ch, 0, :, :], (N, ny, nx))
         print(f'y_shape: {_y.shape}')
         y[curr_batch,:] = np.reshape(_y[:,:],N*ny*nx)
         X[curr_batch,:,:,:]
         curr_batch+=1

   curr_batch = 0
   if years_val is not None:
      x_val = np.zeros((n_batches_val, n_channel, N, ny, nx))
      y_val = np.zeros((n_batches_val, N * ny * nx))
      for year in years_val:
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
            if N<364:
               print(f'I am only using last {N} days in training')
               _X  = _X [_X .shape[0]-N:,:,:,:]
               _y = _y[_y.shape[0]-N:,:]
            
            for ch in range(_X.shape[1]):
               x_val[curr_batch,ch,:,:,:] = _X[:,ch,:,:]
               if ch==label_index:
                  x_val[curr_batch,ch,:,:,:]=np.broadcast_to(x_val[curr_batch, ch, 0, :, :], (N, ny, nx))
            y_val[curr_batch,:] = np.reshape(_y[:,:],N*ny*nx)
            x_val[curr_batch,:,:,:]
            curr_batch+=1
   print(f'shape X, training {X.shape}')
   print(f'shape y, training {y.shape}')
   print(f'shape x_val, training {x_val.shape}')
   print(f'shape y_val, training {y_val.shape}')
   
   # Attach useful information to the model object as attributes
   model.inputs_dynamic = inputs_dynamic
   model.inputs_static = inputs_static
   model.output_feature = output_feature
   model.filename_mask = filename_mask
   model.run_name_format = run_name_format
   model.scaling = scaling
   model.first_day_to_consider = first_day_to_consider
   model.cut_to_364 = cut_to_364
   
   os.makedirs(output_folder, exist_ok=True)

   mask_3d = np.zeros((N, ny, nx))
   for i in range(N):
      mask_3d[i,:,:] = mask
   
   flattened_mask = np.reshape(mask_3d,N*ny*nx)
   
   losses = np.empty(n_epoch)
   losses_val=np.empty(n_epoch)
   optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
   scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,learning_rate,total_steps=n_epoch)
   
   print("Starting the actual training (finished reading inputs)")
   for epoch in range(n_epoch):
      
      loss_train = 0
      for batch_ in range(X.shape[0]):
         X_tmp = np.zeros((1,X.shape[1],X.shape[2],X.shape[3],X.shape[4]))
         X_tmp[:,:,:,:,:]=X[batch_,:,:,:,:]
         _X = torch.from_numpy(X_tmp).type(torch.FloatTensor)
         _X = _X.type(torch.FloatTensor).to(DEVICE) 

         y_tmp = np.zeros((1,y.shape[1]))
         y_tmp[0,:] = y[batch_,:]
         _y = torch.from_numpy(y_tmp).type(torch.FloatTensor)
         if DEVICE != 'cpu' and torch.cuda.device_count()==2:
            _y = _y.type(torch.FloatTensor).to('cuda:1')
         else:
            _y = _y.type(torch.FloatTensor).to(DEVICE)
         
         model.train()
         optimizer.zero_grad()
         # Forward pass
         if DEVICE != 'cpu' and torch.cuda.device_count()==2:
            y_pred = model(_X).to('cuda:1')
         else:
            y_pred = model(_X)
         _y[:, flattened_mask==0] = 0
         
         y_pred[:, flattened_mask==0] = 0
         
         if DEVICE != 'cpu' and torch.cuda.device_count()==2:
            loss = torch.nn.SmoothL1Loss()(y_pred, _y).to('cuda:1')
         else:
            loss = torch.nn.SmoothL1Loss()(y_pred, _y)
         
         loss_train+=np.copy(loss.data.cpu().numpy())
         
         del y_pred
         del _y
         if torch.cuda.device_count()>1: #if two gpus are available, compute the losses on the 2nd one. this is useful if you're running into memory problems
            with torch.cuda.device('cuda:1'):
               torch.cuda.empty_cache()
         with torch.cuda.device('cuda:0'):
            torch.cuda.empty_cache() 
         
         loss.backward()
         del loss
         if torch.cuda.device_count()>1:
            with torch.cuda.device('cuda:1'):
               torch.cuda.empty_cache()
         with torch.cuda.device('cuda:0'):
            torch.cuda.empty_cache() 
         
         optimizer.step()
         #print("after step optimizer")
      losses[epoch] = loss_train / n_batches

      if years_val is not None:
         model.eval()
         loss_val = 0
         for batch_val in range(x_val.shape[0]):
            x_val_ = torch.from_numpy(x_val[batch_val:batch_val+1,:,:,:,:]).type(torch.FloatTensor)
            y_val_ = torch.from_numpy(y_val[batch_val:batch_val+1,:]).type(torch.FloatTensor)
            x_val_ = x_val_.type(torch.FloatTensor).to(DEVICE)
            y_val_ = y_val_.type(torch.FloatTensor).to(DEVICE)

            y_pred_val = model(x_val_)
            loss_val += (torch.nn.SmoothL1Loss()(y_pred_val, y_val_)).data.cpu().numpy()
            del y_pred_val
            del y_val_
            torch.cuda.empty_cache() 
         losses_val[epoch] = np.copy(loss_val) #prediction.data.cpu().numpy()
         
         print("Epoch: %3d, Train loss: %5.3e, Val loss: %5.3e" % (epoch, loss_train / n_batches,loss_val), end='\r')
         if epoch>n_epoch_loss*2-1:
            if np.mean(losses_val[epoch-n_epoch_loss:epoch+1])<np.mean(losses_val[epoch-n_epoch_loss*2:epoch-n_epoch_loss]):
               #still improving performances in validation
               torch.save(model, f'{output_folder}/model.pik')
               torch.save(model.state_dict(), f'{output_folder}/model.pth')
            else:
               print(f'Breaking loop here: {epoch}')
               print(f'The mean loss between epochs {epoch-n_epoch_loss} and {epoch+1} ({np.mean(losses_val[epoch-n_epoch_loss:epoch+1])}) is greater than between epochs {epoch-n_epoch_loss*2} and {epoch-n_epoch_loss} ({np.mean(losses_val[epoch-n_epoch_loss*2:epoch-n_epoch_loss])})')
               losses = losses[0:epoch+1]
               losses_val = losses_val[0:epoch+1]
               break
      else:
         print("Epoch: %3d, loss: %5.3e" % (epoch, loss), end='\r')
      scheduler.step()
      
   
   if plots:
      
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
   if SAVE_PICKLE_LOSSES:
      pickle.dump(losses, open( f'{output_folder}/Losses.p', "wb" ) )
      pickle.dump(losses_val, open( f'{output_folder}/LossesVal.p', "wb" ) )

def evaluate(model, input_folder, output_folder, years, plots=False, evaluating_scenario=['FB']):
   """
   Evaluate Model
   :param model: Model object, with weights pre-loaded
   :param input_folder: Folder with input data
   :param output_folder: Folder to save output
   :param years: Year(s) for which to run evaluation
   :param plots: boolean; whether to save .png files during evaluation
   :param first_day_consider: first day to
   :param evaluating_scenario: list of strings with names of the forcing scenarions to consider
   
   :return: None
   """
   
   print(f'Evaluating on 2: {evaluating_scenario}')
   # Retrieve useful information from the model object's attributes
   inputs_dynamic = model.inputs_dynamic
   inputs_static = model.inputs_static
   output_feature = model.output_feature
   filename_mask = model.filename_mask
   run_name_format = model.run_name_format
   scaling = model.scaling
   first_day_to_consider = model.first_day_to_consider
   cut_to_364 = model.cut_to_364
   
   transforms = {k: float32_clamp_scaling(src_range=v[0], dst_range=v[1]) for k, v in scaling.items()}
   reverse_transforms = {k: float32_clamp_scaling(src_range=v[1], dst_range=v[0]) for k, v in scaling.items()}
   
   label_index = (inputs_static + inputs_dynamic).index(output_feature)
   
   mask = read_pfb(filename_mask).squeeze()  # 3D -> 2D
   mask = mask > 0  # boolean 2D mask
   ny, nx = mask.shape
   if first_day_to_consider>1:
      N = 364-first_day_to_consider
   elif cut_to_364:
      N=364
   mask_3d = np.zeros((N, ny, nx))
   for i in range(N):
      mask_3d[i,:,:] = mask
   
   flattened_mask = np.reshape(mask_3d,N*ny*nx)
   
   model.use_dropout = False
   for parameter in model.parameters():
      parameter.requires_grad = False
   #first_day_to_consider = 150
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
      
         if cut_to_364:
               print('I am only using first 364 days in training')
               X = X[0:364,:,:,:]
               y = y[0:364,:]
         if N<364:
            print(f'I am only using last {N} days in training')
            X = X[X.shape[0]-N:,:,:,:]
            y = y[y.shape[0]-N:,:]
         
         X__ = np.zeros((1,X.shape[1],N,ny,nx))
         y__ = np.zeros((1,N*ny*nx))
         for ch in range(X.shape[1]):
            X__[0,ch,:,:,:] = X[:,ch,:,:]
            if ch==label_index:
               X__[0,ch,:,:,:]=np.broadcast_to(X__[0, ch, 0, :, :], (N, ny, nx))
         y__[0,:]=np.reshape(y,N*ny*nx)
         
         __X = torch.from_numpy(X__).type(torch.FloatTensor) # add time-axis in front of input
         __X = __X.type(torch.FloatTensor).to(DEVICE)
         y_pred = model(__X).data.cpu().numpy().squeeze(axis=0) # remove time-axis from front of output
         pred_ = model(__X).squeeze()
         refer_ = torch.from_numpy(y__).to(DEVICE).squeeze()
         print(list(pred_.size()))
         print(list(refer_.size()))
         pred_[flattened_mask==0] = 0
         refer_[flattened_mask==0] = 0
         y_hat = y__.reshape(N, ny, nx)
         y_pred = y_pred.reshape(N, ny, nx)

         file_path_ml = Path(f'{output_folder}/StatisticsEval_{scen}_{year}.txt')
         file_path_persist0 = Path(f'{output_folder}/StatisticsEvalPersist0_{scen}_{year}.txt')
      
         out_obs = np.empty(y_hat.shape)
         out_pred = np.empty(y_pred.shape)

         create_files=0
         for d in range(y_pred.shape[0]):
            d_pred = y_pred[d,:,:]
            d_hat = y_hat[d,:,:]

            if output_feature in reverse_transforms:
                  #print("doing reverse transforms")
                  d_pred_unscaled = reverse_transforms[output_feature](d_pred)
                  d_hat_unscaled = reverse_transforms[output_feature](d_hat)
            else:
                  d_pred_unscaled = d_pred
                  d_hat_unscaled = d_hat
            
            out_pred[d,:,:] = np.copy(d_pred_unscaled)
            out_obs[d,:,:] = np.copy(d_hat_unscaled)
            
            if d==0:
               d_initial = np.copy(d_hat_unscaled)
            
            if create_files!=0:#files already exist
               file_stats_ml = open(file_path_ml, 'a')
               file_stats_persist0 = open(f'{file_path_persist0}.txt','a')
            else:
               file_stats_ml = open(file_path_ml, 'w')
               file_stats_persist0 = open(f'{file_path_persist0}.txt','w')
               file_stats_ml.write('year,day_input,RMSD,NSE,KGE\n')
               file_stats_persist0.write('year,day_input,RMSD,NSE,KGE\n')
               create_files=1
            
            RMSD, NSE, KGE = compute_stats(d_hat_unscaled, d_pred_unscaled,mask)
            RMSDpersist0, NSEpersist0, KGEpersist0 = compute_stats(d_hat_unscaled, d_initial,mask)
            
            if first_day_to_consider is None:
               curr_day_save = d
            else:
               curr_day_save = d+first_day_to_consider
            file_stats_ml.write(f'{year},{curr_day_save},{RMSD},{NSE},{KGE}\n')
            file_stats_persist0.write(f'{year},{curr_day_save},{RMSDpersist0},{NSEpersist0},{KGEpersist0}\n')
            if plots:
               plot(d_hat_unscaled, d_pred_unscaled, filepath=f'{output_folder}/images/{scen}_{year}_{d}.png')
         
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
         inputs_dynamic=('qflx_infl', 'qflx_tran_veg', 'soil_moisture'),
         inputs_static=('slope_x', 'slope_y','computed_porosity', 'computed_permeability_x'),
         output_feature='soil_moisture',
         input_folder=INPUT_FOLDER,
         output_folder=OUTPUT_FOLDER,
         years=YEARS_TRAIN,
         filename_mask=PATH_MASK,
         run_name_format='Taylor_{year}',
         path_models=PATH_MODELS,
         resolution_hours=TEMPORAL_RESOLUTION,
         batch_size=BATCH_SIZE,
         learning_rate=LEARNING_RATE,
         n_epoch=NUM_EPOCHS,
         scaling=SCALING,
         plots=True,
         random_seed=RANDOM_SEED,
         cut_to_364 = CUT_TO_364,
         first_day_to_consider = FIRST_DAY_YEAR,#None, #FIRST_DAY_YEAR,
         training_scenarios = TRAINING_SCENARIOS,
         years_val = YEARS_VAL
     )

   if EVALUATE:
      DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
      print(f'evaluating on: {DEVICE}')
      print("*********** STARTING EVALUATION ***********")
      model = torch.load(f'{OUTPUT_FOLDER}/model.pik', map_location=DEVICE)
      model.load_state_dict(torch.load(f'{OUTPUT_FOLDER}/model.pth', map_location=DEVICE))
      
      evaluate(
         model,
         input_folder=INPUT_FOLDER,
         output_folder=OUTPUT_FOLDER,
         years=YEARS_TESTING,
         evaluating_scenario = TESTING_SCENARIOS,
         plots=False
      )
