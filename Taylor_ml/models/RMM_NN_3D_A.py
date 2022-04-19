
"""
This 3D CNN model was downloaded from : https://github.com/HydroGEN-pubs/TV-ML/blob/main/HydroGEN/model_defs/RMM_NN_3D_A.py
reference (Maxwell et al., 2021): https://doi.org/10.3390/w13243633
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import sys
import numpy as np
import hydrogen.Shapes as shp

#Add Sand Tank path to the sys path
#sys.path.append('/Users/reed/Projects/HydroFrame-ML/pytorch_convolutional_rnn')
#import convolutional_rnn

from torch.nn.utils.rnn import pack_padded_sequence
# %%
## RMM 3D NN
#Define  the model
class RMM_NN(nn.Module):
    def __init__(self, grid_size=[100, 1, 50],  channels=2, verbose=False,):
        super(RMM_NN, self).__init__()
        self.use_dropout = False
        self.verbose = verbose

        # ---------------------------------------------------------------------
        # Inputs Parameters
        # ---------------------------------------------------------------------
        Cin = channels
        Din = grid_size[0]
        Hin = grid_size[1]
        Win = grid_size[2]
        in_shape = [Cin, Din, Hin, Win]

        # Convolution Layer 1 parameters
        Cout = 16
        Cout = 18
        cnv_kernel_size = 3
        cnv_kernel_size2 = 3
        cnv_stride2 = 1
        cnv_stride = 1
        cnv_padding = 1  # verify that it satisfies equation
        cnv_dilation = 1

        # Pooling Layer parameters
        pool_kernel_size = 2
        pool_stride = 2
        pool_padding = 0
        pool_dilation = 1

        # ---------------------------------------------------------------------
        # Layer 1 definition 
        # ---------------------------------------------------------------------
        self.layer1 = nn.Sequential(
            nn.Conv3d(
                in_channels= Cin,
                out_channels=Cout,
                kernel_size=cnv_kernel_size,
                stride=cnv_stride,
                padding=cnv_padding
            ).float(),
            #nn.LeakyReLU(),
            #nn.LogSoftmax(),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=Cout,
                out_channels=Cout,
                kernel_size=cnv_kernel_size,
                stride=cnv_stride,
                padding=cnv_padding
            ).float(),
            #nn.LeakyReLU(),
            nn.ReLU(),
            nn.MaxPool3d(
                kernel_size=pool_kernel_size,
                stride=pool_stride
            ),
            nn.Conv3d(
                in_channels=Cout,
                out_channels=Cout,
                kernel_size=cnv_kernel_size,
                stride=cnv_stride,
                padding=cnv_padding
            ).float(),
            # #nn.LeakyReLU(),
            nn.ReLU(),
            nn.MaxPool3d(
                kernel_size=pool_kernel_size,
                stride=pool_stride)
        )
        # set input elements zero with probability p = 0.5 (default)
        self.drop_out = nn.Dropout()

        # ---------------------------------------------------------------------
        # Shape calculations
        # ---------------------------------------------------------------------
        c3d_s1 = shp.conv3d_shape(input_shape=in_shape, cout=Cout, padding=cnv_padding,
                              kernel_size=cnv_kernel_size, dilation=cnv_dilation, stride=cnv_stride)
        c3d_s2 = shp.conv3d_shape(input_shape=c3d_s1, cout=Cout, padding=cnv_padding,
                              kernel_size=cnv_kernel_size, dilation=cnv_dilation, stride=cnv_stride)

        pool3d_s1 = shp.pool3d_shape(input_shape=c3d_s1, padding=pool_padding, kernel_size=pool_kernel_size,
                                 dilation=pool_dilation, stride=pool_stride)
        c3d_s3 = shp.conv3d_shape(input_shape=pool3d_s1, cout=Cout, padding=cnv_padding,
                              kernel_size=cnv_kernel_size, dilation=cnv_dilation, stride=cnv_stride)
        pool3d_s2 = shp.pool3d_shape(input_shape=c3d_s3, padding=pool_padding, kernel_size=pool_kernel_size,
                                 dilation=pool_dilation, stride=pool_stride)


        # ---------------------------------------------------------------------
        # Linear Steps
        # ---------------------------------------------------------------------
        L_Fin1 = int(np.prod(np.floor(pool3d_s2)))
        #L_Fin = (int(L_C)*int(L_Hin)*int(L_Win)*int(L_Din))
        L_Fout1 = 700  # dense layer size from NN w/ BC, very sensitive parameter, set to 100, 500, 1000
        
        self.dense = nn.Linear(L_Fin1, L_Fout1).float()

        L_Fin2 = L_Fout1
        L_Fout2 = int(Hin*Win*Din)
        self.out = nn.Linear(L_Fin2, L_Fout2).float()

        # ---------------------------------------------------------------------
        # Print Expected shapes
        # ---------------------------------------------------------------------
        if verbose:
            print("-- Model shapes --")
            print('Input Shape:', in_shape)
            print('Expected C3D Shape1:', c3d_s1)
            print('Expected Pool Shape:', pool3d_s1)
            print('Linear 1', L_Fin1, L_Fout1)
            print('Linear 2', L_Fin2, L_Fout2)

    def forward(self, x):
        out = self.layer1(x)
        if self.verbose: print('Step 1: Layer 1', out.shape)
        out = out.reshape(out.size(0), -1)
        if self.verbose: print('Step 2: Reshape', out.shape)
        if self.use_dropout:
            out = self.drop_out(out)
            if self.verbose: print('Step 2b: Dropout', out.shape)
        out = self.dense(out)
        if self.verbose: print('Step 3: Dense', out.shape)
        out = self.out(out)
        if self.verbose: print('Step 4: Linear Out', out.shape)
        return out 