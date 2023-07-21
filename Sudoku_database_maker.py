# -*- coding: utf-8 -*-

import pandas as pd
import torch
import torch.nn as nn
import argparse
import time
import pickle

from Net import *
from utils import *
from PLL import *
from test_utils import *

file = open('../../Data_raw/one_of_many/sudoku_9_train_e.pkl', 'rb')
many_data = pickle.load(file)
n_sol = 10000;
grid_size = 9
good_grids = np.zeros((n_sol, grid_size,grid_size), dtype = np.uint8)
nfeatures = np.zeros((n_sol,grid_size, grid_size))
print("features done")
n_solt=0
file.close();
for n_grid in range(10000):
    for sudoku in many_data[n_grid]["target_set"]:
        good_grids[n_solt] = np.array(sudoku).astype(np.uint8).reshape(grid_size,grid_size)
        nfeatures[n_solt] = many_data[n_grid]["query"].reshape(grid_size,grid_size);
        n_solt+=1;
        if(n_solt==n_sol):break
    if(n_solt==n_sol):break


info = (grid_size**2, grid_size, 4)
data = (info,nfeatures, good_grids)

print("Writing data to disk...")
file = open("sudoku.pkl",'wb')
pickle.dump(data,file)
file.close()

