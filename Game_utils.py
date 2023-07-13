import torch
import Futoshi
import Sudoku
import numpy as np

class Futoshi_utils:
    def __init__(self, grid_size, device):
        self.grid_size = grid_size
        features = np.zeros((grid_size**2, grid_size**2, 5))
        li = np.linspace(0,1,grid_size)
        for x in range(grid_size):
            for y in range(grid_size):
                i=y*grid_size+x
                for x1 in range(grid_size):
                    for y1 in range(grid_size):
                        j=y1*grid_size+x1
                        features[i,j,0]=li[y]
                        features[i,j,1]=li[x]
                        features[i,j,2]=li[y1]
                        features[i,j,3]=li[x1]
        self.features = torch.Tensor(features).to(device)

    def make_features(self,infos):
        nfeatures =  self.features.unsqueeze(0).repeat(infos.shape[0],1,1,1);
        nfeatures[:,:,:,4] = torch.Tensor(infos[:,:,:])
        return nfeatures

    def check_valid(self, infos, Wb, target, unaryb, debug=1):
        fut = Futoshi.Futoshi(self.grid_size,ine=infos[0,:,:,4])
        Wb = np.trunc(Wb)
        fut.solve(Wb[0], unaryb[0], debug = (debug>1))
        valid = fut.check_validity()
        if debug>=1:
            print("Grid solved")
            print(fut)
            print("Solution valid : ", valid)
        return valid, fut

class Sudoku_utils:
    def __init__(self, grid_size, device):
        self.grid_size = grid_size
        features = np.zeros((grid_size**2, grid_size**2, 4))
        li = np.linspace(0,1,grid_size)
        for x in range(grid_size):
            for y in range(grid_size):
                i=y*grid_size+x
                for x1 in range(grid_size):
                    for y1 in range(grid_size):
                        j=y1*grid_size+x1
                        features[i,j,0]=li[y]
                        features[i,j,1]=li[x]
                        features[i,j,2]=li[y1]
                        features[i,j,3]=li[x1]
        self.features = torch.Tensor(features).to(device)

    def make_features(self,infos):
        nfeatures =  self.features.unsqueeze(0).repeat(infos.shape[0],1,1,1);
        return nfeatures

    def check_valid(self, infos, Wb, target, unaryb, debug=1):
        sud = Sudoku.Sudoku(self.grid_size)
        Wb = np.trunc(Wb)
        sud.solve(Wb[0], unaryb[0], debug = (debug>1))
        valid = sud.check_sudoku()
        if debug>=1:
            print("Grid solved")
            print(sud)
            print("Solution valid : ", valid)
        return valid, fut
