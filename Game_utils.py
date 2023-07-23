import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import pickle
import Futoshi
import Sudoku
import numpy as np

class DataIterable:
    def __init__(self, queries, targets, batch_size, queries_transform_ft = None):
        self.queries = queries
        self.targets = targets
        self.batch_size = batch_size
        self.queries_transform_ft = queries_transform_ft
        self.index = 0
    def __iter__(self):
        return self
    def __next__(self):
        batch_size = self.batch_size
        if (self.index+1)*self.batch_size>self.targets.shape[0]:
            raise StopIteration  # signals "the end"
        infos =  self.queries[self.index*batch_size:(self.index+1)*batch_size]
        if self.queries_transform_ft is not None:
            queries = self.queries_transform_ft(infos)
        targets = self.targets[self.index*batch_size:(self.index+1)*batch_size]
        self.index+=1
        return queries, targets, infos


class Futoshi_utils:
    def __init__(self,  train_size = 500, validation_size = 100, test_size = 100, batch_size = 10, path_to_data = "databases/",device = "cpu" ):
        file = open(path_to_data+"futoshi.pkl",'rb')
        info, queries, targets=pickle.load(file)
        self.nb_var, self.nb_val, self.nb_features = info
        self.queries = torch.Tensor(queries)
        self.targets = torch.Tensor(targets-1).reshape(targets.shape[0], -1)


        self.batch_size = batch_size
        self.train_size = train_size
        self.validation_size = validation_size
        self.test_size = test_size
        

        grid_size = self.nb_val
        features = np.zeros((grid_size**2, grid_size**2, self.nb_features))
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
        self.features = torch.Tensor(features)

    def get_data(self, validation = False, test = False):
        queries = None
        targets = None
        train_size = self.train_size
        test_size = self.test_size
        validation_size = self.validation_size
        batch_size = self.batch_size

        if validation:
            queries = self.queries[train_size*batch_size:train_size*batch_size+validation_size*batch_size]
            targets = self.targets[train_size*batch_size:train_size*batch_size+validation_size*batch_size]
        elif test:
            queries = self.queries[train_size*batch_size+validation_size*batch_size:train_size*batch_size+validation_size*batch_size+test_size*batch_size]
            targets = self.targets[train_size*batch_size+validation_size*batch_size:train_size*batch_size+validation_size*batch_size+test_size*batch_size]
        else:
            queries = self.queries[:train_size*batch_size]
            targets = self.targets[:train_size*batch_size]

        return DataIterable(queries,targets,self.batch_size, queries_transform_ft = self.make_features)

    def make_features(self,infos):
        nfeatures =  self.features.unsqueeze(0).repeat(infos.shape[0],1,1,1);
        nfeatures[:,:,:,4] = torch.Tensor(infos[:,:,:])
        return nfeatures

    @staticmethod 
    def check_valid(query, target, info, W, unaryb =None, debug=1):
        fut = Futoshi.Futoshi(self.nb_val,ine=query[:,:,4])
        fut_sol = Futoshi.Futoshi(self.nb_val,ine=query[:,:,4])
        fut_sol.grid = target.reshape(self.nb_val,self.nb_val).astype(np.int8)+1
        Wb = W
        fut.solve(Wb, unaryb, debug = (debug>1))
        valid = fut.check_validity()
        if debug>=1:
            print("Grid solved")
            print(fut)
            print("Solution valid : ", valid)
        if debug>=1:
            print("SOLVER RETURNED")
            print("nonzero costs : ", W.nonzero()[0].shape[0])
            print("target cost : ",fut_sol.get_cost(W,unaryb))
            #return False, sudt
            print(fut_sol)
            print("solved valid ", valid, " cost : ", fut.get_cost(W,unaryb))
            print(fut)

        return valid, fut
    
    @staticmethod 
    def check_valid(query, target, info, W, unaryb =None, debug=1):
        grid_size = W.shape[3]
        sudt = Sudoku.Sudoku(grid_size)
        sudt.grid = target.reshape(grid_size,grid_size).astype(np.int8)+1

        sud = Sudoku.Sudoku(grid_size)
        sud.solve(W, unaryb, debug = (debug>1))
        if debug>=1:
            print("SOLVER RETURNED")
            print("nonzero costs : ", W.nonzero()[0].shape[0])
            print("target cost : ",sudt.get_cost(W,unaryb))
            #return False, sudt
            print(sudt)
            print("solved valid ", valid, " cost : ", sud.get_cost(W,unaryb))
            print(sud)
        return valid, sud



class Sudoku_utils:
    def __init__(self,  train_size = 500, validation_size = 100, test_size = 100, batch_size = 10, path_to_data = "databases/", device = "cpu" ):
        file = open(path_to_data+"sudoku.pkl",'rb')
        info, queries, targets=pickle.load(file)
        self.nb_var, self.nb_val, self.nb_features = info
        shuffle_index = torch.randperm(queries.shape[0])
        self.queries = torch.Tensor(queries)[shuffle_index]
        self.targets = torch.Tensor(targets-1).reshape(targets.shape[0], -1)[shuffle_index]


        self.batch_size = batch_size
        self.train_size = train_size
        self.validation_size = validation_size
        self.test_size = test_size

        grid_size = self.nb_val
        features = np.zeros((grid_size**2, grid_size**2, self.nb_features))
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
        self.features = torch.Tensor(features)

    def get_data(self, validation = False, test = False):
        queries = None
        targets = None
        train_size = self.train_size
        test_size = self.test_size
        validation_size = self.validation_size
        batch_size = self.batch_size

        if validation:
            queries = self.queries[train_size*batch_size:train_size*batch_size+validation_size*batch_size]
            targets = self.targets[train_size*batch_size:train_size*batch_size+validation_size*batch_size]
        elif test:
            queries = self.queries[train_size*batch_size+validation_size*batch_size:train_size*batch_size+validation_size*batch_size+test_size*batch_size]
            targets = self.targets[train_size*batch_size+validation_size*batch_size:train_size*batch_size+validation_size*batch_size+test_size*batch_size]
        else:
            queries = self.queries[:train_size*batch_size]
            targets = self.targets[:train_size*batch_size]

        return DataIterable(queries,targets,self.batch_size, queries_transform_ft = self.make_features)



    def make_features(self,infos):
        nfeatures =  self.features.unsqueeze(0).repeat(infos.shape[0],1,1,1);
        return nfeatures
    
    @staticmethod 
    def check_valid(query, target, info, W, unaryb =None, debug=1):
        grid_size = W.shape[3]
        sudt = Sudoku.Sudoku(grid_size)
        sudt.grid = target.reshape(grid_size,grid_size).astype(np.int8)+1

        sud = Sudoku.Sudoku(grid_size)
        sud.solve(W, unaryb, debug = (debug>1))
        valid = sud.check_sudoku()
        if debug>=1:
            print("SOLVER RETURNED")
            print("nonzero costs : ", W.nonzero()[0].shape[0])
            print("target cost : ",sudt.get_cost(W,unaryb))
            #return False, sudt
            print(sudt)
            print("solved valid ", valid, " cost : ", sud.get_cost(W,unaryb))
            print(sud)
        return valid, sud


class Sudoku_grounding_utils:
    def __init__(self,  train_size = 500, validation_size = 100, test_size = 80, batch_size = 10, path_to_data = "databases/", device = "cpu"):
        file = open(path_to_data+"sudoku.pkl",'rb')
        info, queries, targets=pickle.load(file)
        self.nb_var, self.nb_val, nb_features = info
        shuffle_index = torch.randperm(queries.shape[0])
        self.queries = torch.Tensor(queries)[shuffle_index]
        self.targets = torch.Tensor(targets-1).reshape(targets.shape[0], -1)[shuffle_index]



        self.nb_features = nb_features+2 # we also give to the nn the values of known digits, else 0
        self.batch_size = batch_size
        self.train_size = train_size
        self.validation_size = validation_size
        self.test_size = test_size
        

        grid_size = self.nb_val
        features = np.zeros((grid_size**2, grid_size**2, nb_features+2))
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
        self.features = torch.Tensor(features)

    def get_data(self, validation = False, test = False):
        queries = None
        targets = None
        train_size = self.train_size
        test_size = self.test_size
        validation_size = self.validation_size
        batch_size = self.batch_size

        if validation:
            queries = self.queries[train_size*batch_size:train_size*batch_size+validation_size*batch_size]
            targets = self.targets[train_size*batch_size:train_size*batch_size+validation_size*batch_size]
        elif test:
            queries = self.queries[train_size*batch_size+validation_size*batch_size:train_size*batch_size+validation_size*batch_size+test_size*batch_size]
            targets = self.targets[train_size*batch_size+validation_size*batch_size:train_size*batch_size+validation_size*batch_size+test_size*batch_size]
        else:
            queries = self.queries[:train_size*batch_size]
            targets = self.targets[:train_size*batch_size]

        return DataIterable(queries,targets,self.batch_size, queries_transform_ft = self.make_features)



    def make_features(self,infos):
        nfeatures =  self.features.unsqueeze(0).repeat(infos.shape[0],1,1,1);
        nfeatures[:,:,:,5]=infos.reshape(infos.shape[0],-1).unsqueeze(1)/9
        nfeatures[:,:,:,4]=infos.reshape(infos.shape[0],-1).unsqueeze(2)/9
        return nfeatures

    @staticmethod 
    def check_valid(query, target, info, W, unaryb =None, debug=1):
        grid_size = W.shape[3]
        sudt = Sudoku.Sudoku(grid_size)
        sudt.grid = target.reshape(grid_size,grid_size).astype(np.int8)+1

        sud = Sudoku.Sudoku(grid_size)
        sud.solve(W, unaryb, debug = (debug>1))
        if debug>=1:
            print("SOLVER RETURNED")
            print("nonzero costs : ", W.nonzero()[0].shape[0])
            print("target cost : ",sudt.get_cost(W,unaryb))
            #return False, sudt
            print(sudt)
            print("solved valid ", valid, " cost : ", sud.get_cost(W,unaryb))
            print(sud)
        return valid, sud





class Sudoku_grounding_utils1:
    def __init__(self,  train_size = 500, validation_size = 100, test_size = 80, batch_size = 10, path_to_data = "databases/", device = "cpu"):
        file = open(path_to_data+"sudoku.pkl",'rb')
        info, queries, targets=pickle.load(file)
        self.nb_var, self.nb_val, nb_features = info
        self.nb_val = self.nb_val+1
        shuffle_index = torch.randperm(queries.shape[0])
        self.queries = torch.Tensor(queries)[shuffle_index]
        self.targets = torch.Tensor(targets-1).reshape(targets.shape[0], -1)[shuffle_index]
        # on cache les indices
        self.targets[torch.where(self.queries.reshape(-1,self.nb_var)!=0)]=self.nb_val


        #self.nb_features = nb_features+2*(nb_val-1) # we also give to the nn the values of known digits, else 0
        self.nb_features = nb_features+2 # we also give to the nn the values of known digits, else 0
        self.batch_size = batch_size
        self.train_size = train_size
        self.validation_size = validation_size
        self.test_size = test_size
        

        grid_size = self.nb_val-1
        features = torch.zeros((self.nb_var, self.nb_var, nb_features+2))
        li = torch.linspace(0,1,grid_size)
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
        self.features = features

    def get_data(self, validation = False, test = False):
        queries = None
        targets = None
        train_size = self.train_size
        test_size = self.test_size
        validation_size = self.validation_size
        batch_size = self.batch_size

        if validation:
            queries = self.queries[train_size*batch_size:train_size*batch_size+validation_size*batch_size]
            targets = self.targets[train_size*batch_size:train_size*batch_size+validation_size*batch_size]
        elif test:
            queries = self.queries[train_size*batch_size+validation_size*batch_size:train_size*batch_size+validation_size*batch_size+test_size*batch_size]
            targets = self.targets[train_size*batch_size+validation_size*batch_size:train_size*batch_size+validation_size*batch_size+test_size*batch_size]
        else:
            queries = self.queries[:train_size*batch_size]
            targets = self.targets[:train_size*batch_size]

        return DataIterable(queries,targets,self.batch_size, queries_transform_ft = self.make_features)



    def make_features(self,infos):
        nfeatures =  self.features.unsqueeze(0).repeat(infos.shape[0],1,1,1);
        nfeatures[:,:,:,5]=infos.reshape(infos.shape[0],-1).unsqueeze(1)/9
        nfeatures[:,:,:,4]=infos.reshape(infos.shape[0],-1).unsqueeze(2)/9
        return nfeatures
        
    @staticmethod 
    def check_valid(query, target, info, W, unaryb =None, debug=1):
        print("nonzero costs : ", W.nonzero()[0].shape[0])
        grid_size = W.shape[3]
        sudt = Sudoku.Sudoku(grid_size)
        sudt.grid = target.reshape(grid_size,grid_size).astype(np.int8)+1

        sud = Sudoku.Sudoku(grid_size)
        sudh = Sudoku.Sudoku(grid_size)
        sud.solve(W, unaryb, debug = (debug>1))
        sudh.grid = sud.grid
        indexes_hints = np.where(info.reshape(grid_size,grid_size)!=0)
        sudh.grid[indexes_hints] = info[indexes_hints]
        valid = sudh.check_sudoku()

        if debug>=1:
            print("SOLVER RETURNED")
            print("nonzero costs : ", W.nonzero()[0].shape[0])
            print("target cost : ",sudt.get_cost(W,unaryb))
            #return False, sudt
            print(sudt)
            print("solved cost : ", sud.get_cost(W,unaryb))
            print(sud)
            print("solved sudoku with hints is valid ?", valid)
            print(sudh)
        return valid, sud



class Sudoku_grounding_utils2:
    def __init__(self,  train_size = 500, validation_size = 100, test_size = 80, batch_size = 10, path_to_data = "databases/", device = "cpu"):
        file = open(path_to_data+"sudoku.pkl",'rb')
        info, queries, targets=pickle.load(file)
        self.nb_var, self.nb_val, nb_features = info
        self.nb_features = nb_features+2*(self.nb_val) # we also give to the nn the values of known digits, else 0
        self.nb_val = self.nb_val+1
        shuffle_index = torch.randperm(queries.shape[0])
        self.queries = torch.Tensor(queries)[shuffle_index]
        self.targets = torch.Tensor(targets-1).reshape(targets.shape[0], -1)[shuffle_index]

        # on cache les indices
        self.targets[torch.where(self.queries.reshape(-1,self.nb_var)!=0)]=self.nb_val-1
        self.device = device


        #self.nb_features = nb_features+2*(nb_val-1) # we also give to the nn the values of known digits, else 0
        self.batch_size = batch_size
        self.train_size = train_size
        self.validation_size = validation_size
        self.test_size = test_size
        

        grid_size = self.nb_val-1
        features = torch.zeros((self.nb_var, self.nb_var, self.nb_features), device = device)
        li = torch.linspace(0,1,grid_size)
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
        self.features = features

    def get_data(self, validation = False, test = False):
        queries = None
        targets = None
        train_size = self.train_size
        test_size = self.test_size
        validation_size = self.validation_size
        batch_size = self.batch_size

        if validation:
            queries = self.queries[train_size*batch_size:train_size*batch_size+validation_size*batch_size]
            targets = self.targets[train_size*batch_size:train_size*batch_size+validation_size*batch_size]
        elif test:
            queries = self.queries[train_size*batch_size+validation_size*batch_size:train_size*batch_size+validation_size*batch_size+test_size*batch_size]
            targets = self.targets[train_size*batch_size+validation_size*batch_size:train_size*batch_size+validation_size*batch_size+test_size*batch_size]
        else:
            queries = self.queries[:train_size*batch_size]
            targets = self.targets[:train_size*batch_size]

        return DataIterable(queries,targets,self.batch_size, queries_transform_ft = self.make_features)



    def make_features(self,infos):
        nfeatures =  self.features.unsqueeze(0).repeat(infos.shape[0],1,1,1);
        infos = infos.to(self.device)
        one_hot_encode_hints = (infos.reshape(self.batch_size,-1)[:,:,None]==torch.arange(1,self.nb_val, device = self.device)[None,None,:])
        nfeatures[:,:,:,4:4+self.nb_val-1]=one_hot_encode_hints[:,:,None,:]
        nfeatures[:,:,:,4+self.nb_val-1:4+2*(self.nb_val-1)]=one_hot_encode_hints[:,None,:,:]
        return nfeatures
    
    @staticmethod 
    def check_valid(query, target, info, W, unaryb =None, debug=1):
        print("nonzero costs : ", W.nonzero()[0].shape[0])
        grid_size = W.shape[3]-1
        sudt = Sudoku.Sudoku(grid_size)
        sudt.grid = target.reshape(grid_size,grid_size).astype(np.int8)+1

        sud = Sudoku.Sudoku(grid_size)
        sudh = Sudoku.Sudoku(grid_size)
        sud.solve(W, unaryb, debug = (debug>1))
        sudh.grid = sud.grid
        indexes_hints = np.where(sud.grid==grid_size+1)
        sudh.grid[indexes_hints] = info[indexes_hints]
        valid = sudh.check_sudoku()

        if debug>=1:
            print("SOLVER RETURNED")
            print("nonzero costs : ", W.nonzero()[0].shape[0])
            print("target cost : ",sudt.get_cost(W,unaryb))
            #return False, sudt
            print(sudt)
            print("solved cost : ", sud.get_cost(W,unaryb))
            print(sud)
            print("solved sudoku with hints is valid ?", valid)
            print(sudh)
        return valid, sud


class Sudoku_visual_utils:
    def __init__(self,  train_size = 500, validation_size = 100, test_size = 80, batch_size = 10, path_to_data = "databases/", device = "cpu"):
        file = open(path_to_data+"sudoku.pkl",'rb')
        info, queries, targets=pickle.load(file)
        self.nb_var, self.nb_val, nb_features = info
        self.nb_features = nb_features+2*(self.nb_val) # we also give to the nn the values of known digits, else 0
        self.nb_val = self.nb_val+1
        shuffle_index = torch.randperm(queries.shape[0])
        self.queries = torch.Tensor(queries)[shuffle_index]
        self.targets = torch.Tensor(targets-1).reshape(targets.shape[0], -1)[shuffle_index]

        # on cache les indices
        self.targets[torch.where(self.queries.reshape(-1,self.nb_var)!=0)]=self.nb_val-1
        self.device = device


        #self.nb_features = nb_features+2*(nb_val-1) # we also give to the nn the values of known digits, else 0
        self.batch_size = batch_size
        self.train_size = train_size
        self.validation_size = validation_size
        self.test_size = test_size
        

        grid_size = self.nb_val-1
        features = torch.zeros((self.nb_var, self.nb_var, self.nb_features), device = device)
        li = torch.linspace(0,1,grid_size)
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
        self.features = features

        ### MNIST
        mnist_train_set = torchvision.datasets.MNIST('../Data_raw', train=True, download=True,
                                     transform=torchvision.transforms.Compose([
                                         torchvision.transforms.Resize((32, 32)),
                                         torchvision.transforms.ToTensor(),
                                         torchvision.transforms.Normalize(
                                         (0.1307,), (0.3081,))
                                     ]))

        mnist_test_set = torchvision.datasets.MNIST('../Data_raw', train=False, download=True,
                                     transform=torchvision.transforms.Compose([
                                         torchvision.transforms.Resize((32, 32)),
                                         torchvision.transforms.ToTensor(),
                                         torchvision.transforms.Normalize(
                                         (0.1307,), (0.3081,))
                                     ]))
        mnist_test_loader = torch.utils.data.DataLoader(mnist_test_set,
                                                         batch_size=1, 
                                                    shuffle=True)
        for idx, (img, label) in enumerate(mnist_train_set):
            img_table[label, np.argmin(img_table[label]!=-1)] = idx
                                                         
 

    def get_data(self, validation = False, test = False):
        queries = None
        targets = None
        train_size = self.train_size
        test_size = self.test_size
        validation_size = self.validation_size
        batch_size = self.batch_size

        if validation:
            queries = self.queries[train_size*batch_size:train_size*batch_size+validation_size*batch_size]
            targets = self.targets[train_size*batch_size:train_size*batch_size+validation_size*batch_size]
        elif test:
            queries = self.queries[train_size*batch_size+validation_size*batch_size:train_size*batch_size+validation_size*batch_size+test_size*batch_size]
            targets = self.targets[train_size*batch_size+validation_size*batch_size:train_size*batch_size+validation_size*batch_size+test_size*batch_size]
        else:
            queries = self.queries[:train_size*batch_size]
            targets = self.targets[:train_size*batch_size]

        return DataIterable(queries,targets,self.batch_size, queries_transform_ft = self.make_features)



    def make_features(self,infos):
        nfeatures =  self.features.unsqueeze(0).repeat(infos.shape[0],1,1,1);
        infos = infos.to(self.device)
        one_hot_encode_hints = (infos.reshape(self.batch_size,-1)[:,:,None]==torch.arange(1,self.nb_val, device = self.device)[None,None,:])
        nfeatures[:,:,:,4:4+self.nb_val-1]=one_hot_encode_hints[:,:,None,:]
        nfeatures[:,:,:,4+self.nb_val-1:4+2*(self.nb_val-1)]=one_hot_encode_hints[:,None,:,:]
        return nfeatures
    
    @staticmethod 
    def check_valid(query, target, info, W, unaryb =None, debug=1):
        print("nonzero costs : ", W.nonzero()[0].shape[0])
        grid_size = W.shape[3]-1
        sudt = Sudoku.Sudoku(grid_size)
        sudt.grid = target.reshape(grid_size,grid_size).astype(np.int8)+1

        sud = Sudoku.Sudoku(grid_size)
        sud.solve(W, unaryb, debug = (debug>1))
        indexes_hints = np.where(sud.grid==grid_size+1)
        sud.grid[indexes_hints] = info[indexes_hints]
        valid = sud.check_sudoku()

        if debug>=1:
            print("SOLVER RETURNED")
            print("nonzero costs : ", W.nonzero()[0].shape[0])
            print("target cost : ",sudt.get_cost(W,unaryb))
            #return False, sudt
            print(sudt)
            print("solved cost : ", sud.get_cost(W,unaryb))
            print(sud)
            print("solved sudoku with hints is valid ?", valid)
            print(sudh)
        return valid, sud
        #taken from https://towardsdatascience.com/implementing-yann-lecuns-lenet-5-in-pytorch-5e05a0911320
    class LeNet5(nn.Module):

        def __init__(self, n_classes):
            super(LeNet5, self).__init__()
            
            self.feature_extractor = nn.Sequential(            
                nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
                nn.Tanh(),
                nn.AvgPool2d(kernel_size=2),
                nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
                nn.Tanh(),
                nn.AvgPool2d(kernel_size=2),
                nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
                nn.Tanh()
            )

            self.classifier = nn.Sequential(
                nn.Linear(in_features=120, out_features=84),
                nn.Tanh(),
                nn.Linear(in_features=84, out_features=n_classes),
            )


        def forward(self, x):
            x = self.feature_extractor(x)
            x = torch.flatten(x, 1)
            logits = self.classifier(x)
            probs = F.softmax(logits, dim=1)

            return logits # probs



