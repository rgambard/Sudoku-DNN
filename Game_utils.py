import torch
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
        queries =  self.queries[self.index*batch_size:(self.index+1)*batch_size]
        if self.queries_transform_ft is not None:
            queries = self.queries_transform_ft(queries)
        targets = self.targets[self.index*batch_size:(self.index+1)*batch_size]
        self.index+=1
        return queries, targets


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
    def check_valid(infos, Wb, target, unaryb=None, debug=1):
        fut = Futoshi.Futoshi(self.nb_val,ine=infos[:,:,4])
        Wb = np.trunc(Wb)
        fut.solve(Wb, unaryb, debug = (debug>1))
        valid = fut.check_validity()
        if debug>=1:
            print("Grid solved")
            print(fut)
            print("Solution valid : ", valid)
        return valid, fut

class Sudoku_utils:
    def __init__(self,  train_size = 500, validation_size = 100, test_size = 100, batch_size = 10, path_to_data = "databases/", device = "cpu" ):
        file = open(path_to_data+"sudoku.pkl",'rb')
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
        return nfeatures
    
    @staticmethod 
    def check_valid(infos, W, target, unaryb =None, debug=1):
        sud = Sudoku.Sudoku(W.shape[3])
        W=W*(W>0.5)
        W = np.trunc(W*10)
        sud.solve(W, unaryb, debug = (debug>1))
        valid = sud.check_sudoku()
        if debug>=1:
            print("Grid solved")
            print(sud)
            print("Solution valid : ", valid)
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
    def check_valid(infos, W, target, unaryb =None, debug=1):
        W = W-W.min(axis=3).min(axis=2)[:,:,None,None]
        sud = Sudoku.Sudoku(W.shape[3])
        sud.solve(W, unaryb, debug = (debug>1))
        valid = sud.check_sudoku()
        if debug>=1:
            print("Grid solved")
            print(sud)
            print("Solution valid : ", valid)
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
    def check_valid(infos, W, target, unaryb =None, debug=1):
        W = W-W.min(axis=3).min(axis=2)[:,:,None,None]
        sud = Sudoku.Sudoku(W.shape[3])
        sud.solve(W, unaryb, debug = (debug>1))
        valid = sud.check_sudoku()
        if debug>=1:
            print("Grid solved")
            print(sud)
            print("Solution valid : ", valid)
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
    def check_valid(infos, W, target, unaryb =None, debug=1):
        W = W-W.min(axis=3).min(axis=2)[:,:,None,None]
        sud = Sudoku.Sudoku(W.shape[3])
        sud.solve(W, unaryb, debug = (debug>1))
        valid = sud.check_sudoku()
        if debug>=1:
            print("Grid solved")
            print(sud)
            print("Solution valid : ", valid)
        return valid, sud
