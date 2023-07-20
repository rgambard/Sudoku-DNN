import torch
import torch.nn.functional as F
import numpy as np
import pandas

import math
import timeit
from random import sample

rng = np.random.default_rng()
masks = None

def get_indexes_torch(y_true, nb_val,  masks, rand_y, masks_complementary):
    device = y_true.device
    bs, nb_masks, nb_var= y_true.shape
    bs, nb_masks,nb_rand_y, mask_width = rand_y.shape

    y_true_masked = y_true[torch.arange(bs, device = device)[:,None, None], torch.arange(nb_masks, device = device)[None,:,None], masks[:,:,:]]
    rand_y = y_true_masked[:,:,None,:] + rand_y
    rand_y = torch.fmod(rand_y,nb_val)
    

    nb_joint_indexes = math.comb(mask_width,2)
    nb_nonjoint_indexes = mask_width*(nb_var-mask_width)
    nb_indexes = nb_joint_indexes + nb_nonjoint_indexes
    final_indexes = torch.zeros((bs, nb_masks, nb_rand_y, nb_indexes, 5), dtype = torch.int8, device = device)

    #rand_y = rng.integers(0,nb_val,(bs,nb_mask, 1+nb_rand_y, mask_width))
    #rand_y[:,:,0] = y_true[np.arange(bs)[:,None,None],masks[None,:,:]]

    indexes = torch.zeros((bs,nb_masks,nb_rand_y,mask_width,nb_var-mask_width,4), dtype = torch.int8, device = device)
    diag_indexes = torch.zeros((bs,nb_masks,nb_rand_y,mask_width,mask_width,4), dtype = torch.int8, device = device)

    indexes[:,:,:,:,:,0] = masks[:,:,None,:,None]
    indexes[:,:,:,:,:,1] = masks_complementary[:,:,None,None,:]
    indexes[:,:,:,:,:,2] = rand_y[:,:,:,:,None]
    indexes[:,:,:,:,:,3] = y_true[torch.arange(bs)[:,None,None,None,None],torch.arange(nb_masks)[None,:,None,None,None],masks_complementary[:,:,None,None,:]]

    diag_indexes[:,:,:,:,:,0] = masks[:,:,None,:,None]
    diag_indexes[:,:,:,:,:,1] = masks[:,:,None,None,:]
    diag_indexes[:,:,:,:,:,2] = rand_y[:,:,:,:,None]
    diag_indexes[:,:,:,:,:,3] = rand_y[:,:,:,None,:]

    triangular_indices = torch.triu_indices(mask_width, mask_width, 1)
    joint_indexes = diag_indexes[:,:,:,triangular_indices[0],triangular_indices[1]]
    non_joint_indexes = indexes.reshape((bs,nb_masks,nb_rand_y,-1,4))

    final_indexes[:,:,:,:,1:] = torch.concatenate((joint_indexes,non_joint_indexes), axis=3)
    final_indexes[:,:,:,:,0] = torch.arange(bs)[:,None,None,None]

    return final_indexes


def init_global_variables(bs, nb_var, nb_val, device):
    global r_rand, masks, er_rand, masks_complementary
    #y_true = torch.randint(0,9,(bs,nb_var))
    triu = np.triu_indices(nb_var,1)
    masks = np.concatenate((triu[0][:,None],triu[1][:,None]),axis=1)
    masks = np.broadcast_to(masks[None,:,:],(bs,masks.shape[0],2))
    r_rand = np.zeros((nb_val,nb_val,2),dtype = np.int8)
    r_rand[:,:,0] = np.arange(nb_val)[:,None]
    r_rand[:,:,1] = np.arange(nb_val)[None,:]
    r_rand = r_rand.reshape((nb_val)**2,2)
    r_rand = np.broadcast_to(r_rand[None,None,:,:], (bs, masks.shape[1], r_rand.shape[0],r_rand.shape[1]))
    nb_val+=1
    er_rand = np.zeros((nb_val,nb_val,2),dtype = np.int8)
    er_rand[:,:,0] = np.arange(nb_val)[:,None]
    er_rand[:,:,1] = np.arange(nb_val)[None,:]
    er_rand = er_rand.reshape((nb_val)**2,2)
    er_rand = np.broadcast_to(er_rand[None,None,:,:], (bs, masks.shape[1], er_rand.shape[0],er_rand.shape[1]))

    masks = torch.from_numpy(np.array(masks)).to(device)
    bs, nb_masks, mask_width = masks.shape
    r_rand = torch.from_numpy(np.array(r_rand)).to(device)
    er_rand = torch.from_numpy(np.array(er_rand)).to(device)
    masks_complementary  = torch.where((masks[:,:,None,:]==torch.arange(nb_var, device = device)[None,None,:,None]).sum(axis=3)==0)[2].reshape(bs,nb_masks,-1) # si mask = [1,2], mask_complementary = [3,4,5,6,...], cad tous les indices qui ne sont pas modifiés

    
#r_ind = get_indexes_torch(y_true, nb_val, masks, r_rand)
def PLL_all2(W, y_true, nb_neigh = 0, T = 1, nb_rand_masks = 100,hints_logit = None):
    global r_rand, masks, er_rand, masks_complementary
    """
    Compute the total PLL loss over all variables and all batch samples

    Input: the predicted cost tensor W
           the true sequence y_true (tensor)
           the number of neighbours to mask (Gangster PLL parameter), int
           unary costs hints_logit (tensor, optional)

    Output: the PLL loss
    """
    device = y_true.device
    nb_var = W.shape[1]
    bs = W.shape[0]
    nb_val = W.shape[3]

    if masks is None:
        init_global_variables(bs,nb_var,nb_val, device)


    y_mod = ((y_true-1))[:,None,:].expand(bs,nb_rand_masks,nb_var).clone()
    rand_masks = torch.randint(0,masks.shape[1],(bs,nb_rand_masks), device = y_true.device)

    if nb_neigh != 0:
        # ajoût de la variable regulatrice
        nb_val = nb_val+1
        Wpad = torch.nn.functional.pad(W,(0,1,0,1))
        randindexes = torch.rand((bs,nb_rand_masks,nb_var-2), device = device).argsort(dim=-1)[...,:nb_neigh]
        randindexes = masks_complementary[torch.arange(bs)[:,None,None],rand_masks[:,:,None],randindexes]
        y_mod[torch.arange(bs)[:,None,None],torch.arange(nb_rand_masks)[None,:,None],randindexes] = nb_val-1
        W = Wpad


    if nb_neigh==0:
        ny_indices = get_indexes_torch(y_mod,nb_val,masks[np.arange(bs)[:,None],rand_masks],r_rand[np.arange(bs)[:,None],rand_masks], masks_complementary[np.arange(bs)[:,None],rand_masks])
    else:
        ny_indices = get_indexes_torch(y_mod,nb_val,masks[np.arange(bs)[:,None],rand_masks],er_rand[np.arange(bs)[:,None],rand_masks],masks_complementary[np.arange(bs)[:,None],rand_masks])
    tny_indices = ny_indices.int()
    Wr = W.reshape(bs, nb_var, nb_var, nb_val, nb_val)
    values_for_each_y = Wr[tny_indices[:,:,:,:,0],tny_indices[:,:,:,:,1],tny_indices[:,:,:,:,2], tny_indices[:,:,:,:,3],tny_indices[:,:,:,:,4]]
    cost_for_each_y = -torch.sum(values_for_each_y, axis = 3)
    log_cost = torch.logsumexp(cost_for_each_y, dim=2)

    PLL = torch.sum(cost_for_each_y[:,:,0])-torch.sum(log_cost)

    return(PLL)


def PLL_all(W, y_true, nb_neigh = 0, T = 1, hints_logit = None, perr = 0):

    """
    Compute the total PLL loss over all variables and all batch samples

    Input: the predicted cost tensor W
           the true sequence y_true (tensor)
           the number of neighbours to mask (Gangster PLL parameter), int
           unary costs hints_logit (tensor, optional)

    Output: the PLL loss
    """
    nb_var = W.shape[1]
    bs = W.shape[0]
    nb_val = W.shape[3]
    if perr > 0 :
        r = torch.rand(nb_var)
        thresh = r<perr
        rand_val = torch.randint(0,nb_val,(bs,thresh.sum().item(),)).to(y_true.device)
        y_true[:,thresh] = rand_val

    y_indices = (y_true-1).unsqueeze(-1).expand(bs,nb_var, nb_val).unsqueeze(1)
    Wr = W.reshape(bs, nb_var, nb_var, nb_val, nb_val)

    L_cost = Wr[torch.arange(bs)[:, None, None, None],
            torch.arange(nb_var)[None, :, None, None],
            torch.arange(nb_var)[None, None, :, None],
            torch.arange(nb_val)[None, None, None, :],
            y_indices]


    if nb_neigh > 0 : #number of neighbours to ignore

        samp = [sample([i for i in range(nb_var)], nb_neigh)for j in range(nb_var)] 
        samp = torch.tensor(samp).reshape(nb_var, -1)

        neigh = torch.ones_like(L_cost)
        neigh[:, torch.arange(nb_var)[:, None], samp] = 0
        L_cost *= neigh

    costs_per_value = torch.sum(L_cost, dim=2) 
    _, pred = torch.min(costs_per_value, -1)

    if hints_logit is not None:
        #breakpoint()
        costs_per_value += hints_logit
        pass
    lsm = F.log_softmax(-costs_per_value/T, dim=2)
    val_lsm = lsm[torch.arange(bs)[:, None],
            torch.arange(nb_var)[None, :], y_true-1]

    return(torch.sum(val_lsm))


def pred_all_var(W, y):

    """
    Predict the value for each variables given all other variables.
    Input: the cost matrix W
           the true sequence y_true
   Output: the predicted values and their associated cost
    """

    nb_var = W.shape[1]
    nb_val = int(nb_var**0.5)
    bs = W.shape[0] 

    y_indices = (y-1).unsqueeze(-1).expand(bs,nb_var, nb_val).unsqueeze(1)
    Wr = W.reshape(bs, nb_var, nb_var, nb_val, nb_val)
    L_cost = Wr[torch.arange(bs)[:, None, None, None],
            torch.arange(nb_var)[None, :, None, None],
            torch.arange(nb_var)[None, None, :, None],
            torch.arange(nb_val)[None, None, None, :],
            y_indices]

    costs_per_value = torch.sum(L_cost, dim=2)
    _, idx = torch.min(costs_per_value, dim=2)

    # PLL
    lsm = F.log_softmax(-costs_per_value, dim=2)
    PLLs = lsm[torch.arange(bs)[:, None],
            torch.arange(nb_var)[None, :], y-1]
    return(torch.stack([(idx+1), PLLs]))


def val_metrics(W, targets):

    """return the number of correctly predicted values and the value of -PLL"""

    best_acc = 0
    for target in targets:
        y_pred, PLL = pred_all_var(W, target)
        target_acc = torch.sum((target-y_pred) == 0)
        target_nPLL = -torch.sum(PLL)

        if target_acc > best_acc:
            best_acc = target_acc
            nPLL = target_nPLL

    return(best_acc, nPLL)
