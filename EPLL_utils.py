import torch
import torch.nn.functional as F
import numpy as np
import pandas

import math
import timeit
from random import sample

def get_indexes_torch(y_true, nb_val,  masks, rand_y, masks_complementary, masks_d2=None, masks_complementary_d2=None):
    device = y_true.device
    bs, nb_masks, nb_var= y_true.shape
    bs, nb_masks,nb_rand_y, mask_width = rand_y.shape
    bs, nb_masks, nb_complementary = masks_complementary.shape
    if masks_d2 is None:
        masks_d2 = masks
        masks_complementary_d2 = masks_complementary

    y_true_masked = y_true[torch.arange(bs, device = device)[:,None, None], torch.arange(nb_masks, device = device)[None,:,None], masks[:,:,:]]
    rand_y = y_true_masked[:,:,None,:] + rand_y
    rand_y = torch.fmod(rand_y,nb_val)
    

    nb_joint_indexes = math.comb(mask_width,2)
    nb_nonjoint_indexes = mask_width*(nb_complementary)
    nb_indexes = nb_joint_indexes + nb_nonjoint_indexes
    final_indexes = torch.zeros((bs, nb_masks, nb_rand_y, nb_indexes, 5), dtype = torch.int, device = device)

    #rand_y = rng.integers(0,nb_val,(bs,nb_mask, 1+nb_rand_y, mask_width))
    #rand_y[:,:,0] = y_true[np.arange(bs)[:,None,None],masks[None,:,:]]

    indexes = torch.zeros((bs,nb_masks,nb_rand_y,mask_width,nb_complementary,4), dtype = torch.int, device = device)
    diag_indexes = torch.zeros((bs,nb_masks,nb_rand_y,mask_width,mask_width,4), dtype = torch.int, device = device)

    indexes[:,:,:,:,:,0] = masks[:,:,None,:,None]
    indexes[:,:,:,:,:,1] = masks_complementary_d2[:,:,None,None,:]
    indexes[:,:,:,:,:,2] = rand_y[:,:,:,:,None]
    indexes[:,:,:,:,:,3] = y_true[torch.arange(bs, device = device)[:,None,None,None,None],torch.arange(nb_masks, device = device)[None,:,None,None,None],masks_complementary[:,:,None,None,:]]

    diag_indexes[:,:,:,:,:,0] = masks[:,:,None,:,None]
    diag_indexes[:,:,:,:,:,1] = masks_d2[:,:,None,None,:]
    diag_indexes[:,:,:,:,:,2] = rand_y[:,:,:,:,None]
    diag_indexes[:,:,:,:,:,3] = rand_y[:,:,:,None,:]

    triangular_indices = torch.triu_indices(mask_width, mask_width, 1, device=device)
    joint_indexes = diag_indexes[:,:,:,triangular_indices[0],triangular_indices[1]]
    non_joint_indexes = indexes.reshape((bs,nb_masks,nb_rand_y,-1,4))

    final_indexes[:,:,:,:,1:] = torch.concatenate((joint_indexes,non_joint_indexes), axis=3)
    final_indexes[:,:,:,:,0] = torch.arange(bs, device = device)[:,None,None,None]

    return final_indexes

def get_random_perms(nb_val, masks, device, nb_rand_perms=20): 
    bs, nb_masks, mask_width = masks.shape
    if mask_width == 1:
        all_perms = torch.zeros((nb_val,mask_width),dtype = torch.int8, device = device)
        all_perms[:,0] = torch.arange(nb_val, device = device)[:]
    if mask_width ==  2:
        all_perms = torch.zeros((nb_val,nb_val,mask_width),dtype = torch.int8, device = device)
        all_perms[:,:,0] = torch.arange(nb_val, device = device)[:,None]
        all_perms[:,:,1] = torch.arange(nb_val, device = device)[None,:]
    elif mask_width == 3:
        all_perms = torch.zeros((nb_val,nb_val,nb_val,mask_width),dtype = torch.int8, device = device)
        all_perms[:,:,:,0] = torch.arange(nb_val, device = device)[:,None, None]
        all_perms[:,:,:,1] = torch.arange(nb_val, device = device)[None,:, None]
        all_perms[:,:,:,2] = torch.arange(nb_val, device = device)[None,None,:]
    elif mask_width == 4:
        all_perms = torch.zeros((nb_val,nb_val,nb_val,nb_val,mask_width),dtype = torch.int8, device = device)
        all_perms[:,:,:,:,0] = torch.arange(nb_val, device = device)[:,None, None,None]
        all_perms[:,:,:,:,1] = torch.arange(nb_val, device = device)[None,:, None,None]
        all_perms[:,:,:,:,2] = torch.arange(nb_val, device = device)[None,None,:,None]
        all_perms[:,:,:,:,3] = torch.arange(nb_val, device = device)[None,None,None,:]

    all_perms = all_perms.reshape((nb_val)**mask_width,mask_width)
    nb_tot_perm, mask_width = all_perms.shape
    rand_perms_indexes = torch.rand((bs,nb_masks,nb_tot_perm),device = device)
    rand_perms_indexes[:,:,0]=-1 # so that we always select the identity first
    rand_perms_indexes = rand_perms_indexes.argsort(dim=-1)[...,:nb_rand_perms]

    rand_perms = all_perms[rand_perms_indexes]
    return rand_perms
    
def new_PLL(W,idx_pairs,y_true, nb_neigh = 0, T=1,   nb_rand_masks = 500, nb_rand_perms=200, mask_width = 2 ,unary_costs= None,missing=None, val=False):
    if val is not False:
        print("val not implemented !!!")
    nb_val = int(W.shape[2]**0.5)
    W = W.reshape(1, W.shape[0], W.shape[1],nb_val, nb_val)
    y_true = y_true.unsqueeze(0)
    idx_pairs = idx_pairs.unsqueeze(0)
    return PLL_all2(W, y_true, nb_neigh = nb_neigh, T=T,   nb_rand_masks = nb_rand_masks, nb_rand_perms=nb_rand_perms, mask_width = mask_width ,hints_logit = unary_costs, idx_pairs =idx_pairs)

#r_ind = get_indexes_torch(y_true, nb_val, masks, r_rand)
def PLL_all2(W, y_true, nb_neigh = 0, T = 1, nb_rand_masks = 100, nb_rand_perms=30, mask_width = 2 ,hints_logit = None, idx_pairs = None):
    #global r_rand, masks, er_rand, masks_complementary
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

    ### ATTENTION -1
    y_mod = y_true[:,None,:].expand(bs,nb_rand_masks,nb_var).clone() #### !!!!! -1
    #rand_masks = torch.randint(0,masks.shape[1],(bs,nb_rand_masks), device = y_true.device)

    #generation aleatoire des masques
    if idx_pairs is None:
        rand_indexes = torch.rand((bs, nb_rand_masks, nb_var), device = y_true.device).argsort(dim=-1)
        masks = rand_indexes[...,:mask_width]
        masks_complementary = rand_indexes[...,mask_width:]
        masks_complementary_indexes = masks_complementary
        masks_d2 = masks
        masks_complementary_d2 = masks_complementary
    else:
        nb_pairs = idx_pairs.shape[2]
        rand_indexes = torch.rand((bs, nb_rand_masks, nb_var), device = device).argsort(dim=-1)
        masks = torch.zeros((bs,nb_rand_masks, mask_width), dtype = torch.int, device = device)
        masks[:,:,0] = rand_indexes[...,0]
        rand_indexes_neigh = torch.rand((bs, nb_rand_masks, nb_pairs), device = device).argsort(dim=-1)
        masks[:,:,1:] = idx_pairs[torch.arange(bs, device = device)[:,None,None], masks[:,:,:1],rand_indexes_neigh[...,:mask_width-1]]
        masks_complementary = idx_pairs[torch.arange(bs, device = device)[:,None,None],masks[:,:,:1] ,rand_indexes_neigh[...,mask_width-1:]]
        masks_complementary_d2 = rand_indexes_neigh[...,mask_width-1:]
        masks_d2 = torch.nn.functional.pad(rand_indexes_neigh[...,:mask_width-1],(1,0), value = 0)

    nb_complementary = masks_complementary.shape[2]


    if hints_logit is not None:# hints_logit is not None:
        #breakpoint()
        nb_var = nb_var+1 # add 1 additionnal variable that represents hints
        Wpad = torch.nn.functional.pad(W,(0,0,0,0,0,1))
        masks_complementary_d2 = torch.nn.functional.pad(masks_complementary_d2,(0,1),value = nb_complementary)
        masks_complementary = torch.nn.functional.pad(masks_complementary,(0,1),value = nb_var-1)
        y_mod = torch.nn.functional.pad(y_mod,(0,1),value = 0)
        Wpad[:,:,nb_var-1,:,0] = hints_logit[:,:]
        W = Wpad

    if nb_neigh != 0:
        # ajout de la valeur regulatrice, de coût 0
        Wpad = torch.nn.functional.pad(W,(0,1))
        randindexes = torch.rand((bs,nb_rand_masks,nb_complementary), device = device).argsort(dim=-1)[...,:nb_neigh]
        randindexes = masks_complementary[torch.arange(bs, device = device)[:,None,None],torch.arange(nb_rand_masks, device = device)[None,:,None],randindexes]
        y_mod[torch.arange(bs, device = device)[:,None,None],torch.arange(nb_rand_masks, device = device)[None,:,None],randindexes] = nb_val
        W = Wpad



    #if nb_neigh==0:
    #    ny_indices = get_indexes_torch(y_mod,nb_val,masks[np.arange(bs)[:,None],rand_masks],r_rand[np.arange(bs)[:,None],rand_masks], masks_complementary[np.arange(bs)[:,None],rand_masks])
    #else:
    rand_perms = get_random_perms(nb_val, masks, device, nb_rand_perms = nb_rand_perms)
    ny_indices = get_indexes_torch(y_mod,nb_val,masks,rand_perms,masks_complementary,masks_d2, masks_complementary_d2)
    tny_indices = ny_indices.int()
    values_for_each_y = W[tny_indices[:,:,:,:,0],tny_indices[:,:,:,:,1],tny_indices[:,:,:,:,2], tny_indices[:,:,:,:,3],tny_indices[:,:,:,:,4]]
    cost_for_each_y = -torch.sum(values_for_each_y, axis = 3)
    log_cost = torch.logsumexp(cost_for_each_y, dim=2)

    PLL = torch.sum(cost_for_each_y[:,:,0]-log_cost)

    return(PLL)


def PLL_all(W, y_true, nb_neigh = 0, T = 1, hints_logit = None):

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
    y_indices = (y_true).unsqueeze(-1).expand(bs,nb_var, nb_val).unsqueeze(1)
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
            torch.arange(nb_var)[None, :], y_true]

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
