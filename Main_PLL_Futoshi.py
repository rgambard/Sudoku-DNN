# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import argparse
import time
import pickle
from tqdm import tqdm

import Solver
import Net
import EPLL_utils
import Game_utils
import multiprocessing as mp
import concurrent.futures

grad_save = 0
W_save = 0
nb_save= 0
def train_PLL(args, game_utils, device):
    print("\n \n \n TRAINING")
    # info = { number of variables in the problem, number of values these variables can take, number of features that are fed to the nn }
    # queries = np array of size (n_samples, n_var,n_var,n_infos_given_to_nn) ( fed to the nn )
    # target set = np array of size (n_samples, n_var) giving sample solutions 


    #### MODEL ####
            #create model
    model = Net.Net(game_utils.nb_var, game_utils.nb_val, game_utils.nb_features,
                hidden_size=args.hidden_size, 
                nblocks=args.nblocks)

    # print model size
    param_size = 0
    for param in model.parameters():
            param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))

    #load saved model if instructed to do so
    if args.saved_dict!="":
        print("loading model from disk", args.saved_dict)
        model.load_state_dict(torch.load(args.saved_dict,  map_location=device))

    #model = Net.VerySimpleNet(81,9,4, device)
    #instanciate optimizer and scheduler
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=args.lr, 
                                 weight_decay=args.weight_decay)

    #optimizer = torch.optim.SGD(model.parameters(), 
    #                             lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = args.scheduler_factor, patience = args.scheduler_patience)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = args.scheduler_factor, patience = args.scheduler_patience)


    print("\n Training with the following parameters: \n" 
            + str(args))

    print("\n \n")
    global grad_save, nb_save
    batch_size = args.batch_size
    #### TRAINING ####
    for epoch in range(1, args.epoch_max):
        print("EPOCH ", epoch, " on ", args.epoch_max)
            
    
        PLL_epoch,PLL1_epoch, loss_epoch, L1_epoch, unary_L1_epoch = 0, 0, 0, 0, 0
        model.train()
        data_iterator = game_utils.get_data() 
        grad_save = 0
        nb_save = 0
        for batch_idx in tqdm(range(args.train_size)):
        
            queries, target, infos = next(data_iterator)
            optimizer.zero_grad()
            NN_input = queries.to(device)  # bs, nb_var, nb_var, nb_feature
            y_true = target.type(torch.LongTensor).to(device) #bs,nb_var
            W, unary = model(NN_input, device, unary = args.unary)
            W = W
            #print(W[0,0,1,0],W[0,0,2,0])

            nb_var = W.shape[1]
            nb_val = W.shape[3]
            bs = W.shape[0]

            Wr = W.reshape(bs, nb_var, nb_var, nb_val,nb_val)
            L1 = torch.linalg.vector_norm(W, ord = 1)  # L1 penalty on predicted cost
            #L1 = torch.sum(torch.abs(W)*(torch.abs(W)>3))
            unary_L1 = torch.linalg.vector_norm(unary, ord=1)
            
            #PLL = -EPLL_utils.PLL_all(W, y_true, nb_neigh = args.k, hints_logit=unary)
            PLL1 = -EPLL_utils.PLL_all2(W, y_true, nb_neigh = args.k, hints_logit=unary)
            PLL=PLL1
            PLL_epoch += PLL.item()
            PLL1_epoch += PLL1.item()
            unary_L1_epoch += unary_L1.item()*args.reg_term_unary
            L1_epoch += L1.item()*args.reg_term
             
            def funcgrad(grad):
                global W_save, grad_save,nb_save
                #W_save += np.abs(W[0,torch.where(data[:,:,:,4]==-1)[0],torch.where(data[0,:,:,4]==-1)[1]].cpu().detach().numpy()).sum(axis=0) ; 
                
                #grad_save += np.abs(grad[:,[0,1],[2,3]].cpu().detach().numpy()).sum(axis=1).sum(axis=0);
                #grad_int = grad[0,torch.where(queries[0,0,:,4]==1)[0],torch.where(queries[0,0,:,4]==1)[1]].cpu().detach().numpy().sum(axis=0)
                #grad_int = grad[0,0,1].cpu().detach().numpy();
                #grad_int1 = grad[0,torch.where(queries[0,:,:,4]==-1)[1],torch.where(queries[0,:,:,4]==-1)[0]].cpu().detach().numpy().sum(axis=0).T
                #grad_int1 = grad[0,1,0].cpu().detach().numpy().T;
                """
                if queries[0,0,0,4]==1.0:
                    W_save += W[0,0,1]
                    grad_int = grad[0,0,:9].cpu().detach().numpy().sum(axis=0)
                    grad_save += grad_int#+grad_int1
                    """
                grad_save+=grad.detach().cpu().numpy()
                #nb_save += (y_true[0,0].detach().cpu()[None]==torch.arange(9)[:])
                #print(grad_save.round(4))
                #print(W[0,8,9].cpu().detach().numpy().round(1))
                #nb_sav += (y_true[1]==0)

            loss = PLL1 + args.reg_term * L1 + args.reg_term_unary * unary_L1
            #W.register_hook(lambda grad: print(" gradient !!!  \n",W[0,torch.where(data[0,:,:,4]==-1)[0],torch.where(data[0,:,:,4]==-1)[1]].round(), "\n" , grad[0,torch.where(data[0,:,:,4]==-1)[0],torch.where(data[0,:,:,4]==-1)[1]])) 
            #if batch_idx%200 == 1: 
            #    unary.register_hook(lambda grad: print(grad[0,0],"\n", unary[0,0]))
            #unary.register_hook(funcgrad)
            loss.backward()
            optimizer.step()
            #if batch_idx%200 == 2: 
            #    print(unary[0,0])
            loss_epoch += loss.item()
        optimizer.zero_grad()
        #print(W_save)
        #print(grad_save[0,0].round(2))
        #print(nb_save)
        #grad_save *= 0
        #print(nb_save)
        test_loss = 0
        # VALIDATION
        data_iterator = game_utils.get_data(validation=True) 
        for batch_idx in range(args.valid_size): 
            queries, target, infos = next(data_iterator)
            NN_input = queries.to(device)  # bs, nb_var, nb_var, nb_feature
            W, unary = model(NN_input, device, unary = args.unary)
            y_true = target.type(torch.LongTensor).to(device) #bs,nb_var
            PLL = -EPLL_utils.PLL_all2(W, y_true, hints_logit = unary)
            loss = PLL #+ args.reg_term * L1
            test_loss += loss.item()
            if(batch_idx == 0):
                print("DEBUG")
                
                #print(y_true[0].reshape(9,9)+1)
                #print(infos[0].reshape(9,9)+1)
                print(queries[0,3,23])
                print(W[0,3,4].cpu().detach().numpy().round(2))
                print(W[0,3,13].cpu().detach().numpy().round(2))
                print(W[0,3,23].cpu().detach().numpy().round(2))
                print(unary[0,3].cpu().detach().numpy().round(2))
                
                print(unary[0,0].cpu().detach().numpy().round(2))
                #print(np.trunc(unary[0,8].cpu().detach().numpy()))
                #print(np.trunc(unary[0,0].cpu().detach().numpy()))
                #print(np.trunc(unary[0,1].cpu().detach().numpy()))
                #print("FINdebug")
                Wb = W[0].cpu().detach().numpy()
                unaryd = unary[0].cpu().detach().numpy()
                unaryd = unaryd-unaryd.min(axis=1)[:,None]
                Wb = Wb-Wb.min(axis=-1).min(axis=-1)[:,:,None,None]
                Wb = Wb*(Wb>args.threshold)
                Solver.solver(Wb,unaryd,onlyDump=True)
                

        scheduler.step(loss_epoch)
        print(f'Training loss: {loss_epoch:.1f} ( =  PLL term : {PLL_epoch:.1f} + PLL1 : {PLL1_epoch:.1f} + L1 term : {L1_epoch:.1f} + unary_L1 : {unary_L1_epoch:.1f} || Testing loss = {test_loss:.1f}')
        print("current lr", optimizer.param_groups[0]['lr'], " ( min lr", args.min_lr," )")
        print("END OF EPOCH")
        if(optimizer.param_groups[0]['lr']<args.min_lr):
            print("lr is too small, stopping training")
            break

    print("\n END OF TRAINING \n")
    return model

def test(args, model, game_utils, device):
    print("\n \n \n TESTING \n")
    print("Test in progress (can take several minutes)")
    # query = np array of size (n_samples, n_var,n_var,n_infos) (given to the function make_feature which returns the features given to the nn for each pair of variables
    # and target set = np array of size (n_samples, n_var) giving sample solutions 
    
    results = []
            
    print("Starting parallel processes for solving: ")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        processes = []
        data_iterator = game_utils.get_data(test=True) 

        for batch_idx in tqdm(range(args.test_size)):
        
            queries, target, infos = next(data_iterator)
            NN_input = queries.to(device)  # bs, nb_var, nb_var, nb_feature
            y_true = target.type(torch.LongTensor).to(device)
            
            W, unary = model(NN_input, device, unary = args.unary)
            Wb = W[0].cpu().detach().numpy()
            unaryd = unary[0].cpu().detach().numpy()
            Wb = Wb*(Wb>args.threshold)
            #solve_parallel(game_utils.check_valid,queries[0].cpu().detach().numpy(),target[0].cpu().detach().numpy(), infos[0].cpu().detach().numpy() ,Wb, unaryd)
            proc = executor.submit(solve_parallel,game_utils.check_valid,queries[0].cpu().detach().numpy(),target[0].cpu().detach().numpy(), infos[0].cpu().detach().numpy() ,Wb, unaryd)
            processes.append(proc)
        print(" Collecting results ( be patient )")
        for process_idx in tqdm(range(args.test_size)):
            query, target, info, W, unary, res, game = processes[process_idx].result()
            if not res:
                breakpoint()
            results.append(res)

    result = np.array(results)
    print("Test done, games succesfully solved : ", np.sum(result),"on", args.test_size)
    print(results)
    return results

def solve_parallel (check_valid_ft, query, target, info, W, unary):
    ret, game = check_valid_ft(query, target, info, W, unary, debug=2)
    return query, target, info, W, unary, ret, game

def main():            

    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, prog='EPLL', description='learn a game rule representation using only valid exemples.')
    
    argparser.add_argument("--path_to_data", type=str, default="./databases/", help="path for loading training data") 
    argparser.add_argument("--game_type", type=str, default="Futoshiki", help="type of game ( Futoshiki, Sudoku )") 
    #with 1oM: ../Data_raw/one_of_many/
    argparser.add_argument("--hidden_size", type=int, default=128, help="width of hidden layers") 
    argparser.add_argument("--nblocks", type=int, default=5, help="number of blocks of 2 layers in ResNet") 
    argparser.add_argument("--epoch_max", type=int, default=200, help="maximum number of epochs") 
    argparser.add_argument("--lr", type=float, default=0.001, help="learning rate") 
    argparser.add_argument("--weight_decay", type=float, default=0.0001, help="weight_decay") 
    argparser.add_argument("--reg_term", type=float, default=1/10000, help="L1 regularization on costs")
    argparser.add_argument("--reg_term_unary", type=float, default=1/10000, help="L1 regularization on unary")
    argparser.add_argument("--unary", type=int, default=0, help="Use unary costs")
    argparser.add_argument("--k", type=int, default=5, help="E-PLL parameter") 
    argparser.add_argument("--batch_size", type=int, default=10, help="training batch size") 
    argparser.add_argument("--train_size", type=int, default=1000, help="number of training samples") 
    argparser.add_argument("--valid_size", type=int, default=64, help="number of validation samples") 
    argparser.add_argument("--test_size", type=int, default=70, help="number of test samples") 
    argparser.add_argument("--saved_dict", type=str, default="./model.pk", help="load weights saved from a previous execution") 
    argparser.add_argument("--save_path", type=str, default="./model_save.pk", help="location to which the weights are saved when training is over") 
    argparser.add_argument("--seed", type=int, default=np.random.randint(0, 2000), help="manual seed ( default = random )") 
    argparser.add_argument("--scheduler_factor", type=int, default=0.5, help="lr scheduler reducing factor when the loss does'nt improve") 
    argparser.add_argument("--scheduler_patience", type=int, default=5, help="lr scheduler patience") 
    argparser.add_argument("--min_lr", type=int, default=10e-6, help="min lr after which the training stops") 
    argparser.add_argument("--threshold", type=float, default=3, help="thresholding of the cost matrix to help the solver") 
 
    args = argparser.parse_args()    

    if torch.cuda.is_available():
        dev = "cuda:0"
        print("GPU connected")
    else:
        dev = "cpu"
        print("No GPU detected. Training on CPUs (be patient)")
    device = torch.device(dev)


    if args.game_type =="Futoshiki":
        game_utils = Game_utils.Futoshi_utils(train_size = args.train_size, validation_size = args.valid_size,
                                              test_size = args.test_size, batch_size = args.batch_size, path_to_data = args.path_to_data)
    elif args.game_type =="Sudoku":
        game_utils = Game_utils.Sudoku_utils(train_size = args.train_size, validation_size = args.valid_size,
                                              test_size = args.test_size, batch_size = args.batch_size, path_to_data = args.path_to_data)
    elif args.game_type =="Sudoku_hints":
        game_utils = Game_utils.Sudoku_hints_utils(train_size = args.train_size, validation_size = args.valid_size,
                                              test_size = args.test_size, batch_size = args.batch_size, path_to_data = args.path_to_data, device=device)
    elif args.game_type =="Sudoku_grounding":
        game_utils = Game_utils.Sudoku_grounding_utils(train_size = args.train_size, validation_size = args.valid_size,
                                              test_size = args.test_size, batch_size = args.batch_size, path_to_data = args.path_to_data, device=device)
    elif args.game_type =="Sudoku_visual":
        game_utils = Game_utils.Sudoku_visual_utils(train_size = args.train_size, validation_size = args.valid_size,
                                              test_size = args.test_size, batch_size = args.batch_size, path_to_data = args.path_to_data, device=device)




    torch.manual_seed(args.seed)
        ### TRAINING ###
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        model = train_PLL(args, game_utils, device)
        end.record()

        torch.cuda.synchronize()
        total_time = start.elapsed_time(end) #time in milliseconds

        print("Total training time: " + str(total_time)) 
    else:
        model = train_PLL(args, game_utils, device)

    if args.save_path!="":
        print("Training completed, saving model to disk...")
        PATH = args.save_path
        torch.save(model.state_dict(), PATH)
        print("model saved to",PATH)

    ### TESTING ###
    test(args, model, game_utils, device)
    print("Done, good bye !")
   
      
if __name__ == "__main__":
    main()
