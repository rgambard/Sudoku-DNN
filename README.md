# Code du stage

## Installation 

Clone the repository and install the requirements ( in a virtual env ) 

```
python -m pip install requirements.txt
```

## Usage 
```
usage: EPLL [-h] [--path_to_data PATH_TO_DATA] [--game_type GAME_TYPE]
            [--hidden_size HIDDEN_SIZE] [--nblocks NBLOCKS] [--lr LR]
            [--weight_decay WEIGHT_DECAY] [--reg_term REG_TERM]
            [--reg_term_unary REG_TERM_UNARY] [--unary UNARY] [--k K]
            [--order ORDER] [--nb_rand_masks NB_RAND_MASKS]
            [--nb_rand_tuples NB_RAND_TUPLES] [--batch_size BATCH_SIZE]
            [--epoch_max EPOCH_MAX] [--train_size TRAIN_SIZE]
            [--valid_size VALID_SIZE] [--test_size TEST_SIZE]
            [--saved_dict SAVED_DICT] [--save_path SAVE_PATH] [--seed SEED]
            [--scheduler_factor SCHEDULER_FACTOR]
            [--scheduler_patience SCHEDULER_PATIENCE] [--min_lr MIN_LR]
            [--threshold THRESHOLD] [--debug DEBUG]

learn a game rule representation using only valid exemples.

options:
  -h, --help            show this help message and exit
  --path_to_data PATH_TO_DATA
                        path for loading training data (default: ./databases/)
  --game_type GAME_TYPE
                        type of game ( Futoshiki, Sudoku, Sudoku_hints,
                        Sudoku_grounding, Sudoku_visual ) (default:
                        Sudoku_hints)
  --hidden_size HIDDEN_SIZE
                        width of hidden layers (default: 128)
  --nblocks NBLOCKS     number of blocks of 2 layers in ResNet (default: 5)
  --lr LR               learning rate (default: 0.001)
  --weight_decay WEIGHT_DECAY
                        weight_decay (default: 0)
  --reg_term REG_TERM   L1 regularization on costs (default: 0)
  --reg_term_unary REG_TERM_UNARY
                        L1 regularization on unary (default: 0)
  --unary UNARY         Use unary costs (default: 1)
  --k K                 E-PLL parameter (default: 40)
  --order ORDER         order of the PLL parameter ( mask width ) (default: 2)
  --nb_rand_masks NB_RAND_MASKS
                        stochastic PLL parameter : number of random masks
                        (default: 100)
  --nb_rand_tuples NB_RAND_TUPLES
                        stochastic PLL parameter : number of random tuple of
                        values to consider (default: 40)
  --batch_size BATCH_SIZE
                        training batch size (default: 8)
  --epoch_max EPOCH_MAX
                        maximum number of epochs (default: 200)
  --train_size TRAIN_SIZE
                        number of training samples (default: 1000)
  --valid_size VALID_SIZE
                        number of validation samples (default: 64)
  --test_size TEST_SIZE
                        number of test samples (default: 70)
  --saved_dict SAVED_DICT
                        load weights saved from a previous execution (default:
                        )
  --save_path SAVE_PATH
                        location to which the weights are saved when training
                        is over (default: ./model_save.pk)
  --seed SEED           manual seed ( default = random ) (default: 982)
  --scheduler_factor SCHEDULER_FACTOR
                        lr scheduler reducing factor when the loss does'nt
                        improve (default: 0.5)
  --scheduler_patience SCHEDULER_PATIENCE
                        lr scheduler patience (default: 5)
  --min_lr MIN_LR       min lr after which the training stops (default: 1e-05)
  --threshold THRESHOLD
                        thresholding of the cost matrix to help the solver
                        (default: 3)
  --debug DEBUG         verbose level (default: 0)
```
## Exemples de commandes
Pour effectuer le test avec le sudoku symbolique, en mode grounding : 

```
python3 Main.py --saved_dict "model_sudoku_grounding.pk" --save_path ""  --game_type Sudoku_grounding --epoch_max 0 --test_size 20 --threshold 4
```

Pour effectuer l'entrainement puis le test avec le sudoku visuel, en mode grounding :
``` 
python3 Main.py --save_path "model_sudoku_visual_grounding.pk" --game_type Sudoku_visual_grounding --epoch_max 1000
```

Pour effectuer l'entrainement puis le test avec le futoshiki :
``` 
python3 Main.py --save_path "model_futoshiki.pk" --game_type Futoshiki --k 10
```

## jeux implémentés ( argument game type ) :

```
Futoshiki, Sudoku, Sudoku_hints, Sudoku_grounding, Sudoku_visual, Sudoku_visual_grounding
```
