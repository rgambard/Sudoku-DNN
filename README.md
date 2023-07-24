# Code du stage

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
