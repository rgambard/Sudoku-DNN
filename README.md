Pour effectuer le test avec le sudoku symbolique, en mode grounding : 
python3 Main_PLL_Futoshi.py --saved_dict "model_sudoku_grounding.pk" --save_path ""  --game_type Sudoku_grounding --epoch_max 0 --test_size 20 --threshold 4
Pour effectuer l'entrainement avec le sudoku visuel, en mode grounding :
 
python3 Main_PLL_Futoshi.py --save_path "model_sudoku_visual_grounding.pk" --game_type Sudoku_visual_grounding --epoch_max 1000
