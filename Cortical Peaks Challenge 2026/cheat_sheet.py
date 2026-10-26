"""
Copy-paste terminal commands, one block per game. Always run in order:

Terminal 1 -> Terminal 2 -> Terminal 3 (server must be up before the
pipeline starts predicting, and the pipeline must be streaming before
the connector looks for it).

BEFORE trusting a new model+game pairing for the first time: check
Terminal 3's startup line "Class order: [...]" and confirm every class
name in it is a key in CLASS_TO_ACTION[<that game>] inside
game_connector_2.py. If it isn't, predictions get silently dropped
(no error, no CMD sent) -- this already happened once with jump_only.

============================================================================
SERVER (same every time, run first)
============================================================================

# Tryout / local testing:
python fake_server.py

# Real competition server:
python server.py ??????
# TODO: no __main__ entrypoint found in the server.py we have -- probably
# lives in a main.py at the repo root that also needs the games/ package.
# ASK Markus 

#since all of the following commands are just the replay all of this is just 
# for trying out now 
#drop the replay part for the competition

============================================================================
DINO JUMP -- BrainSki 1, Jump Only (BSJ)         
============================================================================

# Terminal 2:
python online_pipeline.py --model models/dino_jump.joblib --replay-file /home/micha/Projects/online/recordings/session1_run1.xdf --speed 20

# Terminal 3:
python game_connector.py --model models/dino_jump.joblib --game dino_jump

============================================================================
DINO -- BrainSki 1, Jump & Duck (BSDJ)             
============================================================================

# Terminal 2:
python online_pipeline.py --model models/dino.joblib --replay-file /home/micha/Projects/online/recordings/session1_run1.xdf --speed 20

# Terminal 3:
python game_connector.py --model models/dino.joblib --game dino

============================================================================
SKI -- BrainSki 2, Static (SK1)                   
============================================================================

# Terminal 2:
python online_pipeline.py --model models/brainski.joblib --replay-file /home/micha/Projects/online/recordings/session2_run2.xdf --speed 20

# Terminal 3:
python game_connector.py --model models/brainski.joblib --game ski

============================================================================
SKI DYN -- BrainSki 2, Dynamic (SK2)               
============================================================================

# Terminal 2:
python online_pipeline.py --model models/brainski.joblib --replay-file /home/micha/Projects/online/recordings/session2_run2.xdf --speed 20

# Terminal 3:
python game_connector.py --model models/brainski.joblib --game ski_dyn

============================================================================
PONG (dino works here cause we use the same commands, just here right hand - up, feet - down. The game connector will map the commands to the correct inputs for the game)                                  
============================================================================

# Terminal 2:
python online_pipeline.py --model models/dino.joblib --replay-file /home/micha/Projects/online/recordings/session2_run2.xdf --speed 20

# Terminal 3:
python game_connector.py --model models/dino.joblib --game pong

============================================================================
PONG AI                                            
============================================================================

# Terminal 2:
python online_pipeline.py --model models/dino.joblib --replay-file /home/micha/Projects/online/recordings/session2_run2.xdf --speed 20

# Terminal 3:
python game_connector.py --model models/dino.joblib --game pong_ai
"""