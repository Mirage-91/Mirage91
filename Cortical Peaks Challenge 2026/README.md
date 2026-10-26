# Cortical Peaks Challenge 2026

This folder contains the code that was programmed by the Mirage91 student team for the Cortical Peaks Challenge 2026. The code you can find here is not the code which was actually used for the challenge! 

## What is the Cortical Peaks Challenge? 
The Cortical Peaks Challenge is a Brain Computer Interface (BCI)- based race. A pilot has to compete in three different disciplines against other candidates from other countries and Universities. 

The games in which the teams compete in are the following: 
- **BrainSki 1**\
An arcade-style game in which players must jump (and duck) to avoid obstacles that move toward the player using binary BCI control. Two versions will be played:

        * Jump only -> binary input
        * Jump and duck -> two binary inputs 

- **BrainSki 2**\
A continuous-control downhill skiing game where players navigate the course and aim to reach the finish line while avoiding obstacles. Two versions will be played:

        * Static obstacles
        * Moving obstacles  
- **Pong**\
A BCI-controlled adaptation of the classic Pong game. It requires players to continuously control the paddle position and react in real time against their opponent. 

Find more informations to the event on the [Website](https://www.tugraz.at/institute/ine/graz-bci-conferences/10th-graz-bci-conference-2026/cortical-peaks-challenge-2026). 

## BCI-Pipeline 

In this Brain Computer Interface Pipeline 3 active classes are used doing motor imagery (left_hand / right_hand / feet) plus a rest (a non-active class). 

First offline benchmarking is performed to pick the best feature and classifier combination. After 
we pick the best combination we "freeze" the model iso that it can be deployed in a live streaming loop for the challenge (online). 


### Structure of the project:

```
   recordings (as xdf files)
        │
        ▼
run_pipeline.py                 compares features and classifiers,
        │                       
        ▼
train_final_model.py            trains the selected combination
        ▼
models/<model-name>.joblib      fitted model, loaded for predictions
        │
        ▼
online_pipeline.py              loads model and connects to live EEG,
        │                       makes continuous predictions
        ▼
   LSL stream "ClassifierOutput"   

```
## How to make it run on your machine 

### Step 1 (Offline): 
In the file `run_pipeline.py` run the code with the settings that you want to use. For example using the following 
```
python run_pipeline.py --recordings-dir /path/to/recordings --cv group
```
will print a ranked table of every (feature extractor × classifier) combo,
cross-validated with each session held out in turn. Be prepared to wait a few
minutes (ca. 10-15 min) while it loads, filters, and rejects EEG data.

Configs you can tune is at the top of file `run_pipeline.py`:\
`BANDPOWER_CLASS_CONFIGS`, `DEFAULT_PTP_THRESHOLD` / `DEFAULT_GRAD_THRESHOLD`
/ `DEFAULT_MAX_BAD_CHANNEL_FRAC` (artifact rejection), `DEFAULT_EXCLUDE_CHANNELS`
(channels dropped outright — Fp1/Fp2 by default, blink-prone), `DEFAULT_FREQUENCY_BANDS`
/ `DEFAULT_ERDS_SPECS` (FBCSP/Riemann/ERDS band settings). 

## Step 2 - Freeze the winning model

Once a combination is picked for each game/command-pair you need, 
it needs to be trained on ALL data (remember before we held out data for folds) 
and save it. For example: 

```
python train_final_model.py \
    --recordings-dir /path/to/recordings \
    --classes right_hand rest \
    --feature Riemann --classifier "sLDA (shrinkage)" \
    --model-name jump_only
```

Saves the fitted model to `models/jump_only.joblib` and a `.json`
sidecar with the settings/classes/channels it expects. The `.joblib` file
is what the online pipeline loads; the JSON is readable metadata. The
output folder is created automatically and ignored by Git. When running the
online pipeline from this project directory, refer to the model as
`models/jump_only.joblib`. Re-run this any time you get new data; it will
overwrite files with the same model name.

Repeat with different `--classes`/`--model-name` for each game's command
mapping (e.g. `--classes right_hand feet --model-name jump_and_duck`).
Omit `--classes` entirely to train on all 4 for the full-game model
(`--model-name full_game`); any subset works too (e.g. a 3-way).

## Step 3 (Online): run it live

Run the online file from this project directory:
```
python online_pipeline.py --model models/jump_only.joblib
```

Test against a recorded file first — no hardware needed:
```
python online_pipeline.py --model models/jump_only.joblib \
    --replay-file /path/to/session1_run1.xdf --speed 20
```

Pushes continuous class probabilities (not a single discrete decision) to
an LSL stream named `ClassifierOutput`, type `ClassProb` — matches the old
team's own `main_ML.py` convention. Startup prints the channel order (e.g.
`['rest', 'right_hand']`) for a quick sanity check, and the same order is
now embedded directly in the LSL stream's own metadata, so your teammate's
code (see `game_connector.py` below) reads it straight from the stream
instead of having to hardcode or eyeball the printed line. Get this
backwards and every command inverts silently, so still worth double-checking.

### Important for the online step

The model was trained on 6-second **cue-aligned** epochs (2s quiet baseline + 4s post-cue movement imagery). Live, there's of course no cue. So we use a rolling 6-second buffer of "whatever just happened,". Sadly ,however, this won't give us any baseline as the pilot will always be "active" so to speak. 
We need to expect lower live accuracy than the offline cross-validation numbers suggested. 
`online_pipeline.py --smoothing-window N` averages the last N predictions'
probabilities to reduce jitter/false triggers (default 5; set to 1 for raw,
unsmoothed output) — helps some, but doesn't fix the underlying
baseline/no-baseline mismatch. Retraining on shorter, baseline-free windows
that better match what the online buffer can actually provide is the more
fundamental fix, still TODO.

## Game side: getting predictions into the game

`game_connector.py` is the file your teammate actually needs — it
wraps the LSL details into one function so they don't have to touch arrays,
indices, or model internals:

```python
from game_connector import PredictionClient

client = PredictionClient()   # connects once, at startup

while game_running:
    label, confidence = client.get_prediction()
    if label == "right_hand" and confidence > 0.7:
        trigger_jump()
```

`label` is always a real class name read live from the stream, so it stays
correct even if you swap in a different `--model-name` `.joblib` later.
Requires `pylsl` installed on whichever machine runs the game code too.
`online_pipeline.py` must already be running (live or `--replay-file`) for
this to have a stream to connect to.


## If something breaks 

- **`ModuleNotFoundError`**: if you use venv it might not be activated, 
  or packages didn't install into it. Check `pip install ...` 
  ran with the venv active (prompt should show `(.venv)`), or use the full path:
  `.venv/bin/pip install ...`. 
- **`externally-managed-environment` pip error**: you're not actually
  inside the venv — `source .venv/bin/activate` first. Disregard if you 
  are doing it without a virtual environment. 
- ** Sorry if anything else breaks. You are on your own then.




