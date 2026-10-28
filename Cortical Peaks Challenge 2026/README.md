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
/ `DEFAULT_MAX_BAD_CHANNEL_FRAC` (artifact rejection), `DEFAULT_EXCLUDE_CHANNELS`, `DEFAULT_FREQUENCY_BANDS`
/ `DEFAULT_ERDS_SPECS` (FBCSP/Riemann/ERDS band settings). 

### Step 2 - Freeze the winning model

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

### Step 3 (Online): connect the model to a game

The online setup uses three processes. Start them in order, each in its own
terminal: the **game server**, the **prediction pipeline**, then the **game connector**.
The server must be running before the connector can send commands, and the
pipeline must be streaming before the connector can receive predictions.

#### Terminal 1: start the game server

For local testing, start the game's test server that you can find here in this repo:

```
python fake_server.py
```

To use it in an actual competition, start the official game server as instructed by the official Github repo of [neuroTUM](https://github.com/neuroTUM/Cortical-Peaks-Challenge-2026). The server files are not included in this project, so the
server command depends on that checkout.

#### Terminal 2: start the prediction pipeline

For a live EEG stream, run this from the project directory with the model
trained for the game:

```
python online_pipeline.py --model models/<model-name>.joblib
```

For local testing without EEG hardware, replay an XDF recording instead:

```
python online_pipeline.py --model models/<model-name>.joblib \
        --replay-file recordings/session1_run1.xdf --speed 20
```

The pipeline connects to an LSL stream of type `EEG` in live mode, or reads
the recording in replay mode. It filters the incoming data, keeps a rolling
window matching the model's training window, and emits class-probability
vectors to an LSL stream named `ClassifierOutput` (type `ClassProb`). The
class order is printed at startup and embedded in the stream metadata. Check
that the model's classes make sense for the selected game before proceeding.

#### Terminal 3: start the game connector

Run this project's `game_connector.py` with the game server checkout root on
`PYTHONPATH` (so its `shared` package can be imported). From the project
directory, pass the same model file used by the pipeline and the game name:
Use the folder that contains `shared/`, not the `shared/` folder itself. To
find it, locate `shared/connection/protocol.py` in the game server repository;
the `PYTHONPATH` value is the repository root above `shared/`. For example, if
that file is `/path/to/game-server-repo/shared/connection/protocol.py`, use
`/path/to/game-server-repo` as the path below.

```
PYTHONPATH=/path/to/game-server-repo:$PYTHONPATH python game_connector.py \
        --model models/<model-name>.joblib --game <game>
```

The cheat-sheet game identifiers are `dino_jump` (BrainSki 1, jump only),
`dino` (BrainSki 1, jump and duck), `ski` (BrainSki 2, static), `ski_dyn`
(BrainSki 2, dynamic), `pong`, and `pong_ai`. For example, with the model
available at `models/dino_jump.joblib`:

```
PYTHONPATH=/path/to/game-server-repo:$PYTHONPATH python game_connector.py \
        --model models/dino_jump.joblib --game dino_jump
```

The connector reads the probability stream, selects the highest-probability
class, and sends a game command only when confidence is at least `0.6`
(configurable with `--confidence-threshold`). Rest and classes not mapped to
an action for that game are ignored. The connector also needs `pylsl` and the
game server's `shared.connection` package. It connects to the server at
`127.0.0.1:5000` by default.

| Game (`--game`) | Classes mapped to actions |
| --- | --- |
| `dino_jump` | `right_hand` → jump |
| `dino` | `feet` → jump; `right_hand` → duck |
| `ski`, `ski_dyn` | `left_hand` → rotate left; `right_hand` → rotate right; `feet` → forward |
| `pong`, `pong_ai` | `right_hand` → up; `feet` → down |

Before each new model/game pairing, compare the pipeline's startup `classes`
and `Output ... channel order` with the mapping above. The model needs to
contain the class labels required by that game; otherwise actions for missing
classes cannot be produced. Model names in the cheat sheet (such as `dino`
and `brainski`) are examples and must exist at the path you pass to both
commands.

Replay mode is for trying out the pipeline and connector locally; it does
not replace the live EEG stream or the competition server during a real run.
The model was trained on cue-aligned epochs, while the online pipeline uses
a rolling window of recent EEG without cue timing. Expect live performance
to differ from offline cross-validation. 


## If something breaks 

- **`ModuleNotFoundError`**: if you use venv it might not be activated, 
  or packages didn't install into it. Check `pip install ...` 
  ran with the venv active (prompt should show `(.venv)`), or use the full path:
  `.venv/bin/pip install ...`. 
- **`externally-managed-environment` pip error**: you're not actually
  inside the venv — `source .venv/bin/activate` first. Disregard if you 
  are doing it without a virtual environment. 
-  Sorry if anything else breaks. It's ok to cry. 




