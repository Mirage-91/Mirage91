from psychopy import visual, core, event
from pylsl import StreamInfo, StreamOutlet
import random

# creat marker stream
info = StreamInfo(name='paradigm_P300_markers', type='Markers',
                  channel_count=1, channel_format='string',
                  source_id='P300_paradigm')
outlet = StreamOutlet(info)

# define parameters
FRAMES_STIMULUS = 2
FRAMES_ISI = 8
NUM_SCANS = 16 

# set up window
window = visual.Window(fullscr=True, color="black", units="height", screen=1)

# set up target
target_string = "THISISJUSTAQUICKTEST"
target_stims = []
spacing = 0.06 
start_x = -(len(target_string) - 1) * spacing / 2 # string positioning

for i, char in enumerate(target_string):
    x_pos = start_x + (i * spacing)
    stim = visual.TextStim(window, text=char, pos=(x_pos, 0.45), height=0.06, color="white")
    target_stims.append(stim)

# set up matrix
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [".", "_", "del", "clr"]
letters = []
positions = []

for index, label in enumerate(labels):
    row, column = divmod(index, 6)
    position = ((column - 2.5) * 0.14, (2 - row) * 0.16)
    positions.append({'char': label, 'row': row, 'col': column, 'pos': position})
    letters.append(
        visual.TextStim(window, text=label, pos=position, color="white", height=0.06)
    )

# BufferImageStim allows us to draw the matrix as one stimulus,
# instead of drawing each letter individually. 
matrix = visual.BufferImageStim(window, stim=letters)

face_stim = visual.ImageStim(win=window, image="einstein.png", size=(0.12, 0.14))

# create dictionary with positions per row/column for the images 
pos_flash = {'ROW': {r: [] for r in range(5)}, 'COL': {c: [] for c in range(6)}}
for p in positions:
    pos_flash['ROW'][p['row']].append(p['pos'])
    pos_flash['COL'][p['col']].append(p['pos'])

# pause to allow user to start recording
instructions = visual.TextStim(
    window, 
    text="Start LabRecorder now.\n\nPress SPACE to begin the experiment.", 
    color="white", height=0.05
)

instructions.draw()
window.flip()

event.waitKeys(keyList=["space"])

outlet.push_sample(["START"])

# main paradigm
flash_options = list(range(11)) # 0-4 (Rows), 5-10 (Columns)    --> 11 stimulus groups based on 5x6 grid

for i, current_char in enumerate(target_string):
    if 'escape' in event.getKeys(keyList=["escape"]):
        break
        
    target_stims[i].color = "red"
    
    target_idx = labels.index(current_char)
    target_row, target_col = divmod(target_idx, 6)    # quotient is row index, remainder is column index
    
    matrix.draw()
    for letter_stim in target_stims:
        letter_stim.draw()
    window.flip()
    core.wait(3.0) 
    
    outlet.push_sample([f"START_LETTER_{current_char}"])
    
    for scan in range(NUM_SCANS):
        random.shuffle(flash_options)
        
        for flash_id in flash_options:
            if 'escape' in event.getKeys(keyList=["escape"]):
                window.close()
                core.quit()

            is_row = flash_id < 5
            flash_idx = flash_id if is_row else (flash_id - 5)    # subtract 5, because columns are defined from 5-10
            
            # for training include if the current flash is the target row/column
            is_target = (is_row and flash_idx == target_row) or (not is_row and flash_idx == target_col)
            marker_prefix = "TARGET" if is_target else "NONTARGET"
            marker_type = "ROW" if is_row else "COL"
            full_marker = f"{marker_prefix}_{marker_type}_{flash_idx}"

            # extract image positions for current flash
            active_pos = pos_flash[marker_type][flash_idx]
            
            # send marker at stimulus presentation
            window.callOnFlip(outlet.push_sample, [full_marker])
            
            # stimulus interval
            for frame in range(FRAMES_STIMULUS):
                matrix.draw()
                for letter_stim in target_stims:
                    letter_stim.draw()
                
                for pos in active_pos:
                    face_stim.pos = pos
                    face_stim.draw()
                
                window.flip()
                                
            # interstimulus interval
            for frame in range(FRAMES_ISI):
                matrix.draw()
                for letter_stim in target_stims:
                    letter_stim.draw()
                window.flip()

    target_stims[i].color = "white"

outlet.push_sample(["END"])

finished = visual.TextStim(
    window, text="Paradigm finished.\n\nPress SPACE or ESC to close.", color="white", height=0.05
)
finished.draw()
window.flip()
event.waitKeys(keyList=["space", "escape"])

window.close()
core.quit()

