# Cortical Peaks Challenge 2026

This folder contains the code that was programmed by the Mirage91 student team for the Cortical Peaks Challenge 2026. The code you can find here is not the code which was actually used for the challenge! 

## What is the Cortical Peaks Challenge? 

## BCI-Pipeline 

In this Brain Computer Interface Pipeline 3 active classes are used doing motor imagery (left_hand / right_hand / feet) plus rest. 

First offline benchmarking is performed to pick the best feature and classifier combination. After 
we pick the best combination we "freeze" the model iso that it can be deployed in a live streaming loop for the challenge (online). 


## Structure of the project:

```
   recordings (as xdf files)
        │
        ▼
offline/run_pipeline.py         compares features and classifier,
        |                        used to evaluate the best pair
        │                       
        ▼
offline/train_final_model.py    after picking features/classifier
        |         
        |                        based on previous we save it as a 
        |                       single file (for online)                
        ▼
   online/models/*.joblib
        │
        ▼
online/online_pipeline.py       loads file and connects to live EEG,
        │                       makes continuous predictions
        ▼
   LSL stream "ClassifierOutput"   
