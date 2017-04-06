import sys
import scipy
import os
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from random import random as rand
sys.path.append('liblsl-Python/pylsl')
from pylsl import StreamInlet, resolve_byprop, StreamInfo, StreamOutlet

from EEGNet import EEGNet

#SETTINGS
SAVE_DATA = False
SINGLE_TRIAL_FEEDBACK = True
BLOCK_PREDICTION = True
SUBJECT_ID = '8'
BLOCK = '107'
TRAINING = False

directory = 'Data/subject_' + SUBJECT_ID
filepath = directory + '/s' + SUBJECT_ID + '_b' + BLOCK + '_epoched.mat'

# Check that the file paths are correct
if SAVE_DATA:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print('made directory: ' + directory)
    if os.path.exists(filepath):
        raise ValueError('Path already exists. Do not overwrite data!')

# Create LSL outlet
info = StreamInfo('Python', 'classifications', 3)
outlet = StreamOutlet(info)
print("Outlet Created: Python")

# Create LSL inlet
stream = resolve_byprop('name', 'Matlab')
inlet = StreamInlet(stream[0])
print("Inlet Created: Matlab")

# Initialize variables
counter_epoch = 0
eeg = np.zeros((64, 385, 20))
pupil = np.zeros((20,241))
head_rotation = np.zeros((20,152))
dwell_time = np.zeros((20))
stimulus_type = np.zeros((20))
billboard_id = np.zeros((20))
billboard_cat = np.zeros((20))
classification = np.zeros((20))
confidence = np.zeros((20))
target_cat = 0

# Initialize Deep Learning
np.random.seed(123)
EEGnet = EEGNet(type = 'VR')
EEGnet.model.load_weights('/home/waytowich/EEGnet-VR/weights/jenn/CombinedModelWeights_fold8.hf5')

print("***Now Receiving Data***")
# MAIN LOOP
#t_prev = 0
while True:
    # Exit the loop if it has been more than 20 seconds since the last epoch input
    '''
    elapsed = time.time() - t_prev
    if (counter_epoch > 0) & (elapsed > 20):
        break
     '''    
    time.sleep(1)
    # RECEIVE DATA    
    # try to pull a new chunk     
    chunk, timestamps = inlet.pull_chunk(timeout = 1)
    epoch = np.transpose(np.asarray(chunk))

    # if a chunk is received
    if epoch.shape[0] != 0:
        #if epoch.shape[1] == 385:
        print
        print(epoch[-1,1])
        print(epoch[-1,-1])
        
        # Check for the cue to exit from matlab
        if epoch[-1,-1] == 1:
            target_cat = epoch[-1,-2]
            break
        
        eeg[:,:,counter_epoch] = epoch[0:-2,:]
        head_rotation[counter_epoch,:] = epoch[-2,0:head_rotation.shape[1]]
        stimulus_type[counter_epoch] = epoch[-1,0]
        billboard_id[counter_epoch] = epoch[-1,1]
        dwell_time[counter_epoch] = epoch[-1,2]
        billboard_cat[counter_epoch] = epoch[-1,3]
        pupil[counter_epoch,:] = epoch[-1, 4:245]
        
        # Classify data
        # process data
        eeg_trial = eeg[:,:,counter_epoch]
        eeg_trial = np.reshape(eeg_trial,eeg_trial.shape + (1,))        
        eeg_trial = np.transpose(eeg_trial,(2,0,1))
        eeg_trial = np.reshape(eeg_trial,eeg_trial.shape + (1,))
        
        head_trial = head_rotation[counter_epoch,:]
        head_trial = np.reshape(head_trial,(1,) + head_trial.shape + (1,))
        
        pupil_trial = pupil[counter_epoch,:]
        pupil_trial = np.reshape(pupil_trial,(1,) + pupil_trial.shape + (1,))
        
        dwell_trial = dwell_time[counter_epoch]
        dwell_trial = np.reshape(dwell_trial,(1,) + (1,) + (1,))
        
        probs = EEGnet.model.predict([eeg_trial, head_trial, pupil_trial, dwell_trial])
        pred_class = np.argmax(probs)+1
        target_confidence = probs[0,1]
        stream_out = [float(billboard_id[counter_epoch]), float(pred_class), float(target_confidence)]
        classification[counter_epoch]=pred_class
        confidence[counter_epoch]=target_confidence
        
        # random classifier
        #stream_out = [billboard_id[counter_epoch], np.round(3.0 * rand())+1, rand()] 
        if SINGLE_TRIAL_FEEDBACK:
            outlet.push_sample(stream_out)
            print('Billboard No: %d    Classification: %d    Confidence: %f' %(stream_out[0], stream_out[1], stream_out[2]))

        #t_prev = time.time()        
        counter_epoch += 1

inlet.close_stream()
print('inlet closed')
outlet.__del__()
print('outlet closed')
print('Actual target category: %d' %target_cat)

# Delete trials where the subject missed the billboard
missed_trials = np.where(stimulus_type == 0)
eeg = np.delete(eeg, missed_trials, 2)
head_rotation = np.delete(head_rotation, missed_trials, 0)
stimulus_type = np.delete(stimulus_type, missed_trials, 0)
billboard_id = np.delete(billboard_id, missed_trials, 0)
dwell_time = np.delete(dwell_time, missed_trials, 0)
billboard_cat = np.delete(billboard_cat, missed_trials, 0)
pupil = np.delete(pupil, missed_trials, 0)
classification = np.delete(classification, missed_trials, 0)
confidence = np.delete(confidence, missed_trials, 0)


pred_block=np.argmax([np.mean(confidence[np.where(billboard_cat==1.)[0]]), 
                      np.mean(confidence[np.where(billboard_cat==2.)[0]]), 
                      np.mean(confidence[np.where(billboard_cat==3.)[0]]), 
                      np.mean(confidence[np.where(billboard_cat==4.)[0]])]) + 1

if BLOCK_PREDICTION:
    if pred_block == 1:     
        image = Image.open('Pics/car_side-46.jpg').convert("L")
        plt.figure
        arr = np.asarray(image)
        plt.imshow(arr, cmap='gray')
        plt.show()
        plt.title('I know what you want...')
    if pred_block == 2:
        image = Image.open('Pics/grand_piano-2.jpg')
        plt.figure
        plt.imshow(image)
        plt.title('is it a piano you seek?')
    if pred_block == 3:
        image = Image.open('Pics/laptop-7.jpg')
        plt.figure
        plt.imshow(image)
        plt.title('looking for one of these?')
    if pred_block == 4:
        image = Image.open('Pics/schooner-4.jpg')
        plt.figure
        plt.imshow(image)
        plt.title('is this what you had in mind?')
        

print('Number of targets observed: %d' %np.sum(stimulus_type == 1))

# Save data
if SAVE_DATA:
    target_cat = target_cat * np.ones((len(stimulus_type)))
    scipy.io.savemat(filepath, {'EEG': eeg, 'stimulus_type': stimulus_type, 'billboard_id': billboard_id,'dwell_times': dwell_time, 'pupil': pupil, 'head_rotation': head_rotation, 'billboard_cat': billboard_cat, 'target_category': target_cat, 'classification': classification, 'confidence': confidence})
    print('Data Saved')
print('done')
