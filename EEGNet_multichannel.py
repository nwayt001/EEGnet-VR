from keras.models import Model
from keras.layers.core import Dense, Activation, Permute, SpatialDropout2D
from keras.layers.convolutional import Convolution2D, Convolution1D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.layers import MaxPooling1D, AveragePooling1D
from keras.regularizers import l1l2
from keras.layers import Input, Flatten, merge
from keras.utils import np_utils
import numpy as np
import scipy as sp
from keras.utils.visualize_util import plot
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn import metrics
from sklearn import utils
import sys, math
import matplotlib.pyplot as plt
from scipy import signal


class EEGNet(object):
    def __init__(self, nb_classes=2, numChans = 67, numSamples = 400, regRate = 0.01, dropoutRate = 0.25, type = 'VR', display_models = False):
        self.nb_classes = nb_classes
        self.numChans = numChans
        self.numSamples = numSamples
        self.regRate = regRate
        self.dropoutRate = dropoutRate
        self.display_models = display_models
        self.type = type
        
        if type == 'VR':
            self.EEGNetVR(num_samples = self.numSamples)
    
    def EEGNetVR(self, nb_classes=2, numChans=67, num_samples = 385, regRate=0.001, dropoutRate = 0.2):
        
        ## EEG Data
        eeg_input = Input((numChans, num_samples,1))
        conv1        = Convolution2D(64, numChans, 1, 
                                     W_regularizer = l1l2(l1=regRate, l2=regRate))(eeg_input)
        batchNorm1   = BatchNormalization(axis=1)(conv1)
        elu1         = ELU()(batchNorm1)
        spat_drop1   = SpatialDropout2D(dropoutRate)(elu1)        
        permute_dims = 3, 2, 1
        permute1     = Permute(permute_dims)(spat_drop1)        
        conv2        = Convolution2D(16, 8, 67, border_mode = 'same', 
                                W_regularizer=l1l2(l1=0.0, l2=regRate),
                                subsample = (2, 4))(permute1)
        batchNorm2   = BatchNormalization(axis=1)(conv2)
        elu2         = ELU()(batchNorm2)
        spat_drop2   = SpatialDropout2D(dropoutRate)(elu2)        
        conv3        = Convolution2D(4, 8, 8, border_mode = 'same',
                                W_regularizer=l1l2(l1=0.0, l2=regRate),
                                subsample = (2, 4))(spat_drop2)
        batchNorm3   = BatchNormalization(axis=1)(conv3)
        elu3         = ELU()(batchNorm3)
        spat_drop3   = SpatialDropout2D(dropoutRate)(elu3)
        
        conv4        = Convolution2D(2, 4, 8, border_mode = 'same',
                                W_regularizer=l1l2(l1=0.0, l2=regRate),
                                subsample = (2, 4))(spat_drop3)
        batchNorm4   = BatchNormalization(axis=1)(conv4)
        elu4         = ELU()(batchNorm4)
        spat_drop4   = SpatialDropout2D(dropoutRate)(elu4)
        
        eeg     = Flatten()(spat_drop4)
        dense1     = Dense(nb_classes)(eeg)
        softmax1   = Activation('softmax')(dense1)
        
        self.model = Model(input=eeg_input, output=softmax1)
        
        if self.display_models:
            print(self.model.summary())
            plot(self.model,to_file='model.png')

        # define custom metric for AUC
        def auc(y_true, y_pred):
            fpr, tpr, thresholds = metrics.roc_curve(y_true[:,1],y_pred[:,1], pos_label=1)
            AUC = metrics.auc(fpr, tpr)
            return AUC
                        
        # Set optimizers and Loss functions for each

        self.model.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics=['accuracy'])
            
    
if __name__ == '__main__':
    np.random.seed(123)
    
    sys.path.append('../Dropbox/NEDE_Dropbox/Data/training_v5')

    # Load Training Data
    data = sp.io.loadmat('training_data.mat')
    
    # Train/Test which models
    run = True

    # Only validation, no test for now
    num_sub = 8
    nb_classes = 2
    batch_size = 64
    nb_epochs = 150
    

    cv = np.zeros((num_sub,num_sub))
    for idx, c in enumerate(cv):
        c[idx] = 1.0
        c[(idx+1)%num_sub] = 2.0

    sub = data['subject']
    eeg = data['EEG'].astype('float32')
    head = data['head_rotation'].astype('float32')
    pupil = data['pupil'].astype('float32')
    dwell = data['dwell_times'].astype('float32')
    labels = data['stimulus_type'].astype('float32')
    idx = np.where(labels==2)[1]
    labels[:,idx]=0
    AUC_combined_model=[]
    AUC_eeg_model=[]
    AUC_head_model=[]
    AUC_pupil_model=[]
    AUC_dwell_model=[]

    # class weightings:
    c0_weight = 1.  
    c1_weight = 3.
    for cv in range(10):
        print('PROCESSING CV FOLD {} '.format(cv))
        for i in [8]:        
            
            # grad data for subject i
            sub_idx = np.where(sub==i)[1]
            
            # take 70% train, 15% validate, 15% test
            rand_idx = np.random.permutation(len(sub_idx))
            train_idx = sub_idx[rand_idx[:int(len(sub_idx)*.7)]]
            val_idx = sub_idx[rand_idx[int(len(sub_idx)*.7):int(len(sub_idx)*.85)]]
            test_idx = sub_idx[rand_idx[int(len(sub_idx)*.85):int(len(sub_idx)*1.)]]
                             
            y_train = labels[:,train_idx]                            
            y_val = labels[:,val_idx]
            y_test = labels[:,test_idx]
            y_train = np_utils.to_categorical(y_train,nb_classes)
            y_val = np_utils.to_categorical(y_val,nb_classes)
            y_test = np_utils.to_categorical(y_test,nb_classes)
            
            
            # set all data to a sampling rate of 50hz
            sampling_rate = 1/10 
            train_size = train_idx.shape[0]
            val_size = val_idx.shape[0]
            test_size = test_idx.shape[0]
            
            pupil_duration = 4000 # 4000ms
            eeg_duration = 1500 # 1500ms
            head_duration = 2000 # 2000ms
            
            eeg_start_delay = 500 #EEG measurements starts 500ms after Pupil which starts first 
            head_start_delay = 500 #Head Rotation measurements starts 500ms after Pupil which starts first 
            dwell_start_delay = 1000 #The dwell time begins 1000ms after the pupil measurement begins
            # eeg data        
            train_eeg = eeg[:,:,train_idx]
            train_eeg = np.transpose(train_eeg,(2,0,1))
            train_eeg = np.reshape(train_eeg,train_eeg.shape + (1,))
            train_eeg_resampled = signal.resample(train_eeg, int(eeg_duration*sampling_rate), axis=2)
            train_eeg_full_length = np.zeros((train_size, eeg.shape[0], int(pupil_duration*sampling_rate), 1))
            train_eeg_full_length[:,:,int(eeg_start_delay*sampling_rate):int((eeg_start_delay+eeg_duration)*sampling_rate),:] = train_eeg_resampled
            
            val_eeg = eeg[:,:,val_idx]
            val_eeg = np.transpose(val_eeg,(2,0,1))
            val_eeg = np.reshape(val_eeg,val_eeg.shape + (1,))
            val_eeg_resampled = signal.resample(val_eeg, int(eeg_duration*sampling_rate), axis=2)
            val_eeg_full_length = np.zeros((val_size, eeg.shape[0], int(pupil_duration*sampling_rate), 1))
            val_eeg_full_length[:,:,int(eeg_start_delay*sampling_rate):int((eeg_start_delay+eeg_duration)*sampling_rate),:] = val_eeg_resampled
            
            test_eeg = eeg[:,:,test_idx]
            test_eeg = np.transpose(test_eeg,(2,0,1))
            test_eeg = np.reshape(test_eeg,test_eeg.shape + (1,))
            test_eeg_resampled = signal.resample(test_eeg, int(eeg_duration*sampling_rate), axis=2)
            test_eeg_full_length = np.zeros((test_size, eeg.shape[0], int(pupil_duration*sampling_rate), 1))
            test_eeg_full_length[:,:,int(eeg_start_delay*sampling_rate):int((eeg_start_delay+eeg_duration)*sampling_rate),:] = test_eeg_resampled
            
            # head rotation data
            train_head = head[train_idx,:]
            train_head = np.reshape(train_head,(train_head.shape[0],1,train_head.shape[1]) + (1,))
            train_head_resampled = signal.resample(train_head, int(head_duration*sampling_rate), axis=2)
            train_head_full_length = np.zeros((train_size, 1, int(pupil_duration*sampling_rate), 1))
            train_head_full_length[:,:,int(head_start_delay*sampling_rate):int((head_start_delay+head_duration)*sampling_rate),:] = train_head_resampled
            
            val_head = head[val_idx,:]
            val_head = np.reshape(val_head,(val_head.shape[0],1,val_head.shape[1]) + (1,))
            val_head_resampled = signal.resample(val_head, int(head_duration*sampling_rate), axis=2)
            val_head_full_length = np.zeros((val_size, 1, int(pupil_duration*sampling_rate), 1))
            val_head_full_length[:,:,int(head_start_delay*sampling_rate):int((head_start_delay+head_duration)*sampling_rate),:] = val_head_resampled

            test_head = head[test_idx,:]
            test_head = np.reshape(test_head,(test_head.shape[0],1,test_head.shape[1]) + (1,))
            test_head_resampled = signal.resample(test_head, int(head_duration*sampling_rate), axis=2)
            test_head_full_length = np.zeros((test_size, 1, int(pupil_duration*sampling_rate), 1))
            test_head_full_length[:,:,int(head_start_delay*sampling_rate):int((head_start_delay+head_duration)*sampling_rate),:] = test_head_resampled
            
            # pupil data
            train_pupil = pupil[train_idx,:]
            train_pupil = np.reshape(train_pupil,(train_pupil.shape[0],1,train_pupil.shape[1]) + (1,))
            train_pupil_resampled = signal.resample(train_pupil, int(pupil_duration*sampling_rate), axis=2)
            
            val_pupil = pupil[val_idx,:]
            val_pupil = np.reshape(val_pupil,(val_pupil.shape[0],1,val_pupil.shape[1]) + (1,))
            val_pupil_resampled = signal.resample(val_pupil, int(pupil_duration*sampling_rate), axis=2)
            
            test_pupil = pupil[test_idx,:]
            test_pupil = np.reshape(test_pupil,(test_pupil.shape[0],1,test_pupil.shape[1]) + (1,))
            test_pupil_resampled = signal.resample(test_pupil, int(pupil_duration*sampling_rate), axis=2)
            
            # dwell time data
            train_dwell = dwell[:,train_idx]
            train_dwell = train_dwell.transpose()
            train_dwell = np.reshape(train_dwell,train_dwell.shape + (1,))
            train_dwell_full_length = np.zeros((train_size, 1, int(pupil_duration*sampling_rate), 1))
            for i in range(train_size):
                dwell_length_idx = math.ceil(train_dwell[i,0,0]*1000*sampling_rate)
                train_dwell_full_length[i,:,int(dwell_start_delay*sampling_rate):int(dwell_start_delay*sampling_rate)+dwell_length_idx,:] = 1
                
            
            val_dwell = dwell[:,val_idx]
            val_dwell = val_dwell.transpose()
            val_dwell = np.reshape(val_dwell,val_dwell.shape + (1,))
            val_dwell_full_length = np.zeros((val_size, 1, int(pupil_duration*sampling_rate), 1))
            for i in range(val_size):
                dwell_length_idx = math.ceil(val_dwell[i,0,0]*1000*sampling_rate)
                val_dwell_full_length[i,:,int(dwell_start_delay*sampling_rate):int(dwell_start_delay*sampling_rate)+dwell_length_idx,:] = 1
            
            test_dwell = dwell[:,test_idx]
            test_dwell = test_dwell.transpose()
            test_dwell = np.reshape(test_dwell,test_dwell.shape + (1,))
            test_dwell_full_length = np.zeros((test_size, 1, int(pupil_duration*sampling_rate), 1))
            for i in range(test_size):
                dwell_length_idx = math.ceil(test_dwell[i,0,0]*1000*sampling_rate)
                test_dwell_full_length[i,:,int(dwell_start_delay*sampling_rate):int(dwell_start_delay*sampling_rate)+dwell_length_idx,:] = 1
            
            train_full = np.concatenate((train_eeg_full_length,train_head_full_length,train_pupil_resampled,train_dwell_full_length), axis=1)
            val_full = np.concatenate((val_eeg_full_length,val_head_full_length,val_pupil_resampled,val_dwell_full_length), axis=1)
            test_full = np.concatenate((test_eeg_full_length,test_head_full_length,test_pupil_resampled,test_dwell_full_length), axis=1)
            
#            t1 = np.linspace(-500,1000,num=100)
#            t2 = np.linspace(-500,1000,num=152)
#            
#            plt.plot(t1,train_head_resampled[0,:,0],'r',t2,train_head[0,:,0],'b')
           
            # TRAIN / TEST COMBINED MODEL
            EEGnet = EEGNet(type = 'VR')
            weightsfilename = 'weights/test_multichannel/MultichannelModelWeights_fold' + str(i) +'.hf5'
            checkpointer = ModelCheckpoint(filepath = weightsfilename, verbose=0,
                                               save_best_only = True)
                
                # Train Combined model
            hist = EEGnet.model.fit([train_full],y_train,
                                 batch_size = batch_size, nb_epoch = nb_epochs, verbose=1,
                                 validation_data = ([val_full],y_val),
                                 callbacks = [checkpointer], class_weight={0:c0_weight, 1:c1_weight}) #class_weight = {0:1., 1:1.25}
                
                # load in best validation
            EEGnet.model.load_weights(weightsfilename)
                
                # calculate auc
            probs = EEGnet.model.predict([test_full])
            fpr, tpr, thresholds = metrics.roc_curve(y_test[:,1], probs[:,1], pos_label=1)
            AUC = metrics.auc(fpr, tpr)
            print('Combined Model AUC: ', AUC)
            AUC_combined_model.append(AUC)
                
            
        # save AUC results
        avg_auc_combined = np.mean(AUC_combined_model)

        
        print('Avg AUC Combined Model: ', avg_auc_combined)

        
        print('subject wise AUC Combined Model:')
        print(AUC_combined_model)
        
        results = dict()
        results['AUC_combined'] = AUC_combined_model

        results['avg_AUC_combined'] = avg_auc_combined

        sp.io.savemat('results_multichannel/results_test_r5_125_1.mat',results)

    
        