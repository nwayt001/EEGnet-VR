from keras.models import Model
from keras.layers.core import Dense, Activation, Permute, SpatialDropout2D, Dropout
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
import matplotlib.pyplot as plt
import time

class EEGNet(object):
    def __init__(self, nb_classes=2, numChans = 64, num_eeg_samples = 385, l1rate = 0.001, l2rate=0.001, dropoutRate = 0.25, s_dropout=0.2,
                 num_head_rotation_samples=152, num_pupil_samples=241, num_dwell_time_samples=1, display_models = False):
        self.nb_classes = nb_classes
        self.numChans = numChans
        self.num_eeg_samples = num_eeg_samples
        self.l1rate = l1rate
        self.l2rate = l2rate
        self.num_head_rotation_samples = num_head_rotation_samples
        self.num_pupil_samples = num_pupil_samples
        self.num_dwell_time_samples = num_dwell_time_samples
        self.dropoutRate = dropoutRate
        self.display_models = display_models
        self.type = type
        
        ## EEG Data
        eeg_input = Input((numChans, num_eeg_samples,1))
        x = Convolution2D(16, numChans, 1, W_regularizer = l1l2(l1=l1rate, l2=l2rate), activation='elu')(eeg_input)
        x = BatchNormalization(axis=1)(x)
        x = SpatialDropout2D(dropoutRate)(x)        
        permute_dims = 3, 2, 1
        x = Permute(permute_dims)(x)        
        x = Convolution2D(4, 2, 64, border_mode = 'same', W_regularizer=l1l2(l1=l1rate, l2=l2rate), activation='elu', subsample = (2, 4))(x)
        x = BatchNormalization(axis=1)(x)
        x = SpatialDropout2D(dropoutRate)(x)   
        x = Convolution2D(4, 8, 8, border_mode = 'same', W_regularizer=l1l2(l1=l1rate, l2=l2rate), activation='elu', subsample = (2, 4))(x)
        x = BatchNormalization(axis=1)(x)
        x = SpatialDropout2D(dropoutRate)(x)
        eeg = Flatten()(x)
        eeg_output = Dense(nb_classes,activation='softmax')(eeg)
        self.eeg_model = Model(input=eeg_input, output=eeg_output)        
        if self.display_models:
            print(self.eeg_model.summary())
            plot(self.eeg_model,to_file='EEG_model.png')
            
        # Head Rotation Data        
        head_rotation_input = Input((num_head_rotation_samples,1))        
        x = Convolution1D(4,20,activation='elu',W_regularizer=l1l2(l1=l1rate, l2=l2rate))(head_rotation_input)
        x = MaxPooling1D(pool_length=2)(x)
        x = Dropout(dropoutRate)(x)
        x = Convolution1D(4,20,activation='elu',W_regularizer=l1l2(l1=l1rate, l2=l2rate))(x)
        x = MaxPooling1D(pool_length=2)(x)
        x = Dropout(dropoutRate)(x)
        x = Convolution1D(4,5,activation='elu',W_regularizer=l1l2(l1=l1rate, l2=l2rate))(x)
        x = MaxPooling1D(pool_length=2)(x)
        x = Dropout(dropoutRate)(x)
        x = Flatten()(x)
        head_rotation = Dense(5,activation='elu',W_regularizer=l1l2(l1=l1rate, l2=l2rate))(x)
        head_rotation_output = Dense(nb_classes,activation='softmax')(head_rotation)
        self.head_rotation_model = Model(input = head_rotation_input, output = head_rotation_output)
        if self.display_models:
            print(self.head_rotation_model.summary())
            plot(self.head_rotation_model,to_file='head_rotation_model.png')
            
        # pupilometry Data
        pupil_input = Input((num_pupil_samples,1))        
        x = AveragePooling1D(pool_length=2)(pupil_input)
        #x = AveragePooling1D(pool_length=2)(x)
        x = Convolution1D(4,20,activation='elu',W_regularizer=l1l2(l1=l1rate, l2=l2rate))(x)
        x = MaxPooling1D(pool_length=2)(x)
        x = Dropout(dropoutRate)(x)
        x = Convolution1D(4,10,activation='elu',W_regularizer=l1l2(l1=l1rate, l2=l2rate))(x)
        x = MaxPooling1D(pool_length=2)(x)
        x = Dropout(dropoutRate)(x)
        x = Convolution1D(4,10,activation='elu',W_regularizer=l1l2(l1=l1rate, l2=l2rate))(x)
        x = MaxPooling1D(pool_length=2)(x)
        x = Dropout(dropoutRate)(x)
        x = Flatten()(x)
        pupil = Dense(5,activation='elu',W_regularizer=l1l2(l1=l1rate, l2=l2rate))(x)
        pupil_output = Dense(nb_classes,activation='softmax')(pupil)
        self.pupil_model = Model(input = pupil_input, output = pupil_output)
        if self.display_models:
            print(self.pupil_model.summary())
            plot(self.pupil_model,to_file='pupil_model.png')
        
        # Dwell Data
        dwell_input = Input((num_dwell_time_samples,1))
        dwell = Dense(1)(dwell_input)
        dwell = Flatten()(dwell)
        dwell_output = Dense(nb_classes,activation='softmax')(dwell)
        self.dwell_model = Model(input=dwell_input, output=dwell_output)
        if self.display_models:
            print(self.dwell_model.summary())
            plot(self.dwell_model,to_file='dwell_model.png')
            
        # Combined Model
        merged_features = merge([eeg, head_rotation, pupil, dwell], mode = 'concat')
        combined_output = Dense(nb_classes, activation='softmax',W_regularizer=l1l2(l1=l1rate, l2=l2rate))(merged_features)
        
        self.model = Model(input = [eeg_input, head_rotation_input, pupil_input, dwell_input], output = combined_output)
        if self.display_models:
            print(self.model.summary())
            plot(self.model,to_file='model.png')

        # define custom metric for AUC
        def auc(y_true, y_pred):
            fpr, tpr, thresholds = metrics.roc_curve(y_true[:,1],y_pred[:,1], pos_label=1)
            AUC = metrics.auc(fpr, tpr)
            return AUC
                        
        # Set optimizers and Loss functions for each
        self.eeg_model.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics=['accuracy'])
        self.head_rotation_model.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics=['accuracy'])
        self.pupil_model.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics=['accuracy'])
        self.dwell_model.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics=['accuracy'])
        self.model.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics=['accuracy'])
            
    
if __name__ == '__main__':
    np.random.seed(123)
    
    # Load Training Data
    data = sp.io.loadmat('/home/waytowich/Dropbox/Data_v3/training_data/training_data.mat')
    
    # Train/Test which models
    run_combined = True
    run_eeg = True
    run_head = True
    run_pupil = True
    run_dwell = True
    
    num_sub = 8
    num_bootstrap = 8
    nb_classes = 2
    batch_size = 64
    nb_epochs = 150

    sub = data['subject']
    eeg = data['EEG'].astype('float32')
    head = data['head_rotation'].astype('float32')
    pupil = data['pupil'].astype('float32')
    dwell = data['dwell_times'].astype('float32')
    labels = data['stimulus_type'].astype('float32')
    block = data['block']
    idx = np.where(labels==2)[1]
    labels[:,idx]=0

    # get subject list
    sub_list=[]
    for i in range(1,np.max(sub)+1):
        if len(np.where(sub==i)[1]!=0):
            sub_list.append(i)

    PARM_L1=[0.0, 0.001, 0.01, 0.1]
    PARM_L2=[0.0, 0.001, 0.01, 0.1]
    PARM_DROP=[0.0, 0.01, 0.1, 0.25]
    PARM_C1_WEIGHT=[2, 3, 4]

    AUC_combined_model=np.zeros((len(PARM_L1),len(PARM_L2),len(PARM_DROP),len(PARM_C1_WEIGHT),num_bootstrap,len(sub_list)))
    AUC_eeg_model=np.zeros((len(PARM_L1),len(PARM_L2),len(PARM_DROP),len(PARM_C1_WEIGHT),num_bootstrap,len(sub_list)))
    AUC_head_model=np.zeros((len(PARM_L1),len(PARM_L2),len(PARM_DROP),len(PARM_C1_WEIGHT),num_bootstrap,len(sub_list)))
    AUC_pupil_model=np.zeros((len(PARM_L1),len(PARM_L2),len(PARM_DROP),len(PARM_C1_WEIGHT),num_bootstrap,len(sub_list)))
    AUC_dwell_model=np.zeros((len(PARM_L1),len(PARM_L2),len(PARM_DROP),len(PARM_C1_WEIGHT),num_bootstrap,len(sub_list)))
                                    
    # class weightings:
    c0_weight = 1.  
    c1_weight = 3.
    t = time.time()
    for idxi,i in enumerate(sub_list): 
        for idxl1, l1_rate in enumerate(PARM_L1):
            for idxl2, l2_rate in enumerate(PARM_L2):   
                for idxdrp, drp_rate in enumerate(PARM_DROP):
                    for idxc1, c1_weight in enumerate(PARM_C1_WEIGHT):
                        for cv in range(8):
                            print('PROCESSING CV FOLD {}  FROM SUB {}'.format(cv,i))
                    
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
                            
                            # eeg data        
                            train_eeg = eeg[:,:,train_idx]
                            train_eeg = np.transpose(train_eeg,(2,0,1))
                            train_eeg = np.reshape(train_eeg,train_eeg.shape + (1,))
                            
                            val_eeg = eeg[:,:,val_idx]
                            val_eeg = np.transpose(val_eeg,(2,0,1))
                            val_eeg = np.reshape(val_eeg,val_eeg.shape + (1,))
                            
                            test_eeg = eeg[:,:,test_idx]
                            test_eeg = np.transpose(test_eeg,(2,0,1))
                            test_eeg = np.reshape(test_eeg,test_eeg.shape + (1,))
                            
                            # head rotation data
                            train_head = head[train_idx,:]
                            train_head = np.reshape(train_head,train_head.shape + (1,))
                            
                            val_head = head[val_idx,:]
                            val_head = np.reshape(val_head,val_head.shape + (1,))
                    
                            test_head = head[test_idx,:]
                            test_head = np.reshape(test_head,test_head.shape + (1,))
                            
                            # pupil data
                            train_pupil = pupil[train_idx,:]
                            train_pupil = np.reshape(train_pupil,train_pupil.shape + (1,))
                            
                            val_pupil = pupil[val_idx,:]
                            val_pupil = np.reshape(val_pupil,val_pupil.shape + (1,))
                            
                            test_pupil = pupil[test_idx,:]
                            test_pupil = np.reshape(test_pupil,test_pupil.shape + (1,))
                    
                            # dwell time data
                            train_dwell = dwell[:,train_idx]
                            train_dwell = train_dwell.transpose()
                            train_dwell = np.reshape(train_dwell,train_dwell.shape + (1,))
                            
                            val_dwell = dwell[:,val_idx]
                            val_dwell = val_dwell.transpose()
                            val_dwell = np.reshape(val_dwell,val_dwell.shape + (1,))
                            
                            test_dwell = dwell[:,test_idx]
                            test_dwell = test_dwell.transpose()
                            test_dwell = np.reshape(test_dwell,test_dwell.shape + (1,))
                            
                            # TRAIN / TEST COMBINED MODEL
                            if run_combined:
                                EEGnet = EEGNet(l1rate = l1_rate, l2rate = l2_rate, droptoutRate = drp_rate)
                                weightsfilename = 'weights/optimization/CombinedModelWeights_fold' + str(i) +'.hf5'
                                checkpointer = ModelCheckpoint(filepath = weightsfilename, verbose=0,
                                                               save_best_only = True)
                                
                                # Train Combined model
                                hist = EEGnet.model.fit([train_eeg, train_head, train_pupil, train_dwell],y_train,
                                                 batch_size = batch_size, nb_epoch = nb_epochs, verbose=1,
                                                 validation_data = ([val_eeg, val_head, val_pupil, val_dwell],y_val),
                                                 callbacks = [checkpointer], class_weight={0:c0_weight, 1:c1_weight}) 
                                
                                # load in best validation
                                EEGnet.model.load_weights(weightsfilename)
                                
                                # calculate auc
                                probs = EEGnet.model.predict([test_eeg, test_head, test_pupil, test_dwell])
                                fpr, tpr, thresholds = metrics.roc_curve(y_test[:,1], probs[:,1], pos_label=1)
                                AUC = metrics.auc(fpr, tpr)
                                print('Combined AUC: {}, Parms: {},{},{},{},{},{}'.format(AUC,l1_rate,l2_rate,drp_rate,c1_weight,cv,i))
                                AUC_combined_model[idxl1,idxl2,idxdrp,idxc1,cv,idxi] = AUC
                                
                            
                            # TRAIN / TEST EEG MODEL
                            if run_eeg:
                                EEGnet = EEGNet(l1rate = l1_rate, l2rate = l2_rate, droptoutRate = drp_rate)
                                weightsfilename = 'weights/optimization/EEGModelWeights_fold' + str(i) +'.hf5'
                                checkpointer = ModelCheckpoint(filepath = weightsfilename, verbose=0,
                                                               save_best_only = True)
                                
                                # Train EEG model only
                                hist = EEGnet.eeg_model.fit(train_eeg,y_train,
                                                 batch_size = batch_size, nb_epoch = nb_epochs, verbose=1,
                                                 validation_data = (val_eeg,y_val),
                                                 callbacks = [checkpointer], class_weight={0:c0_weight, 1:c1_weight})
                                
                                # load in best validation
                                EEGnet.eeg_model.load_weights(weightsfilename)
                                # calculate auc
                                probs = EEGnet.eeg_model.predict(test_eeg)
                                fpr, tpr, thresholds = metrics.roc_curve(y_test[:,1], probs[:,1], pos_label=1)
                                AUC = metrics.auc(fpr, tpr)
                                print('EEG AUC: {}, Parms: {},{},{},{},{},{}'.format(AUC,l1_rate,l2_rate,drp_rate,c1_weight,cv,i))
                                AUC_eeg_model[idxl1,idxl2,idxdrp,idxc1,cv,idxi] = AUC
                                
                            # TRAIN / TEST HEAD MODEL
                            if run_head:
                                EEGnet = EEGNet(l1rate = l1_rate, l2rate = l2_rate, droptoutRate = drp_rate)
                                weightsfilename = 'weights/optimization/HeadModelWeights_fold' + str(i) +'.hf5'
                                checkpointer = ModelCheckpoint(filepath = weightsfilename, verbose=0,
                                                               save_best_only = True)
                                
                                # Train EEG model only
                                hist = EEGnet.head_rotation_model.fit(train_head,y_train,
                                                 batch_size = batch_size, nb_epoch = nb_epochs, verbose=1,
                                                 validation_data = (val_head,y_val),
                                                 callbacks = [checkpointer], class_weight={0:c0_weight, 1:c1_weight})
                                
                                # load in best validation
                                EEGnet.head_rotation_model.load_weights(weightsfilename)
                                # calculate auc
                                probs = EEGnet.head_rotation_model.predict(test_head)
                                fpr, tpr, thresholds = metrics.roc_curve(y_test[:,1], probs[:,1], pos_label=1)
                                AUC = metrics.auc(fpr, tpr)
                                print('Head AUC: {}, Parms: {},{},{},{},{},{}'.format(AUC,l1_rate,l2_rate,drp_rate,c1_weight,cv,i))
                                AUC_head_model[idxl1,idxl2,idxdrp,idxc1,cv,idxi] = AUC
                                
                            # TRAIN / TEST PUPIL MODEL
                            if run_pupil:
                                EEGnet = EEGNet(l1rate = l1_rate, l2rate = l2_rate, droptoutRate = drp_rate)
                                weightsfilename = 'weights/optimization/PupilModelWeights_fold' + str(i) +'.hf5'
                                checkpointer = ModelCheckpoint(filepath = weightsfilename, verbose=0,
                                                               save_best_only = True)
                                
                                # Train EEG model only
                                hist = EEGnet.pupil_model.fit(train_pupil,y_train,
                                                 batch_size = batch_size, nb_epoch = nb_epochs, verbose=1,
                                                 validation_data = (val_pupil,y_val),
                                                 callbacks = [checkpointer], class_weight={0:c0_weight, 1:c1_weight})
                                
                                # load in best validation
                                EEGnet.pupil_model.load_weights(weightsfilename)
                                # calculate auc
                                probs = EEGnet.pupil_model.predict(test_pupil)
                                fpr, tpr, thresholds = metrics.roc_curve(y_test[:,1], probs[:,1], pos_label=1)
                                AUC = metrics.auc(fpr, tpr)
                                print('Pupil AUC: {}, Parms: {},{},{},{},{},{}'.format(AUC,l1_rate,l2_rate,drp_rate,c1_weight,cv,i))
                                AUC_pupil_model[idxl1,idxl2,idxdrp,idxc1,cv,idxi] = AUC
                                
                            # TRAIN / TEST DWELL MODEL
                            if run_dwell:
                                EEGnet = EEGNet(l1rate = l1_rate, l2rate = l2_rate, droptoutRate = drp_rate)
                                weightsfilename = 'weights/optimization/DwellModelWeights_fold' + str(i) +'.hf5'
                                checkpointer = ModelCheckpoint(filepath = weightsfilename, verbose=0,
                                                               save_best_only = True)
                                
                                # Train EEG model only
                                hist = EEGnet.dwell_model.fit(train_dwell,y_train,
                                                 batch_size = batch_size, nb_epoch = nb_epochs, verbose=1,
                                                 validation_data = (val_dwell,y_val),
                                                 callbacks = [checkpointer], class_weight={0:c0_weight, 1:c1_weight})
                                
                                # load in best validation
                                EEGnet.dwell_model.load_weights(weightsfilename)
                                # calculate auc
                                probs = EEGnet.dwell_model.predict(test_dwell)
                                fpr, tpr, thresholds = metrics.roc_curve(y_test[:,1], probs[:,1], pos_label=1)
                                AUC = metrics.auc(fpr, tpr)
                                print('Dwell AUC: {}, Parms: {},{},{},{},{},{}'.format(AUC,l1_rate,l2_rate,drp_rate,c1_weight,cv,i))
                                AUC_dwell_model[idxl1,idxl2,idxdrp,idxc1,cv,idxi] = AUC
                            
                            results = dict()
                            results['AUC_combined'] = AUC_combined_model
                            results['AUC_eeg'] = AUC_eeg_model
                            results['AUC_head'] = AUC_head_model
                            results['AUC_pupil'] = AUC_pupil_model
                            results['AUC_dwell'] = AUC_dwell_model  
                    
                            sp.io.savemat('results/results_optimization.mat',results)
        
        