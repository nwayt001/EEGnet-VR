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

class EEGNet(object):
    def __init__(self, nb_classes=2, numChans = 64, numSamples = 128, regRate = 0.01, dropoutRate = 0.25, type = 'VR', display_models = False):
        self.nb_classes = nb_classes
        self.numChans = numChans
        self.numSamples = numSamples
        self.regRate = regRate
        self.dropoutRate = dropoutRate
        self.display_models = display_models
        self.type = type
        
        if type == 'V1':
            self.EEGNetV1(nb_classes, numChans, numSamples, regRate, dropoutRate)
        elif type == 'V2':
            self.EEGNetv2(nb_classes, numChans, numSamples, regRate, dropoutRate)
        elif type == 'VR':
            self.EEGNetVR()
    
    def EEGNetVR(self, nb_classes=2, numChans=64, num_eeg_samples = 385, regRate=0.001, dropoutRate = 0.2,
                 num_head_rotation_samples = 152, num_pupil_samples=241, num_dwell_time_samples=1):
        
        ## EEG Data
        eeg_input = Input((numChans, num_eeg_samples,1))
        conv1        = Convolution2D(16, numChans, 1, 
                                     W_regularizer = l1l2(l1=regRate, l2=regRate))(eeg_input)
        batchNorm1   = BatchNormalization(axis=1)(conv1)
        elu1         = ELU()(batchNorm1)
        spat_drop1   = SpatialDropout2D(dropoutRate)(elu1)        
        permute_dims = 3, 2, 1
        permute1     = Permute(permute_dims)(spat_drop1)        
        conv2        = Convolution2D(4, 2, 64, border_mode = 'same', 
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
        eeg     = Flatten()(spat_drop3)
        dense1     = Dense(nb_classes)(eeg)
        softmax1   = Activation('softmax')(dense1)
        
        self.eeg_model = Model(input=eeg_input, output=softmax1)        
        if self.display_models:
            print(self.eeg_model.summary())
            plot(self.eeg_model,to_file='EEG_model.png')
        # Head Rotation Data        
        head_rotation_input = Input((num_head_rotation_samples,1))        
        x = Convolution1D(4,20,activation='elu',W_regularizer=l1l2(l1=0.0, l2=regRate))(head_rotation_input)
        x = MaxPooling1D(pool_length=2)(x)
        x = Convolution1D(4,20,activation='elu',W_regularizer=l1l2(l1=0.0, l2=regRate))(x)
        x = MaxPooling1D(pool_length=2)(x)
        x = Convolution1D(4,5,activation='elu',W_regularizer=l1l2(l1=0.0, l2=regRate))(x)
        x = MaxPooling1D(pool_length=2)(x)
        x = Flatten()(x)
        head_rotation = Dense(5,activation='elu',W_regularizer=l1l2(l1=0.0, l2=regRate))(x)
        head_rotation_output = Dense(nb_classes,activation='softmax')(head_rotation)
        self.head_rotation_model = Model(input = head_rotation_input, output = head_rotation_output)
        if self.display_models:
            print(self.head_rotation_model.summary())
            plot(self.head_rotation_model,to_file='head_rotation_model.png')
        # pupilometry Data
        pupil_input = Input((num_pupil_samples,1))        
        x = AveragePooling1D(pool_length=2)(pupil_input)
        #x = AveragePooling1D(pool_length=2)(x)
        x = Convolution1D(4,20,activation='elu',W_regularizer=l1l2(l1=0.0, l2=regRate))(x)
        x = MaxPooling1D(pool_length=2)(x)
        x = Convolution1D(4,10,activation='elu',W_regularizer=l1l2(l1=0.0, l2=regRate))(x)
        x = MaxPooling1D(pool_length=2)(x)
        x = Convolution1D(4,10,activation='elu',W_regularizer=l1l2(l1=0.0, l2=regRate))(x)
        x = MaxPooling1D(pool_length=2)(x)
        x = Flatten()(x)
        pupil = Dense(5,activation='elu',W_regularizer=l1l2(l1=0.0, l2=regRate))(x)
        pupil_output = Dense(nb_classes,activation='softmax')(pupil)
        self.pupil_model = Model(input = pupil_input, output = pupil_output)
        if self.display_models:
            print(self.pupil_model.summary())
            plot(self.pupil_model,to_file='pupil_model.png')
        
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
        combined_output = Dense(nb_classes, activation='softmax',W_regularizer=l1l2(l1=0.0, l2=regRate))(merged_features)
        
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
            
    def EEGNetV1(self, nb_classes, numChans = 64, numSamples = 128, 
               regRate = 0.01, dropoutRate = 0.25):
        """ Keras implementation of EEGNet (arXiv 1611.08024)
        This model is only defined for 128Hz signals. For any other sampling rate
        you'll need to scale the length of the kernels at layer conv2 and conv3
        appropriately (double the length for 256Hz, half the length for 64Hz, etc.)
        
        This also implements a slight variant of the original EEGNet article:
            1. stride is used instead of maxpooling
            2. spatialdropout is used instead of dropout
        
        @params 
        nb_classes: total number of final categories
        numChans: number of EEG channels
        numSamples: number of EEG sample points per trial
        regRate: regularization rate for L1 and L2 regularizations
        dropoutRate: dropout fraction
        
        Assumes a pretty recent version of keras (version 1.1.2 is known to work) 
        and either Theano 0.8.2 or Tensorflow 0.11
        """
    
        # start the model
        input_main   = Input((1, numChans, numSamples))
        conv1        = Convolution2D(16, numChans, 1, 
                                     input_shape=(1, numChans, numSamples),
                                     W_regularizer = l1l2(l1=regRate, l2=regRate))(input_main)
        batchNorm1   = BatchNormalization(axis=1)(conv1)
        elu1         = ELU()(batchNorm1)
        spat_drop1   = SpatialDropout2D(dropoutRate)(elu1)
        
        permute_dims = 2, 1, 3
        permute1     = Permute(permute_dims)(spat_drop1)
        
        conv2        = Convolution2D(4, 2, 32, border_mode = 'same', 
                                W_regularizer=l1l2(l1=0.0, l2=regRate),
                                subsample = (2, 4))(permute1)
        batchNorm2   = BatchNormalization(axis=1)(conv2)
        elu2         = ELU()(batchNorm2)
        spat_drop2   = SpatialDropout2D(dropoutRate)(elu2)
        
        conv3        = Convolution2D(4, 8, 4, border_mode = 'same',
                                W_regularizer=l1l2(l1=0.0, l2=regRate),
                                subsample = (2, 4))(spat_drop2)
        batchNorm3   = BatchNormalization(axis=1)(conv3)
        elu3         = ELU()(batchNorm3)
        spat_drop3   = SpatialDropout2D(dropoutRate)(elu3)
        flatten1     = Flatten()(spat_drop3)
        
        dense1     = Dense(nb_classes)(flatten1)
        softmax1   = Activation('softmax')(dense1)
        
        self.model = Model(input=input_main, output=softmax1)
        if self.display_models:
            print(self.model.summary())
    
    
    def EEGNetv2(self, nb_classes, numChans = 64, numSamples = 128, 
               regRate = 0.01, dropoutRate = 0.25):
        """ An experimental version of a new version of EEGNet which tries to use
        all kernel sizes at layers 2 and 3 simultaneously, using the Keras 
        functional API. The number of parameters in this model is kept 
        fairly low due to the use of bottleneck/squeeze layers, which are 1x1 conv 
        layers with fewer output kernels than input kernels.
        
        As with the original EEGNet model, this model is defined for 128Hz signals
        only. You will need to scale the length of the kernels in layers 2 and 3
        to match the sampling rate of your signal. 
        
        @params 
        nb_classes: total number of final categories
        numChans: number of EEG channels
        numSamples: number of EEG sample points per trial
        regRate: regularization rate for L1 and L2 regularizations
        dropoutRate: dropout fraction
        
        Assumes a pretty recent version of keras (version 1.1.2 is known to work) 
        and either Theano 0.8.2 or Tensorflow 0.11
        """
        
        # kernels ranging from more temporal to more spatial at layers 2/3
        configs  = [[16, 4], [8, 8], [4, 16], [2, 32]]
        configs2 = [[8, 4], [4, 8], [2, 16]]
        
        input_main   = Input(shape=((1, numChans, numSamples)))
        conv1        = Convolution2D(16, numChans, 1, 
                              W_regularizer = l1l2(l1=regRate, l2=regRate), name='conv1')(input_main)
        BatchNorm1   = BatchNormalization(axis=1, name='BatchNorm1')(conv1) 
        elu1         = ELU()(BatchNorm1)
        drop1        = SpatialDropout2D(dropoutRate)(elu1)
        
        permute_dims = 2, 1, 3
        permute1     = Permute(permute_dims)(drop1)    
        
        # first branching layers with 4 branches
        # each branch looks at 1 of 4 different kernel sizes
        branch1_1  = Convolution2D(4, configs[0][0], configs[0][1], 
                                     border_mode='same', name='conv2_1',
                                     W_regularizer = l1l2(l1=0., l2=regRate),
                                     subsample = (2, 4))(permute1)
        branch1_1  = BatchNormalization(axis=1, name='BatchNorm2_1')(branch1_1)
        branch1_1  = ELU()(branch1_1)
        branch1_1  = SpatialDropout2D(dropoutRate)(branch1_1)
       
        
        branch1_2  = Convolution2D(4, configs[1][0], configs[1][1], 
                                     border_mode='same', name='conv2_2',
                                     W_regularizer = l1l2(l1=0., l2=regRate),
                                     subsample = (2, 4))(permute1)
        branch1_2  = BatchNormalization(axis=1, name='BatchNorm2_2')(branch1_2)
        branch1_2  = ELU()(branch1_2)
        branch1_2  = SpatialDropout2D(dropoutRate)(branch1_2)
    
        
        branch1_3  = Convolution2D(4, configs[2][0], configs[2][1], 
                                     border_mode='same', name='conv2_3', 
                                     W_regularizer = l1l2(l1=0., l2=regRate),
                                     subsample = (2, 4))(permute1)
        branch1_3  = BatchNormalization(axis=1, name='BatchNorm2_3')(branch1_3)
        branch1_3  = ELU()(branch1_3)
        branch1_3  = SpatialDropout2D(dropoutRate)(branch1_3)
        
        
        branch1_4  = Convolution2D(4, configs[3][0], configs[3][1], 
                                     border_mode='same', name='conv2_4',
                                     W_regularizer = l1l2(l1=0., l2=regRate),
                                     subsample = (2, 4))(permute1)
        branch1_4  = BatchNormalization(axis=1, name='BatchNorm2_4')(branch1_4)
        branch1_4  = ELU()(branch1_4)
        branch1_4  = SpatialDropout2D(dropoutRate)(branch1_4)
    
        
        # merge the branches, then perform the bottleneck/squeeze
        merge1     = merge([branch1_1, branch1_2, branch1_3, branch1_4], 
                             mode='concat', concat_axis=1)
        merge1     = Convolution2D(4, 1, 1)(merge1)
        merge1     = ELU()(merge1)
     
        # second branching layer with 3 branches
        # each branch looks at 1 of 3 different kernel sizes
        branch2_1  = Convolution2D(4, configs2[0][0], configs2[0][1], 
                                     border_mode='same', name='conv3_1',
                                     W_regularizer = l1l2(l1=0., l2=regRate),
                                     subsample = (2, 4))(merge1)
        branch2_1  = BatchNormalization(axis=1, name='BatchNorm3_1')(branch2_1)
        branch2_1  = ELU()(branch2_1)
        branch2_1  = SpatialDropout2D(dropoutRate)(branch2_1)
    
    
        branch2_2  = Convolution2D(4, configs2[1][0], configs2[1][1], 
                                     border_mode='same', name='conv3_2',
                                     W_regularizer = l1l2(l1=0., l2=regRate),
                                     subsample = (2, 4))(merge1)
        branch2_2  = BatchNormalization(axis=1, name='BatchNorm3_2')(branch2_2)
        branch2_2  = ELU()(branch2_2)
        branch2_2  = SpatialDropout2D(dropoutRate)(branch2_2)
       
        
        branch2_3  = Convolution2D(4, configs2[2][0], configs2[2][1], 
                                     border_mode='same', name='conv3_3',
                                     W_regularizer = l1l2(l1=0., l2=regRate),
                                     subsample = (2, 4))(merge1)
        branch2_3  = BatchNormalization(axis=1, name='BatchNorm3_3')(branch2_3)
        branch2_3  = ELU()(branch2_3)
        branch2_3  = SpatialDropout2D(dropoutRate)(branch2_3)
     
        # merge the branches, then perform the bottleneck/squeeze
        merge2     = merge([branch2_1, branch2_2, branch2_3], mode='concat', 
                             concat_axis=1)
        merge2     = Convolution2D(4, 1, 1, name='squeeze9')(merge2)
        merge2     = ELU()(merge2)
    
        # flatten, then go to softmax
        flatten    = Flatten(name='flatten')(merge2)
        dense1     = Dense(nb_classes)(flatten)
        softmax    = Activation("softmax", name='softmax')(dense1)
    
        self.model = Model(input=input_main, output=softmax)
        if self.display_models:
            print(self.model.summary())
    
if __name__ == '__main__':
    np.random.seed(123)
    
    # Load Training Data
    data = sp.io.loadmat('/home/waytowich/Dropbox/Data/training_data_v4/training_data.mat')
    
    # Train/Test which models
    run_combined = True
    run_eeg = True
    run_head = True
    run_pupil = True
    run_dwell = True

    
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
                EEGnet = EEGNet(type = 'VR')
                weightsfilename = 'weights/jenn/CombinedModelWeights_fold' + str(i) +'.hf5'
                checkpointer = ModelCheckpoint(filepath = weightsfilename, verbose=0,
                                               save_best_only = True)
                
                # Train Combined model
                hist = EEGnet.model.fit([train_eeg, train_head, train_pupil, train_dwell],y_train,
                                 batch_size = batch_size, nb_epoch = nb_epochs, verbose=1,
                                 validation_data = ([val_eeg, val_head, val_pupil, val_dwell],y_val),
                                 callbacks = [checkpointer], class_weight={0:c0_weight, 1:c1_weight}) #class_weight = {0:1., 1:1.25}
                
                # load in best validation
                EEGnet.model.load_weights(weightsfilename)
                
                # calculate auc
                probs = EEGnet.model.predict([test_eeg, test_head, test_pupil, test_dwell])
                fpr, tpr, thresholds = metrics.roc_curve(y_test[:,1], probs[:,1], pos_label=1)
                AUC = metrics.auc(fpr, tpr)
                print('Combined Model AUC: ', AUC)
                AUC_combined_model.append(AUC)
                
            
            # TRAIN / TEST EEG MODEL
            if run_eeg:
                EEGnet = EEGNet(type = 'VR')
                weightsfilename = 'weights/jenn/EEGModelWeights_fold' + str(i) +'.hf5'
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
                print(['EEG Model AUC: ', AUC])
                AUC_eeg_model.append(AUC)
                
            # TRAIN / TEST HEAD MODEL
            if run_head:
                EEGnet = EEGNet(type = 'VR')
                weightsfilename = 'weights/jenn/HeadModelWeights_fold' + str(i) +'.hf5'
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
                print(['Head Model AUC: ', AUC])
                AUC_head_model.append(AUC)
                
            # TRAIN / TEST PUPIL MODEL
            if run_pupil:
                EEGnet = EEGNet(type = 'VR')
                weightsfilename = 'weights/jenn/PupilModelWeights_fold' + str(i) +'.hf5'
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
                print(['Pupil Model AUC: ', AUC])
                AUC_pupil_model.append(AUC)
                
            # TRAIN / TEST DWELL MODEL
            if run_dwell:
                EEGnet = EEGNet(type = 'VR')
                weightsfilename = 'weights/jenn/DwellModelWeights_fold' + str(i) +'.hf5'
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
                print(['Dwell Model AUC: ', AUC])
                AUC_dwell_model.append(AUC)
            
        # save AUC results
        avg_auc_combined = np.mean(AUC_combined_model)
        avg_auc_eeg = np.mean(AUC_eeg_model)
        avg_auc_head = np.mean(AUC_head_model)
        avg_auc_pupil = np.mean(AUC_pupil_model)
        avg_auc_dwell = np.mean(AUC_dwell_model)
        
        print('Avg AUC Combined Model: ', avg_auc_combined)
        print('Avg AUC EEG Model: ', avg_auc_eeg)
        print('Avg AUC Head Model: ', avg_auc_head)
        print('Avg AUC Pupil Model: ', avg_auc_pupil)
        print('Avg AUC Dwell Model: ', avg_auc_dwell)
        
        print('subject wise AUC Combined Model:')
        print(AUC_combined_model)
        print('subject wise AUC EEG Model:')
        print(AUC_eeg_model)
        print('subject wise AUC head Model:')
        print(AUC_head_model)
        print('subject wise AUC pupil Model:')
        print(AUC_pupil_model)
        print('subject wise AUC dwell Model:')
        print(AUC_dwell_model)
        
        results = dict()
        results['AUC_combined'] = AUC_combined_model
        results['AUC_eeg'] = AUC_eeg_model
        results['AUC_head'] = AUC_head_model
        results['AUC_pupil'] = AUC_pupil_model
        results['AUC_dwell'] = AUC_dwell_model
        results['avg_AUC_combined'] = avg_auc_combined
        results['avg_AUC_eeg'] = avg_auc_eeg
        results['avg_AUC_head'] = avg_auc_head
        results['avg_AUC_pupil'] = avg_auc_pupil
        results['avg_AUC_dwell'] = avg_auc_dwell
    
        sp.io.savemat('results/results_jenn_r5_125_1.mat',results)

    
        