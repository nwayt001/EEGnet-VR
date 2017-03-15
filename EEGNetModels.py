from keras.models import Model
from keras.layers.core import Dense, Activation, Permute, SpatialDropout2D
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.regularizers import l1l2
from keras.layers import Input, Flatten, merge


def EEGNet(nb_classes, numChans = 64, numSamples = 128, 
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
    
    return Model(input=input_main, output=softmax1)
    
    
    
def EEGNetv2(nb_classes, numChans = 64, numSamples = 128, 
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

    return Model(input=input_main, output=softmax)

    
    
