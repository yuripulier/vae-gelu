import os

from models.VAE import VariationalAutoencoder
from utils.loaders import load_mnist


# run params
SECTION = 'vae'
RUN_ID = '0002'
DATA_NAME = 'digits'
RUN_FOLDER = '/kaggle/working/run/{}/'.format(SECTION)
RUN_FOLDER += '_'.join([RUN_ID, DATA_NAME])

if not os.path.exists(RUN_FOLDER):
    os.makedirs(RUN_FOLDER)
    os.makedirs(os.path.join(RUN_FOLDER, 'viz'))
    os.makedirs(os.path.join(RUN_FOLDER, 'images'))
    os.makedirs(os.path.join(RUN_FOLDER, 'weights'))

mode =  'build' #'load' #

# Load data
(x_train, y_train), (x_test, y_test) = load_mnist()

# training parameters
LEARNING_RATE = 0.0005
R_LOSS_FACTOR = 1000

BATCH_SIZE = 32
EPOCHS = 200
PRINT_EVERY_N_BATCHES = 100
INITIAL_EPOCH = 0

# architecture and training
list_of_parameters = [[False, False, False, False], [True, False, False, False], [False, True, False, False], [True, True, False, False], # Primeiro experimento
                      [True, True, True, False], [True, True, False, True], [True, True, True, True]] # Segundo e Terceiro experimento

for parameters in list_of_parameters:
    
    encoder_gelu = parameters[0]
    decoder_gelu = parameters[1]
    use_batch_norm = parameters[2]
    use_dropout = parameters[3]
    
    vae = VariationalAutoencoder(
        input_dim = (28,28,1)
        , encoder_conv_filters = [32,64,64, 64]
        , encoder_conv_kernel_size = [3,3,3,3]
        , encoder_conv_strides = [1,2,2,1]
        , decoder_conv_t_filters = [64,64,32,1]
        , decoder_conv_t_kernel_size = [3,3,3,3]
        , decoder_conv_t_strides = [1,2,2,1]
        , z_dim = 2
        , encoder_gelu = encoder_gelu
        , decoder_gelu = decoder_gelu
        , aproximate_gelu = False
        , use_batch_norm = use_batch_norm
        , use_dropout= use_dropout
    )

    if encoder_gelu == False and decoder_gelu == False:
        variant_model = 'base'
    elif encoder_gelu == True and decoder_gelu == False:
        variant_model = 'encoder_gelu'
    elif encoder_gelu == False and decoder_gelu == True:
        variant_model = 'decoder_gelu'
    elif encoder_gelu == True and decoder_gelu == True and use_batch_norm == False:
        variant_model = 'full_gelu'
    elif encoder_gelu == True and decoder_gelu == True and use_batch_norm == True and use_dropout == False:
        variant_model = 'full_gelu_bn'
    elif encoder_gelu == True and decoder_gelu == True and use_batch_norm == False and use_dropout == True:
        variant_model = 'full_gelu_dropout'
    elif encoder_gelu == True and decoder_gelu == True and use_batch_norm == True and use_dropout == True:
        variant_model = 'full_gelu_bn_dropout'
    
    if mode == 'build':
        vae.save(RUN_FOLDER+'/'+variant_model)
    else:
        vae.load_weights(os.path.join(RUN_FOLDER, str('weights/weights-'+variant_model+".h5")))

    vae.compile(LEARNING_RATE, R_LOSS_FACTOR)

    history = vae.train(     
        x_train
        , x_test
        , batch_size = BATCH_SIZE
        , epochs = EPOCHS
        , run_folder = RUN_FOLDER
        , print_every_n_batches = PRINT_EVERY_N_BATCHES
        , initial_epoch = INITIAL_EPOCH
    )

    with open(str('/kaggle/working/trainHistoryDict-'+variant_model), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
