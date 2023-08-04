# For VAE
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras import backend as K

import pandas as pd
import numpy as np
import os
import random
from sklearn.metrics import mean_squared_error

latent_dim = 2
#sequence_length = 41

import struct
from codecs import decode


def set_seed(seed):

    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)


def sampling(args):

    z_mean, z_log_sigma = args
    #batch_size = tf.shape(z_mean)[0]
    batch_size = 1
    epsilon = K.random_normal(
        shape=(batch_size, latent_dim), mean=0., stddev=1.)

    return z_mean + K.exp(0.5 * z_log_sigma) * epsilon
    #return 1


def vae_loss(inp, original, out, z_log_sigma, z_mean, sequence_length):

    reconstruction = K.mean(K.square(original - out)) * sequence_length
    kl = -0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))

    return reconstruction + kl


def get_model(sequence_length):

    map_col = dict()
    map_col['traffic_volume_past'] = 0

    set_seed(33)

    ### encoder ###

    inp = Input(shape=(sequence_length, 1))
    inp_original = Input(shape=(sequence_length, 1))

    cat_inp = []
    cat_emb = []
    for cat, i in map_col.items():
        inp_c = Input(shape=(sequence_length,))
        emb = Embedding(1, 6)(inp_c)
        cat_inp.append(inp_c)
        cat_emb.append(emb)

    concat = Concatenate()(cat_emb + [inp])
    enc = LSTM(64)(concat)

    z = Dense(32, activation="relu")(enc)

    z_mean = Dense(latent_dim)(z)
    z_log_sigma = Dense(latent_dim)(z)

    encoder = Model(cat_inp + [inp], [z_mean, z_log_sigma])

    ### decoder ###

    inp_z = Input(shape=(latent_dim,))

    dec = RepeatVector(sequence_length)(inp_z)
    dec = Concatenate()([dec] + cat_emb)
    dec = LSTM(64, return_sequences=True)(dec)

    out = TimeDistributed(Dense(1))(dec)

    decoder = Model([inp_z] + cat_inp, out)

    ### encoder + decoder ###

    z_mean, z_log_sigma = encoder(cat_inp + [inp])
    z = Lambda(sampling)([z_mean, z_log_sigma])
    pred = decoder([z] + cat_inp)

    vae = Model(cat_inp + [inp, inp_original], pred)
    vae.add_loss(vae_loss(inp, inp_original, pred, z_log_sigma, z_mean, sequence_length))
    vae.compile(loss=None, optimizer=Adam(learning_rate=1e-3))

    return vae, encoder, decoder

### UTILITY FUNCTION FOR 3D SEQUENCE GENERATION ###

def gen_seq(ts, id_df, seq_length, seq_cols, id):

    data_matrix = id_df[seq_cols]
    num_elements = len(data_matrix)

    stop = id+seq_length
    x = data_matrix[stop-seq_length:stop]
    return x


def drop_fill_pieces(sequence_input, sequence_target, missing_val=np.nan):

    sequence_input = np.copy(sequence_input)
    sequence_target = np.copy(sequence_target)

    for i in range(len(sequence_input)):
        sequence_input[i, -1, :] = missing_val
        sequence_target[i, -1, :] = missing_val

    return sequence_input, sequence_target

### UTILITY CLASS FOR SEQUENCES SCALING ###


class Scaler1D:

    def fit(self, X):
        self.mean = np.nanmean(np.asarray(X).ravel())
        self.std = np.nanstd(np.asarray(X).ravel())
        return self

    def transform(self, X):
        return (X - self.mean)/self.std

    def inverse_transform(self, X):
        return (X*self.std) + self.mean


def Return_Model(sequence_length):
    es = EarlyStopping(patience=10, verbose=1, min_delta=0.001,
                       monitor='val_loss', mode='auto', restore_best_weights=True)
    vae, enc, dec = get_model(sequence_length)
    return vae, enc, dec, es


def Train_Model(vae, es, matrix, sets_training, retraining, dataset, sequence_length, epochs, noise="None", level=0):
    
    #gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    #for device in gpu_devices:
    #    tf.config.experimental.set_memory_growth(device, True)

    if retraining:

        es = EarlyStopping(patience=10, verbose=1, min_delta=0.001,
                           monitor='loss', mode='auto', restore_best_weights=True)
        vae, enc, dec = get_model(sequence_length)

        vae.load_weights(f"trained_models/vae-{dataset}_{sequence_length-1}_{epochs}_{noise}_{level}.h5")

        return vae, enc, dec, es

    else:
        a_full = []
        for i in range(matrix.shape[0]):
            a_full.append(np.append([0], matrix[i][:-1]))  # One shifted
        d = {'traffic_volume_past': a_full}
        X = pd.DataFrame(data=d)

        b_full = []
        for i in range(matrix.shape[0]):
            b_full.append(matrix[i])  # One shifted
        d = {'traffic_volume': b_full}
        Y = pd.DataFrame(data=d)

        ### GENERATE 3D SEQUENCES ###

        # sequence_length = 41

        sequence_input = []
        sequence_target = []

        ids = sets_training

        for ts in range(matrix.shape[0]):
            for id in ids:
                seq = gen_seq(
                    ts, X.iloc[ts], sequence_length, 'traffic_volume_past', id)
                sequence_input.append(seq)

                seq = gen_seq(ts, Y.iloc[ts],
                              sequence_length, 'traffic_volume', id)
                sequence_target.append(seq)

        sequence_input = np.asarray(sequence_input)
        sequence_input = sequence_input.reshape(
            sequence_input.shape[0], sequence_input.shape[1], 1)
        sequence_target = np.asarray(sequence_target)
        sequence_target = sequence_target.reshape(
            sequence_target.shape[0], sequence_target.shape[1], 1)

        sequence_input, sequence_target_drop = drop_fill_pieces(
            sequence_input, sequence_target)

        sequence_input_train = sequence_input[:]

        sequence_target_train = sequence_target[:]

        sequence_target_drop_train = sequence_target_drop[:]

        ### SCALE SEQUENCES AND MASK NANs ###

        scaler_target = Scaler1D().fit(sequence_target_train)

        sequence_target_train = scaler_target.transform(sequence_target_train)

        sequence_target_drop_train = scaler_target.transform(
            sequence_target_drop_train)

        mask_value = -999.
        sequence_target_drop_train[np.isnan(
            sequence_target_drop_train)] = mask_value

        es = EarlyStopping(patience=10, verbose=1, min_delta=0.001,
                           monitor='loss', mode='auto', restore_best_weights=True)
        vae, enc, dec = get_model(sequence_length)
        sequence_input_train[np.isnan(sequence_input_train)] = 0
        sequence_input_train[sequence_input_train == 1] = 1 - 0.000001
        print([sequence_input_train[:, :, 0]] + [sequence_target_drop_train, sequence_target_train])
        vae.fit([sequence_input_train[:, :, 0]] + [sequence_target_drop_train, sequence_target_train],
                epochs=epochs, shuffle=False, batch_size=1)
        # , callbacks=[es]

        vae.save_weights(f"trained_models/vae-{dataset}_{sequence_length-1}_{epochs}_{noise}_{level}.h5")

        return vae, enc, dec, es


def Test_Model(vae, matrix, sets_testing, sequence_length):

    a_full = []
    for i in range(matrix.shape[0]):
        a_full.append(np.append([0], matrix[i][:-1]))  # One shifted
    d = {'traffic_volume_past': a_full}
    X = pd.DataFrame(data=d)

    b_full = []
    for i in range(matrix.shape[0]):
        b_full.append(matrix[i])  # One shifted
    d = {'traffic_volume': b_full}
    Y = pd.DataFrame(data=d)

    sequence_input = []
    sequence_target = []

    ids = sets_testing

    for ts in range(matrix.shape[0]):
        for id in ids:
            seq = gen_seq(ts, X.iloc[ts], sequence_length,
                          'traffic_volume_past', id)
            sequence_input.append(seq)

            seq = gen_seq(ts, Y.iloc[ts],
                          sequence_length, 'traffic_volume', id)
            sequence_target.append(seq)

    sequence_input = np.asarray(sequence_input)
    sequence_input = sequence_input.reshape(
        sequence_input.shape[0], sequence_input.shape[1], 1)
    sequence_target = np.asarray(sequence_target)
    sequence_target = sequence_target.reshape(
        sequence_target.shape[0], sequence_target.shape[1], 1)

    sequence_input, sequence_target_drop = drop_fill_pieces(
        sequence_input, sequence_target)

    sequence_input_test = sequence_input[:]

    sequence_target_test = sequence_target[:]

    sequence_target_drop_test = sequence_target_drop[:]

    ### SCALE SEQUENCES AND MASK NANs ###

    scaler_target = Scaler1D().fit(sequence_target_test)

    sequence_target_test = scaler_target.transform(sequence_target_test)

    sequence_target_drop_test = scaler_target.transform(
        sequence_target_drop_test)

    mask_value = -999.
    sequence_target_drop_test[np.isnan(sequence_target_drop_test)] = mask_value

    vae = Model(vae.input[:-1], vae.output)

    sequence_input_test[np.isnan(sequence_input_test)] = 0
    sequence_input_test[sequence_input_test == 1] = 1 - 0.000001

    reconstruc_test = scaler_target.inverse_transform(
        vae.predict([sequence_input_test[:, :, 0]] + [sequence_target_drop_test]))

    dif_vae = []
    labels_full = []
    pred = []

    for i in range(reconstruc_test.shape[0]):
        seq = np.copy(sequence_target_test[i])
        seq[seq == mask_value] = np.nan
        seq = scaler_target.inverse_transform(seq)

        dif_vae.append(np.absolute(seq[sequence_length-1]-reconstruc_test[i][sequence_length-1]))
        labels_full.append(seq[sequence_length-1])
        pred.append(reconstruc_test[i][sequence_length-1])

    error = mean_squared_error(labels_full, pred, squared=False)
    print(f"Testing root mean squared error of testing {(error):.3f}")
    return error
