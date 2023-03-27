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

latent_dim = 2
sequence_length = 41


def set_seed(seed):

    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)


def sampling(args):

    z_mean, z_log_sigma = args
    batch_size = tf.shape(z_mean)[0]
    epsilon = K.random_normal(
        shape=(batch_size, latent_dim), mean=0., stddev=1.)

    return z_mean + K.exp(0.5 * z_log_sigma) * epsilon


def vae_loss(inp, original, out, z_log_sigma, z_mean):

    reconstruction = K.mean(K.square(original - out)) * sequence_length
    kl = -0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))

    return reconstruction + kl


def get_model():

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

    # print(cat_inp)
    # print(cat_emb)

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
    vae.add_loss(vae_loss(inp, inp_original, pred, z_log_sigma, z_mean))
    vae.compile(loss=None, optimizer=Adam(lr=1e-3))

    return vae, encoder, decoder

### UTILITY FUNCTION FOR 3D SEQUENCE GENERATION ###


def gen_seq(ts, id_df, seq_length, seq_cols, id):

    data_matrix = id_df[seq_cols]
    num_elements = len(data_matrix)

    stop = id+seq_length
    return data_matrix[stop-sequence_length:stop]


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


def Return_Model():
    es = EarlyStopping(patience=10, verbose=1, min_delta=0.001,
                       monitor='val_loss', mode='auto', restore_best_weights=True)
    vae, enc, dec = get_model()
    return vae, enc, dec, es


def Train_Model(vae, es, matrix, sets_training, retraining, dataset):

    if retraining:

        es = EarlyStopping(patience=10, verbose=1, min_delta=0.001,
                           monitor='loss', mode='auto', restore_best_weights=True)
        vae, enc, dec = get_model()

        vae.load_weights(f"trained_models/vae-{dataset}.h5")

        return vae, enc, dec, es

    else:

        a_full = []
        for i in range(matrix.shape[0]):
            a_full.append(np.append([0], matrix[i][:-1]))  # One shifted
        d = {'traffic_volume_past': a_full}
        X = pd.DataFrame(data=d)
        print(len(X['traffic_volume_past']))
        print(len(X['traffic_volume_past'].iloc[0]))

        b_full = []
        for i in range(matrix.shape[0]):
            b_full.append(matrix[i])  # One shifted
        d = {'traffic_volume': b_full}
        Y = pd.DataFrame(data=d)
        print(len(Y['traffic_volume']))
        print(len(Y.iloc[0]['traffic_volume']))

        ### GENERATE 3D SEQUENCES ###

        sequence_length = 41

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

        print(sequence_input.shape, sequence_target.shape)

        sequence_input, sequence_target_drop = drop_fill_pieces(
            sequence_input, sequence_target)

        print(sequence_input.shape, sequence_target_drop.shape)

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
        vae, enc, dec = get_model()
        vae.fit([sequence_input_train[:, :, 0]] + [sequence_target_drop_train, sequence_target_train],
                epochs=10, shuffle=False, callbacks=[es])

        vae.save_weights(f"trained_models/vae-{dataset}.h5")

        return vae, enc, dec, es


def Test_Model(vae, matrix, sets_testing):

    a_full = []
    for i in range(matrix.shape[0]):
        a_full.append(np.append([0], matrix[i][:-1]))  # One shifted
    d = {'traffic_volume_past': a_full}
    X = pd.DataFrame(data=d)
    print(len(X['traffic_volume_past']))
    print(len(X['traffic_volume_past'].iloc[0]))

    b_full = []
    for i in range(matrix.shape[0]):
        b_full.append(matrix[i])  # One shifted
    d = {'traffic_volume': b_full}
    Y = pd.DataFrame(data=d)
    print(len(Y['traffic_volume']))
    print(len(Y.iloc[0]['traffic_volume']))

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

    print(sequence_input.shape, sequence_target.shape)

    sequence_input, sequence_target_drop = drop_fill_pieces(
        sequence_input, sequence_target)

    print(sequence_input.shape, sequence_target_drop.shape)

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

    reconstruc_test = scaler_target.inverse_transform(
        vae.predict([sequence_input_test[:, :, 0]] + [sequence_target_drop_test]))

    dif_vae = []

    for i in range(reconstruc_test.shape[0]):
        seq = np.copy(sequence_target_test[i])
        seq[seq == mask_value] = np.nan
        seq = scaler_target.inverse_transform(seq)

        dif_vae.append(np.absolute(seq[40]-reconstruc_test[i][40]))

    print(len(dif_vae))

    return vae, dif_vae
