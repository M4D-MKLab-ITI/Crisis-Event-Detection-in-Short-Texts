import numpy as np

# keras layers
import keras
from keras.models import Model
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, AveragePooling1D, GRU, GlobalAveragePooling1D
from keras.layers import Embedding, Dropout, Input


# custom keras layers
from keras_multi_head import MultiHeadAttention
from keras_layer_normalization import LayerNormalization

# my modules
from .my_layers import p_w_ff, Positional_Encoding


def self_attention_head(emb):
    att = MultiHeadAttention(head_num=5)(emb)
    emb = keras.layers.add([emb, att])
    emb = LayerNormalization()(emb)
    ff = p_w_ff()(emb)
    emb = keras.layers.add([emb, ff])
    return LayerNormalization()(emb)


def cnn(emb, ks, input_shape, ma=False):
    cnn1 = Conv1D(filters=128, kernel_size=int(ks), padding='same', activation='relu')(
        emb)
    if ma:
        cnn1 = MaxPooling1D(pool_size=int(input_shape / 10))(cnn1)
    else:
        cnn1 = MaxPooling1D(pool_size=input_shape)(cnn1)
        cnn1 = Flatten()(cnn1)
    return cnn1


def cnn_parallel_stack(emb, seq_len, ma=False):
    kernels = np.array([3, 4, 5])
    if ma:
        kernels -= 1
    cnn1 = cnn(emb, kernels[0], seq_len, ma)
    cnn2 = cnn(emb, kernels[1], seq_len, ma)
    cnn3 = cnn(emb, kernels[2], seq_len, ma)
    return keras.layers.concatenate([cnn1, cnn2, cnn3])


def attention_stack(model, num_encoders=1):
    for i in range(num_encoders):
        model = self_attention_head(model)
    return model


def mcnn(seq_len, vocab_size, n_class, embedding_dim, embedding_matrix, ad=False, ma=False, positional=False):
    """
        mcnn model

        :param seq_len: Sequence Length
        :param vocab_size: Number of words in the vocabulary
        :param n_class: Number of output neurons
        :param embedding_dim: Dimension of embedding vector of each word
        :param embedding_matrix: Word to embedding vector matrix
        :param ad: internal flag for building admcnn
        :param ma: internal flag for building mcnnma
        :param positional: positional encoding flag
        """
    model_input = Input(shape=(seq_len,), name="text_input")

    emb = Embedding(vocab_size + 1, embedding_dim, weights=[embedding_matrix],
                    input_length=seq_len,
                    trainable=False)(model_input)
    if positional:
        emb = Positional_Encoding()(emb)


    if ad:
        emb = attention_stack(emb)

    model = cnn_parallel_stack(emb, seq_len, ma)

    if ma:
        model = MultiHeadAttention(head_num=4)(model)
        model = GlobalAveragePooling1D()(model)

    model = Dropout(0.5)(model)
    model = Dense(n_class, activation='softmax')(model)
    model = Model(inputs=model_input, outputs=model)
    return model


def ad_mcnn(seq_len, vocab_size, n_class, embedding_dim, embedding_matrix) -> Model:
    """
    admcnn model

    :param seq_len: Sequence Length
    :param vocab_size: Number of words in the vocabulary
    :param n_class: Number of output neurons
    :param embedding_dim: Dimension of embedding vector of each word
    :param embedding_matrix: Word to embedding vector matrix
    """
    return mcnn(seq_len,
                vocab_size,
                n_class,
                embedding_dim,
                embedding_matrix,
                ad=True,
                ma=False)


def ad_pgru(seq_len, vocab_size, n_class, embedding_dim, embedding_matrix) -> Model:
    """
    mcnn model

    :param seq_len: Sequence Length
    :param vocab_size: Number of words in the vocabulary
    :param n_class: Number of output neurons
    :param embedding_dim: Dimension of embedding vector of each word
    :param embedding_matrix: Word to embedding vector matrix
    """
    model_input = Input(shape=(seq_len,), name="text_input")

    emb = Embedding(vocab_size + 1, embedding_dim, weights=[embedding_matrix],
                    input_length=seq_len,
                    trainable=False)(model_input)

    emb = attention_stack(emb)

    gru1 = GRU(400)(emb)
    gru2 = GRU(400)(emb)
    gru3 = GRU(400)(emb)

    model = keras.layers.concatenate([gru3, gru1, gru2])

    model = Dropout(0.5)(model)
    model = Dense(n_class, activation='softmax')(model)
    model = Model(inputs=model_input, outputs=model)
    return model


def stacked_sae(seq_len, vocab_size, n_class, embedding_dim, embedding_matrix) -> Model:
    """
    mcnn model

    :param seq_len: Sequence Length
    :param vocab_size: Number of words in the vocabulary
    :param n_class: Number of output neurons
    :param embedding_dim: Dimension of embedding vector of each word
    :param embedding_matrix: Word to embedding vector matrix
    """
    model_input = Input(shape=(seq_len,), name="text_input")

    emb = Embedding(vocab_size + 1, embedding_dim, weights=[embedding_matrix],
                    input_length=seq_len,
                    trainable=False)(model_input)

    emb = attention_stack(emb, num_encoders=4)
    model = AveragePooling1D(pool_size=seq_len)(emb)
    model = Flatten()(model)
    model = Dropout(0.5)(model)
    model = Dense(n_class, activation='softmax')(model)
    model = Model(inputs=model_input, outputs=model)
    return model


def mcnn_ma(seq_len, vocab_size, n_class, embedding_dim, embedding_matrix):
    """
        mcnn-ma model

        :param seq_len: Sequence Length
        :param vocab_size: Number of words in the vocabulary
        :param n_class: Number of output neurons
        :param embedding_dim: Dimension of embedding vector of each word
        :param embedding_matrix: Word to embedding vector matrix
        """
    return mcnn(seq_len,
                vocab_size,
                n_class,
                embedding_dim,
                embedding_matrix,
                ad=False,
                ma=True)