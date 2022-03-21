# keras layers
import keras
from keras.models import Model
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, AveragePooling1D
from keras.layers import Embedding, Dropout, Input
from tfdeterminism import patch
patch()

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


def enhanced_cnn_stack(emb, ks, input_shape, dr=1):
    cnn1 = Conv1D(filters=128, kernel_size=ks, padding='same', activation='relu')(
        emb)
    cnn1 = MaxPooling1D(pool_size=input_shape)(cnn1)
    cnn1 = Flatten()(cnn1)
    return cnn1


def mcnn(input_shape, vocab_size, n_class, embedding_dim, embedding_matrix, ad=False):
    """
        mcnn model

        :param seq_len: Sequence Length
        :param vocab_size: Number of words in the vocabulary
        :param n_class: Number of output neurons
        :param embedding_dim: Dimension of embedding vector of each word
        :param embedding_matrix: Word to embedding vector matrix
        :param ad: internal flag for building admcnn
        """
    model_input = Input(shape=(input_shape,), name="text_input")

    emb = Embedding(vocab_size + 1, embedding_dim, weights=[embedding_matrix],
                    input_length=input_shape,
                    trainable=False)(model_input)

    if ad:
        emb = self_attention_head(emb)

    #ecnn1 = enhanced_cnn_stack(emb, 3, input_shape, dr=4)
    #ecnn2 = enhanced_cnn_stack(emb, 4, input_shape, dr=2)
    #ecnn3 = enhanced_cnn_stack(emb, 5, input_shape, dr=1)

    #model = keras.layers.concatenate([ecnn1, ecnn2, ecnn3])
    model = AveragePooling1D(pool_size=input_shape)(emb)
    model = Flatten()(model)

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
                ad=True)
