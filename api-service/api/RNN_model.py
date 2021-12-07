import os
import json
import numpy as np
import tensorflow as tf
from functools import lru_cache

RNN_MODEL_PATH = "/../persistent/rnn_model"
EXAMPLE_IMAGE_PATH = "/../persistent/image/test_image.jpeg"
embedding_dim = 256
units = 512
top_k = 10000
max_length = 52
vocab_size = top_k + 1
# Shape of the vector extracted from InceptionV3 is (64, 2048)
# These two variables represent that vector shape
features_shape = 2048
attention_features_shape = 64


# # load image feature extraction model
# image_model = tf.keras.applications.InceptionV3(include_top=False,weights='imagenet')
# new_input = image_model.input
# hidden_layer = image_model.layers[-1].output
# image_features_extract_model = Model(new_input, hidden_layer)

# load image feature extraction model
@lru_cache
def load_cnn_model():
    image_model = tf.keras.applications.InceptionV3(include_top=False,weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output
    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
    return image_features_extract_model


def generate_caption_rnn(img_path):
    cnn, encoder, decoder, tokenizer = build_model_and_load_weights_rnn()

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_and_process_image_cnn(img_path), 0)
    img_tensor_val = cnn(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0],
                                                 -1,
                                                 img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input,
                                                         features,
                                                         hidden)

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            caption = ' '.join(result[1:-1])
            return {'caption': caption}

        dec_input = tf.expand_dims([predicted_id], 0)

    caption = ' '.join(result[1:-1])
    return {'caption': caption}


@lru_cache
def build_model_and_load_weights_rnn():
    tokenizer = load_tokenizer_rnn()
    cnn = load_cnn_model()
    encoder = CNN_Encoder(embedding_dim)
    decoder = RNN_Decoder(embedding_dim, units, vocab_size)

    hidden = decoder.reset_state(batch_size=1)
    temp_input = tf.expand_dims(load_and_process_image_cnn(EXAMPLE_IMAGE_PATH), 0)
    img_tensor_val = cnn(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0],
                                                -1,
                                                img_tensor_val.shape[3]))

    #encoder.build((None, *img_tensor_val.shape[1:]))
    features = encoder(img_tensor_val)
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

    encoder.load_weights(os.path.join(RNN_MODEL_PATH,"encoder.h5"))
    decoder.load_weights(os.path.join(RNN_MODEL_PATH,"decoder.h5"))

    return cnn, encoder, decoder, tokenizer


def load_and_process_image_cnn(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img

@lru_cache
def load_tokenizer_rnn():
    # load tokenizer
    with open(os.path.join(RNN_MODEL_PATH,'tokenizer.json')) as f:
        data = json.load(f)
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)
    return tokenizer


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # attention_hidden_layer shape == (batch_size, 64, units)
        attention_hidden_layer = (tf.nn.tanh(self.W1(features) +
                                            self.W2(hidden_with_time_axis)))

        # score shape == (batch_size, 64, 1)
        # This gives you an unnormalized score for each image feature.
        score = self.V(attention_hidden_layer)

        # attention_weights shape == (batch_size, 64, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x

class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
        # defining attention as a separate model
        context_vector, attention_weights = self.attention(features, hidden)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)

        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))