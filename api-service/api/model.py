import os
import sys
import re
import numpy as np
import pickle
from PIL import Image
from functools import lru_cache

# Tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization

# Pytorch and CLIP
import torch
import clip

## Define global parameters
# Path to model weights and vectorization
TRANSFORMER_MODEL_PATH = "/../persistent/transformer_model"
PREFIX_MODEL_PATH = "/../persistent/prefix_model"
EXAMPLE_IMAGE_PATH = "/../persistent/image/test_image.jpeg"
sys.path.append(TRANSFORMER_MODEL_PATH)
sys.path.append(EXAMPLE_IMAGE_PATH)
# Vocabulary size
VOCAB_SIZE = 15000
# Made up length for the transformed image feature
IMAGE_LENGTH = 16
# Fixed length allowed for any sequence
SEQ_LENGTH = 25
# Number of attention heads
NUM_HEADS = 10
# Dimension for the image embeddings and token embeddings
EMBED_DIM = 512
# Per-layer units in the feed-forward network
FF_DIM = EMBED_DIM*4
# Number of encoder blocks
NUM_ENC_LAYERS = 2
# Number of decoder blocks
NUM_DEC_LAYERS = 6

# claim clip model
global device
device = "cuda" if torch.cuda.is_available() else "cpu"

@lru_cache
def load_clip():
    print('Download CLIP model...')
    clip_model, clip_preprocess = clip.load("ViT-B/16", device=device)
    print('Done!')
    return clip_model, clip_preprocess

def load_and_process_image(path):
    # load clip model and clip preprocess
    clip_model, clip_preprocess = load_clip()
    
    # load image and preprocess
    image = clip_preprocess(Image.open(path)).unsqueeze(0).to(device)

    # embed preprocessed image with clip encoder
    if device == 'cpu':
        with torch.no_grad():
            embedded_img = clip_model.encode_image(image).numpy()
    elif device == 'cuda':
        with torch.no_grad():
            embedded_img = clip_model.encode_image(image).cpu().numpy()
    return embedded_img


def generate_caption_transformer(img_path):
    caption_model, vectorization = build_model_and_load_weights_transformer()

    # get index_word and word_index conversion
    vocab = vectorization.get_vocabulary()
    index_word = dict(zip(range(len(vocab)), vocab))
    # word_index = dict(zip(vocab, range(len(vocab))))

    # Load raw image and embed with CLIP
    embedded_img = load_and_process_image(img_path)

    # Pass the image features to the Transformer encoder
    encoded_img = caption_model.encoder(embedded_img, training=False)

    # Generate the caption using the Transformer decoder
    decoded_caption = "<start>"
    for i in range(SEQ_LENGTH - 1):
        tokenized_caption = vectorization([decoded_caption])[:, :-1]
        mask = tf.math.not_equal(tokenized_caption, 0)
        predictions = caption_model.decoder(
            tokenized_caption, encoded_img, training=False, mask=mask
        )
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = index_word[sampled_token_index]
        if sampled_token == "<end>":
            break
        decoded_caption += " " + sampled_token

    decoded_caption = decoded_caption.replace("<start>", "")
    decoded_caption = decoded_caption.replace("<end>", "").strip()
    # print("Predicted Caption: ", decoded_caption)
    return {'caption':decoded_caption}

def generate_caption_prefix(img_path):
    caption_model, vectorization = build_model_and_load_weights_prefix()

    # get index_word and word_index conversion
    vocab = vectorization.get_vocabulary()
    index_word = dict(zip(range(len(vocab)), vocab))
    # word_index = dict(zip(vocab, range(len(vocab))))

    # Load raw image and embed with CLIP
    embedded_img = load_and_process_image(img_path)

    # Pass the image features to the Transformer encoder
    encoder_output = caption_model.encoder(embedded_img, training=False)
    prefix_length = tf.shape(encoder_output)[1]

    # Generate the caption using the Transformer decoder
    decoded_caption = "<start>"
    for i in range(SEQ_LENGTH - 1):
        tokenized_caption = vectorization([decoded_caption])

        # concatenate dummy tokens for prefix and token
        caption_prefix = tf.concat([VOCAB_SIZE*tf.ones((1, prefix_length),dtype=tf.int64), tokenized_caption], axis=1)

        # shift to create input and target
        seq_inp = tokenized_caption[:, :-1]
        seq_true = caption_prefix[:, 1:]

        # obtain mask (0 is the pad token)
        mask = tf.math.not_equal(seq_true, 0)

        # pass thtought decoder
        predictions = caption_model.decoder(
            seq_inp, encoder_output, training=False, mask=mask
        )

        # extract predicted token (location: i + prefix_length - 1)
        sampled_token_index = np.argmax(predictions[0, i + prefix_length, :])
        sampled_token = index_word[sampled_token_index]
        #print(sampled_token)
        if sampled_token == "<end>":
            break
        decoded_caption += " " + sampled_token

    decoded_caption = decoded_caption.replace("<start>", "")
    decoded_caption = decoded_caption.replace("<end>", "").strip()

    return {'caption':decoded_caption}


@lru_cache
def build_model_and_load_weights_transformer():
    ## load saved vectorization
    vectorization_weight = pickle.load(open(os.path.join(TRANSFORMER_MODEL_PATH,"vectorization_weights.pkl"), "rb"))
    # Initiate vectorization
    vectorization = TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=SEQ_LENGTH,
        standardize=custom_standardization,
    )
    # Call `adapt` with some dummy data 
    vectorization.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
    # Load weights from saved vectorization
    vectorization.set_weights(vectorization_weight['weights'])

    # initiate a new model
    encoder = TransformerEncoder(embed_dim=EMBED_DIM, ff_dim=FF_DIM, image_length=IMAGE_LENGTH, 
                                 num_heads=NUM_HEADS, num_layers=NUM_ENC_LAYERS)
    decoder = TransformerDecoder(embed_dim=EMBED_DIM, ff_dim=FF_DIM, 
                                 num_heads=NUM_HEADS, num_layers=NUM_DEC_LAYERS)

    # USE AN EXAMPLE IMAGE TO CALL THE MODEL
    # load and embed example image
    embedded_img = load_and_process_image(EXAMPLE_IMAGE_PATH)
    # Pass the image features to the Transformer encoder
    encoded_img = encoder(embedded_img, training=False)
    # Generate the caption using the Transformer decoder
    example_caption = "<start> this is the example caption <end>"
    tokenized_caption = vectorization([example_caption])[:, :-1]
    mask = tf.math.not_equal(tokenized_caption, 0)
    predictions = decoder(tokenized_caption, encoded_img, training=False, mask=mask)
    
    # load model weights
    encoder.load_weights(os.path.join(TRANSFORMER_MODEL_PATH,"encoder.h5"))
    decoder.load_weights(os.path.join(TRANSFORMER_MODEL_PATH,"decoder.h5"))
    caption_model = ImageCaptioningModel(encoder=encoder, decoder=decoder)

    return caption_model, vectorization

@lru_cache
def build_model_and_load_weights_prefix():
    ## load saved vectorization
    vectorization_weight = pickle.load(open(os.path.join(PREFIX_MODEL_PATH,"vectorization_weights.pkl"), "rb"))
    # Initiate vectorization
    vectorization = TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=SEQ_LENGTH,
        standardize=custom_standardization,
    )
    # Call `adapt` with some dummy data 
    vectorization.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
    # Load weights from saved vectorization
    vectorization.set_weights(vectorization_weight['weights'])

    # initiate a new model
    encoder = TransformerEncoder(embed_dim=EMBED_DIM, ff_dim=FF_DIM, image_length=IMAGE_LENGTH, 
                                 num_heads=NUM_HEADS, num_layers=NUM_ENC_LAYERS)
    decoder = TransformerDecoderPrefix(embed_dim=EMBED_DIM, ff_dim=FF_DIM, 
                                       num_heads=NUM_HEADS, num_layers=NUM_DEC_LAYERS)

    # USE AN EXAMPLE IMAGE TO CALL THE MODEL
    # load and embed example image
    embedded_img = load_and_process_image(EXAMPLE_IMAGE_PATH)   
    # Pass the image features to the Transformer encoder
    encoder_output = encoder(embedded_img, training=False)
    # Generate the caption using the Transformer decoder
    example_caption = "<start> this is the example caption <end>"
    tokenized_caption = vectorization([example_caption])[:, :-1]
    # concatenate dummy tokens for prefix and token
    seq_prefix = tf.concat([VOCAB_SIZE*tf.ones((tokenized_caption.shape[0], encoder_output.shape[1]),dtype=tf.int64), tokenized_caption],1)
    # shift to create input and target
    seq_inp = tokenized_caption[:, :-1]
    seq_true = seq_prefix[:, 1:]
    # obtain mask (0 is the pad token)
    mask = tf.math.not_equal(seq_true, 0)
    # pass thtought decoder
    seq_pred = decoder(seq_inp, encoder_output, training=False, mask=mask)

    # load model weights
    encoder.load_weights(os.path.join(PREFIX_MODEL_PATH,"encoder.h5"))
    decoder.load_weights(os.path.join(PREFIX_MODEL_PATH,"decoder.h5"))
    caption_model = ImageCaptioningModelPrefix(encoder=encoder, decoder=decoder)

    return caption_model, vectorization


def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)

    strip_chars = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
    strip_chars = strip_chars.replace("<", "")
    strip_chars = strip_chars.replace(">", "")

    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")


class TransformerEncoderBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, ff_dim, num_heads, dropout_rate = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=dropout_rate
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.ffn_1 = layers.Dense(ff_dim)
        self.ffn_2 = layers.Dense(embed_dim, activation="relu")
        self.dropout_1 = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False, mask=None):
        attention_output_1 = self.attention_1(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=None,
            training=training,
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)
        ffn_out1 = self.ffn_1(out_1)
        ffn_out1 = self.ffn_2(ffn_out1)
        ffn_out1 = self.dropout_1(ffn_out1, training=training)
        ffn_out1 = self.layernorm_2(ffn_out1 + out_1)
        
        return ffn_out1


class TransformerEncoder(tf.keras.Model):
    def __init__(self, embed_dim, ff_dim, image_length, num_heads, num_layers, dropout_rate = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.image_length = image_length
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.project = layers.Dense(embed_dim*image_length, activation="relu")
        self.layernorm_1 = layers.LayerNormalization()
        self.enc_layers = [TransformerEncoderBlock(embed_dim, ff_dim, num_heads, dropout_rate = dropout_rate) 
                           for _ in range(num_layers)]

    def call(self, inputs, training=False, mask=None):
        encoder_outputs = self.project(inputs)
        encoder_outputs = tf.reshape(encoder_outputs, [-1, self.image_length, self.embed_dim])
        encoder_outputs = self.layernorm_1(encoder_outputs)
        for i in range(self.num_layers):
            encoder_outputs = self.enc_layers[i](encoder_outputs, training=training, mask=mask)

        return encoder_outputs


class PositionalEmbedding(tf.keras.layers.Layer):
    # Use fixed positional encoding with sin and cosine transform
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embed_scale = tf.math.sqrt(tf.cast(embed_dim, tf.float32))
        self.encoded_positions = positional_encoding(self.sequence_length, self.embed_dim)

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_tokens = embedded_tokens * self.embed_scale
        return embedded_tokens + self.encoded_positions[:,:length,:]

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)

def get_angles(pos, i, embed_dim):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(embed_dim))
    return pos * angle_rates

def positional_encoding(position, embed_dim):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(embed_dim)[np.newaxis, :],
                            embed_dim)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)
        
# class PositionalEmbedding(tf.keras.layers.Layer):
#     # Learn weights for postition embedding with an Embedding layer    
#     def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
#         super().__init__(**kwargs)
#         self.token_embeddings = layers.Embedding(
#             input_dim=vocab_size, output_dim=embed_dim
#         )
#         self.position_embeddings = layers.Embedding(
#             input_dim=sequence_length, output_dim=embed_dim
#         )
#         self.sequence_length = sequence_length
#         self.vocab_size = vocab_size
#         self.embed_dim = embed_dim
#         self.embed_scale = tf.math.sqrt(tf.cast(embed_dim, tf.float32))

#     def call(self, inputs):
#         length = tf.shape(inputs)[-1]
#         positions = tf.range(start=0, limit=length, delta=1)
#         embedded_tokens = self.token_embeddings(inputs)
#         embedded_tokens = embedded_tokens * self.embed_scale
#         embedded_positions = self.position_embeddings(positions)
#         return embedded_tokens + embedded_positions

#     def compute_mask(self, inputs, mask=None):
#         return tf.math.not_equal(inputs, 0)


class TransformerDecoderBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, ff_dim, num_heads, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.self_attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=dropout_rate
        )
        self.cross_attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=dropout_rate
        )
        self.ffn_layer_1 = layers.Dense(ff_dim, activation="relu")
        self.ffn_layer_2 = layers.Dense(embed_dim)
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.dropout_1 = layers.Dropout(dropout_rate)
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, training=False, padding_mask=None, combined_mask=None):
        # masked self attention
        self_attention_output_1 = self.self_attention_1(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=combined_mask,
            training=training,
        )
        out_1 = self.layernorm_1(inputs + self_attention_output_1)

        # cross attention
        cross_attention_output_1 = self.cross_attention_1(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
            training=training,
        )
        out_2 = self.layernorm_2(out_1 + cross_attention_output_1)

        # feed-forward network
        ffn_out = self.ffn_layer_1(out_2)
        ffn_out = self.ffn_layer_2(ffn_out)
        ffn_out = self.dropout_1(ffn_out, training=training)
        ffn_out = self.layernorm_3(ffn_out + out_2, training=training)

        return ffn_out


class TransformerDecoder(tf.keras.Model):
    def __init__(self, embed_dim, ff_dim, num_heads, num_layers, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.embedding = PositionalEmbedding(
            embed_dim=EMBED_DIM, sequence_length=SEQ_LENGTH, vocab_size=VOCAB_SIZE
        )
        self.dropout = layers.Dropout(dropout_rate)
        self.dec_layers = [TransformerDecoderBlock(embed_dim, ff_dim, num_heads, dropout_rate=dropout_rate)
                           for _ in range(num_layers)]
        self.out = layers.Dense(VOCAB_SIZE, activation="softmax")
        self.supports_masking = True
        
    def call(self, inputs, encoder_outputs, training=False, mask=None):
        inputs = self.embedding(inputs)
        causal_mask = self.get_causal_attention_mask(inputs)

        if mask is not None:
            padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)
            combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
            combined_mask = tf.minimum(combined_mask, causal_mask)

        decoder_outputs = self.dropout(inputs, training=training)
        for i in range(self.num_layers):
            decoder_outputs = self.dec_layers[i](decoder_outputs, encoder_outputs, training=training, 
                                                 padding_mask=padding_mask, combined_mask=combined_mask)
        
        decoder_outputs = self.out(decoder_outputs)
        return decoder_outputs

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)


class TransformerDecoderBlockNoCross(tf.keras.layers.Layer):
    def __init__(self, embed_dim, ff_dim, num_heads, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.self_attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=dropout_rate
        )
        self.ffn_layer_1 = layers.Dense(ff_dim, activation="relu")
        self.ffn_layer_2 = layers.Dense(embed_dim)
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.dropout_1 = layers.Dropout(dropout_rate)
        self.supports_masking = True

    def call(self, inputs, training=False, padding_mask=None, combined_mask=None):
        # masked self attention
        self_attention_output_1 = self.self_attention_1(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=combined_mask,
            training=training,
        )
        out_1 = self.layernorm_1(inputs + self_attention_output_1)

        # feed-forward network
        ffn_out = self.ffn_layer_1(out_1)
        ffn_out = self.ffn_layer_2(ffn_out)
        ffn_out = self.dropout_1(ffn_out, training=training)
        ffn_out = self.layernorm_2(ffn_out + out_1, training=training)

        return ffn_out

class TransformerDecoderPrefix(tf.keras.Model):
    def __init__(self, embed_dim, ff_dim, num_heads, num_layers, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.embedding = PositionalEmbedding(
            embed_dim=EMBED_DIM, sequence_length=SEQ_LENGTH, vocab_size=VOCAB_SIZE
        )
        self.layernorm = layers.LayerNormalization()
        self.dropout = layers.Dropout(dropout_rate)
        self.dec_layers = [TransformerDecoderBlockNoCross(embed_dim, ff_dim, num_heads, dropout_rate=dropout_rate)
                           for _ in range(num_layers)]
        self.out = layers.Dense(VOCAB_SIZE+1, activation="softmax") # +1, last token is the image prefix token
        self.supports_masking = True
        
    def call(self, inputs, encoder_outputs, training=False, mask=None):
        inputs = self.embedding(inputs)
        inputs = self.layernorm(inputs)
        prefix_size = tf.shape(encoder_outputs)[1]
        prefix_inputs = tf.concat([encoder_outputs,inputs], axis=1) # add prefix to text embedding
        causal_mask = self.get_causal_attention_mask(prefix_inputs, prefix_size=prefix_size)

        if mask is not None:
            padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)
            combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
            combined_mask = tf.minimum(combined_mask, causal_mask)

        decoder_outputs = self.dropout(prefix_inputs, training=training)
        for i in range(self.num_layers):
            decoder_outputs = self.dec_layers[i](decoder_outputs, training=training, 
                                                 padding_mask=padding_mask, combined_mask=combined_mask)
        
        decoder_outputs = self.out(decoder_outputs)
        return decoder_outputs

    def get_causal_attention_mask(self, inputs, prefix_size=0):
        # Causal self-attention mask for prefix LM:
        # See T5 paper: https://arxiv.org/pdf/1910.10683.pdf
        # Or UNILM paper: https://arxiv.org/pdf/1905.03197.pdf

        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        
        # create full attension for prefix
        if prefix_size > 0:
            mask = tf.concat([tf.ones((sequence_length,prefix_size),dtype=tf.int32), mask[:, prefix_size:]],axis=1)

        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)


class ImageCaptioningModel(tf.keras.Model):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.acc_tracker = keras.metrics.Mean(name="accuracy")

    # def call(self, inputs, training=False):
    #     tokenized_caption, embeded_img, mask = inputs
    #     encoder_outputs = self.encoder(embeded_img, training=training, mask=mask)
    #     decoder_outputs = self.decoder(tokenized_caption, encoder_outputs, training=training, mask=mask)
    #     return decoder_outputs

    def calculate_loss(self, y_true, y_pred, mask):
        loss = self.loss(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    def calculate_accuracy(self, y_true, y_pred, mask):
        accuracy = tf.equal(tf.cast(y_true, dtype=tf.int32), tf.cast(tf.argmax(y_pred, axis=2),dtype=tf.int32))
        accuracy = tf.math.logical_and(mask, accuracy)
        accuracy = tf.cast(accuracy, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)

    def _compute_caption_loss_and_acc(self, img_embed, batch_seq, training=True):
        encoder_out = self.encoder(img_embed, training=training)
        batch_seq_inp = batch_seq[:, :-1]
        batch_seq_true = batch_seq[:, 1:]
        mask = tf.math.not_equal(batch_seq_true, 0)
        batch_seq_pred = self.decoder(
            batch_seq_inp, encoder_out, training=training, mask=mask
        )
        loss = self.calculate_loss(batch_seq_true, batch_seq_pred, mask)
        acc = self.calculate_accuracy(batch_seq_true, batch_seq_pred, mask)
        return loss, acc

    def train_step(self, batch_data):
        batch_img_embed, batch_seq = batch_data
        batch_loss = 0
        batch_acc = 0

        # pass each the captions to the decoder along with the encoder outputs 
        # and compute the loss as well as accuracy for each caption
        with tf.GradientTape() as tape:
            loss, acc = self._compute_caption_loss_and_acc(
                batch_img_embed, batch_seq, training=True
            )

            # update loss and accuracy
            batch_loss += loss
            batch_acc += acc

        # get the list of all the trainable weights
        train_vars = (
            self.encoder.trainable_variables + self.decoder.trainable_variables
        )

        # compute gradients
        grads = tape.gradient(loss, train_vars)

        # update trainable weights
        self.optimizer.apply_gradients(zip(grads, train_vars))

        # update the trackers
        self.loss_tracker.update_state(batch_loss)
        self.acc_tracker.update_state(batch_acc)

        # return loss and accuracy
        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}

    def test_step(self, batch_data):
        batch_img, batch_seq = batch_data
        batch_loss = 0
        batch_acc = 0

        # get image embeddings
        # img_embed = self.cnn_model(batch_img)
        img_embed = batch_img

        # pass each the captions to the decoder along with the encoder outputs 
        # and compute the loss as well as accuracy for each caption
        loss, acc = self._compute_caption_loss_and_acc(
            img_embed, batch_seq, training=False
        )

        # update batch loss and batch accuracy
        batch_loss += loss
        batch_acc += acc

        # update the trackers
        self.loss_tracker.update_state(batch_loss)
        self.acc_tracker.update_state(batch_acc)

        # return loss and accuracy
        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker, self.acc_tracker]


class ImageCaptioningModelPrefix(tf.keras.Model):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.acc_tracker = keras.metrics.Mean(name="accuracy")

    def calculate_loss(self, y_true, y_pred, mask):
        loss = self.loss(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    def calculate_accuracy(self, y_true, y_pred, mask):
        accuracy = tf.equal(tf.cast(y_true, dtype=tf.int32), tf.cast(tf.argmax(y_pred, axis=-1),dtype=tf.int32))
        accuracy = tf.math.logical_and(mask, accuracy)
        accuracy = tf.cast(accuracy, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)

    def _compute_caption_loss_and_acc(self, img_embed, batch_seq, training=True):
        # encoder image feature into prefix
        encoder_output = self.encoder(img_embed, training=training)
        prefix_length = self.encoder.image_length

        # concatenate dummy tokens for prefix (value = VOCAB_SIZE, i.e. 15000) and token
        batch_seq_prefix = tf.concat([VOCAB_SIZE*tf.ones((tf.shape(batch_seq)[0], prefix_length),dtype=tf.int64), batch_seq],1)

        # shift to create input and target
        batch_seq_inp = batch_seq[:, :-1]
        batch_seq_true = batch_seq_prefix[:, 1:]

        # obtain mask (0 is the pad token)
        mask = tf.math.not_equal(batch_seq_true, 0)

        # pass through decoder
        batch_seq_pred = self.decoder(
            batch_seq_inp, encoder_output, training=training, mask=mask
        )

        # exclude prefix when computing loss
        loss = self.calculate_loss(batch_seq_true[:,prefix_length-1:], batch_seq_pred[:,prefix_length-1:,:], mask[:,prefix_length-1:])
        acc = self.calculate_accuracy(batch_seq_true[:,prefix_length-1:], batch_seq_pred[:,prefix_length-1:,:], mask[:,prefix_length-1:])
        return loss, acc

    def train_step(self, batch_data):
        batch_img_embed, batch_seq = batch_data
        batch_loss = 0
        batch_acc = 0

        # pass each the captions to the decoder along with the encoder outputs 
        # and compute the loss as well as accuracy for each caption
        with tf.GradientTape() as tape:
            loss, acc = self._compute_caption_loss_and_acc(
                batch_img_embed, batch_seq, training=True
            )

            # update loss and accuracy
            batch_loss += loss
            batch_acc += acc

        # get the list of all the trainable weights
        train_vars = (
            self.encoder.trainable_variables + self.decoder.trainable_variables
        )

        # compute gradients
        grads = tape.gradient(loss, train_vars)

        # update trainable weights
        self.optimizer.apply_gradients(zip(grads, train_vars))

        # update the trackers
        self.loss_tracker.update_state(batch_loss)
        self.acc_tracker.update_state(batch_acc)

        # return loss and accuracy
        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}

    def test_step(self, batch_data):
        batch_img_embed, batch_seq = batch_data
        batch_loss = 0
        batch_acc = 0


        # pass each the captions to the decoder along with the encoder outputs 
        # and compute the loss as well as accuracy for each caption
        loss, acc = self._compute_caption_loss_and_acc(
            batch_img_embed, batch_seq, training=False
        )

        # update batch loss and batch accuracy
        batch_loss += loss
        batch_acc += acc

        # update the trackers
        self.loss_tracker.update_state(batch_loss)
        self.acc_tracker.update_state(batch_acc)

        # return loss and accuracy
        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker, self.acc_tracker]