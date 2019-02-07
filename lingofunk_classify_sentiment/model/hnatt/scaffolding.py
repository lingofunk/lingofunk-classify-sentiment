import datetime
import os
import pickle

import numpy as np

import keras
from keras import backend as K
from keras import initializers, regularizers
from keras.callbacks import *
from keras.engine.topology import Layer
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import CustomObjectScope
from lingofunk_classify_sentiment.config import config, fetch
from lingofunk_classify_sentiment.data.load import load_glove_embedding
from lingofunk_classify_sentiment.model.hnatt.preprocess import normalize

# Uncomment below for debugging
# from tensorflow.python import debug as tf_debug
# sess = K.get_session()
# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
# K.set_session(sess)


def dot_with_kernel(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == "tensorflow":
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


class Attention(Layer):
    def __init__(
        self,
        has_bias=True,
        embedding_regularizer="l2",
        context_regularizer="l2",
        bias_regularizer="l2",
        regularizer=None,
        **kwargs
    ):
        super(Attention, self).__init__(**kwargs)
        self.initializer = initializers.get("glorot_uniform")
        self.embedding_regularizer = regularizers.get(embedding_regularizer)
        self.context_regularizer = regularizers.get(context_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.regularizer = regularizer
        self.supports_masking = True
        self.has_bias = has_bias
        self.bias = None

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.context = self.add_weight(
            name="context",
            shape=(input_shape[-1],),
            initializer=self.initializer,
            regularizer=self.context_regularizer,
            trainable=True,
        )
        if self.has_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(input_shape[-1],),
                initializer=initializers.get("zeros"),
                regularizer=self.bias_regularizer,
                trainable=True,
            )
        self.embedding = self.add_weight(
            (input_shape[-1], input_shape[-1]),
            initializer=self.initializer,
            name="embedding",
            regularizer=self.embedding_regularizer,
        )
        super(Attention, self).build(input_shape)

    def call(self, x, mask=None):
        uit = dot_with_kernel(x, self.embedding)

        if self.has_bias:
            uit += self.bias

        uit = K.tanh(uit)
        ait = dot_with_kernel(uit, self.context)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        print(input_shape)
        return (input_shape[0], input_shape[-1])


class HNATT:
    def __init__(self):
        self.MAX_SENTENCE_LENGTH = 0
        self.MAX_SENTENCE_COUNT = 0
        self.VOCABULARY_SIZE = 0
        self.word_embedding = None
        self.model = None
        self.word_attention_model = None
        self.tokenizer = None
        self.class_count = 2

    def _generate_embedding(self, path, dim):
        return load_glove_embedding(path, dim, self.tokenizer.word_index)

    def _build_model(self, n_classes=2, embedding_dim=300, embeddings_path=False):
        l2_reg = regularizers.l2(1e-8)
        # embedding_weights = np.random.normal(0, 1, (len(self.tokenizer.word_index) + 1, embedding_dim))
        # embedding_weights = np.zeros((len(self.tokenizer.word_index) + 1, embedding_dim))
        embedding_weights = np.random.normal(
            0, 1, (len(self.tokenizer.word_index) + 1, embedding_dim)
        )
        if embeddings_path:
            embedding_weights = self._generate_embedding(embeddings_path, embedding_dim)

        # Generate word-attention-weighted sentence scores
        sentence_in = Input(shape=(self.MAX_SENTENCE_LENGTH,), dtype="int32")
        embedded_word_seq = Embedding(
            self.VOCABULARY_SIZE,
            embedding_dim,
            weights=[embedding_weights],
            input_length=self.MAX_SENTENCE_LENGTH,
            trainable=True,
            mask_zero=True,
            name="word_embeddings",
        )(sentence_in)
        word_encoder = Bidirectional(
            GRU(50, return_sequences=True, kernel_regularizer=l2_reg)
        )(embedded_word_seq)
        dense_transform_w = Dense(
            100, activation="relu", name="dense_transform_w", kernel_regularizer=l2_reg
        )(word_encoder)
        attention_weighted_sentence = Model(
            sentence_in,
            Attention(name="word_attention", regularizer=l2_reg)(dense_transform_w),
        )
        self.word_attention_model = attention_weighted_sentence
        attention_weighted_sentence.summary()

        # Generate sentence-attention-weighted document scores
        texts_in = Input(
            shape=(self.MAX_SENTENCE_COUNT, self.MAX_SENTENCE_LENGTH), dtype="int32"
        )
        attention_weighted_sentences = TimeDistributed(attention_weighted_sentence)(
            texts_in
        )
        sentence_encoder = Bidirectional(
            GRU(50, return_sequences=True, kernel_regularizer=l2_reg)
        )(attention_weighted_sentences)
        dense_transform_s = Dense(
            100, activation="relu", name="dense_transform_s", kernel_regularizer=l2_reg
        )(sentence_encoder)
        attention_weighted_text = Attention(
            name="sentence_attention", regularizer=l2_reg
        )(dense_transform_s)
        prediction = Dense(n_classes, activation="softmax")(attention_weighted_text)
        model = Model(texts_in, prediction)
        model.summary()

        model.compile(  # optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0),
            # optimizer=SGD(lr=0.01, decay=1e-6, nesterov=True),
            optimizer=Adam(lr=0.001),
            loss="categorical_crossentropy",
            metrics=["acc"],
        )

        return model

    def load_weights(self, weights_path=fetch(config["models"]["hnatt"]["weights"])):
        with CustomObjectScope({"Attention": Attention}):
            self.model = load_model(weights_path)
            self.word_attention_model = self.model.get_layer("time_distributed_1").layer
            tokenizer_path = fetch(config["models"]["hnatt"]["tokenizer"])
            tokenizer_state = pickle.load(open(tokenizer_path, "rb"))
            self.tokenizer = tokenizer_state["tokenizer"]
            self.MAX_SENTENCE_COUNT = tokenizer_state["maxSentenceCount"]
            self.MAX_SENTENCE_LENGTH = tokenizer_state["maxSentenceLength"]
            self.VOCABULARY_SIZE = tokenizer_state["vocabularySize"]
            self._create_reverse_word_index()

    def _fit_on_texts(self, texts):
        self.tokenizer = Tokenizer(filters='"()*,-/;[\]^_`{|}~', oov_token="UNK")
        all_sentences = []
        max_sentence_count = 0
        max_sentence_length = 0
        for text in texts:
            sentence_count = len(text)
            if sentence_count > max_sentence_count:
                max_sentence_count = sentence_count
            for sentence in text:
                sentence_length = len(sentence)
                if sentence_length > max_sentence_length:
                    max_sentence_length = sentence_length
                all_sentences.append(sentence)

        self.MAX_SENTENCE_COUNT = min(max_sentence_count, 20)
        self.MAX_SENTENCE_LENGTH = min(max_sentence_length, 50)
        self.tokenizer.fit_on_texts(all_sentences)
        self.VOCABULARY_SIZE = len(self.tokenizer.word_index) + 1
        self._create_reverse_word_index()

    def _create_reverse_word_index(self):
        self.reverse_word_index = {
            value: key for key, value in self.tokenizer.word_index.items()
        }

    def _encode_texts(self, texts):
        encoded_texts = np.zeros(
            (len(texts), self.MAX_SENTENCE_COUNT, self.MAX_SENTENCE_LENGTH)
        )
        for i, text in enumerate(texts):
            encoded_text = np.array(
                pad_sequences(
                    self.tokenizer.texts_to_sequences(text),
                    maxlen=self.MAX_SENTENCE_LENGTH,
                )
            )[: self.MAX_SENTENCE_COUNT]
            encoded_texts[i][-len(encoded_text) :] = encoded_text
        return encoded_texts

    def _save_tokenizer_on_epoch_end(self, path, epoch):
        if epoch == 0:
            tokenizer_state = {
                "tokenizer": self.tokenizer,
                "maxSentenceCount": self.MAX_SENTENCE_COUNT,
                "maxSentenceLength": self.MAX_SENTENCE_LENGTH,
                "vocabularySize": self.VOCABULARY_SIZE,
            }
            pickle.dump(tokenizer_state, open(path, "wb"))

    def train(
        self,
        train_x,
        train_y,
        batch_size=32,
        epochs=1,
        embedding_dim=300,
        embeddings_path=False,
    ):
        # fit tokenizer
        self._fit_on_texts(train_x)
        self.model = self._build_model(
            n_classes=train_y.shape[-1],
            embedding_dim=embedding_dim,
            embeddings_path=embeddings_path,
        )
        encoded_train_x = self._encode_texts(train_x)
        callbacks = [
            # EarlyStopping(
            # 	monitor='acc',
            # 	patience=2,
            # ),
            ReduceLROnPlateau(),
            # keras.callbacks.TensorBoard(
            # 	log_dir="logs/final/{}".format(datetime.datetime.now()),
            # 	histogram_freq=1,
            # 	write_graph=True,
            # 	write_images=True
            # )
            LambdaCallback(
                on_epoch_end=lambda epoch, logs: self._save_tokenizer_on_epoch_end(
                    fetch(config["models"]["hnatt"]["tokenizer"]), epoch
                )
            ),
        ]

        callbacks.append(
            ModelCheckpoint(
                filepath=fetch(config["models"]["hnatt"]["weights"]),
                monitor="val_acc",
                save_best_only=True,
                save_weights_only=False,
            )
        )
        self.model.fit(
            x=encoded_train_x,
            y=train_y,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            callbacks=callbacks,
            validation_split=0.1,
            shuffle=True,
        )

    def _encode_input(self, x, log=False):
        x = np.array(x)
        if not x.shape:
            x = np.expand_dims(x, 0)
        texts = np.array([normalize(text) for text in x])
        return self._encode_texts(texts)

    def predict(self, x):
        encoded_x = self._encode_texts(x)
        return self.model.predict(encoded_x)

    def classify(self, normalised_text):
        preds = self.predict([normalised_text])[0]
        class_type = np.argmax(preds).astype(float)
        return class_type

    def prob_classify(self, normalised_text):
        preds = self.predict([normalised_text])[0]
        return np.max(preds)

    def activation_maps(self, text, websafe=False):
        normalized_text = normalize(text)
        encoded_text = self._encode_input(text)[0]

        # get word activations
        hidden_word_encoding_out = Model(
            inputs=self.word_attention_model.input,
            outputs=self.word_attention_model.get_layer("dense_transform_w").output,
        )
        hidden_word_encodings = hidden_word_encoding_out.predict(encoded_text)
        word_context = self.word_attention_model.get_layer(
            "word_attention"
        ).get_weights()[0]
        u_wattention = encoded_text * np.exp(
            np.squeeze(np.dot(hidden_word_encodings, word_context))
        )
        if websafe:
            u_wattention = u_wattention.astype(float)

        # generate word, activation pairs
        nopad_encoded_text = encoded_text[-len(normalized_text) :]
        nopad_encoded_text = [
            list(filter(lambda x: x > 0, sentence)) for sentence in nopad_encoded_text
        ]
        reconstructed_texts = [
            [self.reverse_word_index[int(i)] for i in sentence]
            for sentence in nopad_encoded_text
        ]
        nopad_wattention = u_wattention[-len(normalized_text) :]
        nopad_wattention = nopad_wattention / np.expand_dims(
            np.sum(nopad_wattention, -1), -1
        )
        nopad_wattention = np.array(
            [
                attention_seq[-len(sentence) :]
                for attention_seq, sentence in zip(nopad_wattention, nopad_encoded_text)
            ]
        )
        word_activation_maps = []
        for i, text in enumerate(reconstructed_texts):
            word_activation_maps.append(list(zip(text, nopad_wattention[i])))

        # get sentence activations
        hidden_sentence_encoding_out = Model(
            inputs=self.model.input,
            outputs=self.model.get_layer("dense_transform_s").output,
        )
        hidden_sentence_encodings = np.squeeze(
            hidden_sentence_encoding_out.predict(np.expand_dims(encoded_text, 0)), 0
        )
        sentence_context = self.model.get_layer("sentence_attention").get_weights()[0]
        u_sattention = np.exp(np.dot(hidden_sentence_encodings, sentence_context))
        if websafe:
            u_sattention = u_sattention.astype(float)
        nopad_sattention = u_sattention[-len(normalized_text) :]

        nopad_sattention = nopad_sattention / np.expand_dims(
            np.sum(nopad_sattention, -1), -1
        )

        activation_map = list(zip(word_activation_maps, nopad_sattention))

        return activation_map
