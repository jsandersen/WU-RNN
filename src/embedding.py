import tensorflow as tf
import numpy as np
import yaml
from src.preprocessing import clean_doc

# load conifig
with open('config.yaml', 'r') as f:
    conf = yaml.load(f)
MAX_NUM_WORDS = conf["EMBEDDING"]["MAX_NUM_WORDS"]
MAX_SEQUENCE_LENGTH = conf["EMBEDDING"]["MAX_SEQUENCE_LENGTH"]

def get_data_tensor(texts, training_size):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_NUM_WORDS, oov_token=1)
    tokenizer.fit_on_texts(texts[:training_size])
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    return tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH), word_index

def get_embeddings_index(model):
    embeddings_index = model.wv.vocab
    for word, vocab in embeddings_index.items():
        embeddings_index[word] = model.wv.vectors[vocab.index]
    return embeddings_index, model.vector_size

def get_embedding_layer(word_index, embedding_index, embedding_dim, static=False):
    num_words = min(MAX_NUM_WORDS, len(word_index))
    embedding_matrix = np.zeros((num_words+1, embedding_dim))
    for word, i in word_index.items():
        if i > MAX_NUM_WORDS:
            continue
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return tf.keras.layers.Embedding(
        embedding_matrix.shape[0],
        embedding_matrix.shape[1],
        weights=[embedding_matrix],
        input_length=MAX_SEQUENCE_LENGTH,
        mask_zero=True,
        trainable=static)

class TextIdCoverter:

    def __init__(self, word_index):
        self.word_index = word_index
        self.id_index = {value:key for key,value in word_index.items()}
        
    def id2text(self, ids):
        ids = ids.reshape((MAX_SEQUENCE_LENGTH))
        return ' '.join('[?]' if id == 1 else self.id_index[id] for id in ids if id != 0)

    def text2id(self, text):
        text = clean_doc(text)
        text = [self.word_index.get(id) or 1 for id in text.split(' ')]
        text = tf.keras.preprocessing.sequence.pad_sequences([text], maxlen=MAX_SEQUENCE_LENGTH)[0]
        return text