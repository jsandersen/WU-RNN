from tensorflow.keras.layers import Dropout, Lambda, Bidirectional, GlobalMaxPooling1D ,GlobalAveragePooling1D, LSTM, Activation, Embedding, Input, Dense
from tensorflow.keras.models import Model
import yaml

# load config
with open('config.yaml', 'r') as f:
    conf = yaml.load(f)
MAX_SEQUENCE_LENGTH = conf["EMBEDDING"]["MAX_SEQUENCE_LENGTH"]
DROPOUT = conf["MODEL"]["DROPOUT"]
RECURRENT_DROPOUT = conf["MODEL"]["RECURRENT_DROPOUT"]
FILTER_SIZE = conf["MODEL"]["FILTER_SIZE"]

def get_model(n_classes, embedding_layer):
    main_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    x = embedding_layer(main_input)
    x = Dropout(DROPOUT)(x)
    x = LSTM(FILTER_SIZE, dropout=RECURRENT_DROPOUT, recurrent_dropout=RECURRENT_DROPOUT, name="lstm", return_sequences=True)(x)
    x = Dense(n_classes)(x)
    x = Activation('softmax')(x)
    x = Lambda(lambda x: x[:,-1])(x)
    model = Model(inputs=[main_input], outputs=[x])
    
    return model