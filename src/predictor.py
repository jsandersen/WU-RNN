import numpy as np
from tensorflow.keras import backend as K
from tqdm import tqdm

class Predictor:
    f = None
    n_iter = None
    
    def __init__(self, model, n_iter):
        self.n_iter = n_iter
        
        model_input = model.layers[0].input
        model_output = model.output
        dense_layer_output = model.layers[-3].output
        activation_layer_output = model.layers[-2].output
        
        self.f = K.function(
            [model_input, K.learning_phase()],
            [dense_layer_output, model_output, activation_layer_output]
        )
    
    def predict(self, x):
        predictions = []
        for i in range(0, self.n_iter):
            predictions.append(self.f([x, True])[1])
        p_mean = np.asarray(predictions).mean(axis=0)
        
        return p_mean
    
    # return prob-word, epi-word, aleo-word, prob-input
    def predict_with_uncertainty(self, x):
        predictions = []
        predictions2 = []
        predictions3 = []
        for i in range(0, self.n_iter):
            out = self.f([x, True])
            predictions.append(out[2])
            predictions2.append(out[1])
            predictions3.append(out[0])
        p_hat = np.asarray(predictions)
        
        p, epistemic, aleatoric = self._estimate_uncertainty(p_hat)
        return p, epistemic, aleatoric, np.array(self._relevance(predictions3))
    
    def _relevance(self, p):
        relevance_list = []
        p = np.array(p)
        for i in range(p.shape[1]):
            p_reshaped = p[:, i, :, :]
            relevance_list.append(p_reshaped)
        return relevance_list

    # @see https://www.sciencedirect.com/science/article/abs/pii/S016794731930163X
    def _estimate_uncertainty(self, p_hat):
        p_hat_list = []
        p_a = []
        p_e = []
        
        for i in range(p_hat.shape[1]):
            p_hat_reshaped = p_hat[:, i, :, :]
            
            epistemic = np.mean(p_hat_reshaped**2, axis=0) - np.mean(p_hat_reshaped, axis=0)**2
            aleatoric = np.mean(p_hat_reshaped*(1-p_hat_reshaped), axis=0)

            p_hat_list.append(p_hat_reshaped)
            p_a.append(aleatoric)
            p_e.append(epistemic)
        return np.array(p_hat_list), np.array(p_e), np.array(p_a)
