import tensorflow as tf
from tensorflow import keras as tfk
import tensorflow.python.keras.backend as K

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2

from tqdm import tqdm
from scipy.special import softmax
from matplotlib import pyplot as plt
from abc import ABC, abstractmethod
import numpy as np

class NeuralNetwork(ABC):
    
    def __init__(self, input_shape, n_classes, mcd, T, opt, loss, softmax):
        model = self._create_model(input_shape, n_classes)
        model.compile(opt, loss, metrics=["accuracy"])
        self.mcd = mcd
        self.T = T
        self.softmax = softmax
        self.model = model
        self.input_shape = input_shape
        self.n_classes = n_classes
    
    @abstractmethod
    def _create_model(self, input_shape, n_classes, mcd):
        pass
    
    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = tfk.models.load_model(path)

    def _print_history(self, history):
        #  "Accuracy"
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        # "Loss"
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
    
    def fit(self, X_train, y_train, X_test, y_test, batch_size=-1, epochs=-1, callbacks = None, validation_split = 0, save=None, load=None):
        
        history = self.model.fit(
                X_train, 
                to_categorical(y_train), 
                batch_size=batch_size, 
                epochs=epochs, 
                shuffle=True, 
                validation_split=0.1,    
                callbacks=callbacks,
                verbose=0
        )
        
        self._print_history(history)
        
    def predict_proba(self, X):
        res = []
        
        if self.mcd:
            preds = []
            
            model2 = self._create_model(self.input_shape, self.n_classes, mcd=True)
            model2.layers[1].set_weights(self.model.layers[1].get_weights())
            model2.layers[3].set_weights(self.model.layers[3].get_weights())
            model2.layers[5].set_weights(self.model.layers[5].get_weights())
            
            preds = []
            for i in tqdm(range(self.T)):
                preds.append(model2.predict(X))
            preds = np.array(preds)
            res = preds.mean(axis=0)
        else:
            res =  self.model.predict(X)
        
        if not self.softmax:
            return softmax(res, axis=1)
        else:
            return res
        
class MLP(NeuralNetwork):
    
    def __init__(self, input_shape, n_classes, mcd=False, T=1, opt=tfk.optimizers.Adam(learning_rate=0.0001), loss="categorical_crossentropy"): # 0.0001
        super().__init__(input_shape, n_classes, mcd, T, opt, loss, softmax=True)
    
    def _create_model(self, input_shape, n_classes, mcd=False):
        input_layer = tfk.Input(shape=(input_shape,), dtype='float32')
    
        print(mcd)
        
        if mcd: 
            x = tfk.layers.Dense(500, activation='relu', kernel_regularizer=l2(0.0001))(input_layer)
            x = tfk.layers.Dropout(0.5)(x, training=True)
            x = tfk.layers.Dense(500, activation='relu', kernel_regularizer=l2(0.0001))(x)
            x = tfk.layers.Dropout(0.5)(x, training=True)
        else:
            x = tfk.layers.Dense(500, activation='relu', kernel_regularizer=l2(0.0001))(input_layer)
            x = tfk.layers.Dropout(0.5)(x)
            x = tfk.layers.Dense(500, activation='relu', kernel_regularizer=l2(0.0001))(x)
            x = tfk.layers.Dropout(0.5)(x)                                       
                                        
        output_layer = tfk.layers.Dense(n_classes, activation='softmax', kernel_regularizer=l2(0.0001))(x)
        model = tfk.Model(input_layer, output_layer)
        
        return model