import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.keras.layers import Dense, Dropout, Flatten #.core
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor # or Classifier
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dropout
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils
from tensorflow.keras.layers import Dense, Conv1D, Flatten
import collections
import operator
import joblib


def unpack(model, training_config, weights): ##https://github.com/tensorflow/tensorflow/issues/34697 #fixes an error that the early stopping callback throws up in the nested cv #something about the parralele fitting step needing everything to be pickle-able and the callback isnt 
    restored_model = deserialize(model)
    if training_config is not None:
        restored_model.compile(
            **saving_utils.compile_args_from_training_config(
                training_config
            )
        )
    restored_model.set_weights(weights)
    return restored_model

# Hotfix function
def make_keras_picklable():

    def __reduce__(self):
        model_metadata = saving_utils.model_metadata(self)
        training_config = model_metadata.get("training_config", None)
        model = serialize(self)
        weights = self.get_weights()
        return (unpack, (model, training_config, weights))

    cls = Model
    cls.__reduce__ = __reduce__


def build_nn(HP_OPTIMIZER, HP_NUM_HIDDEN_LAYERS, units, activation, learning_rate, HP_L1_REG, HP_L2_REG, rate, kernel_initializer, network_shape):
        make_keras_picklable()
        opt = HP_OPTIMIZER
        chosen_opt = getattr(tf.keras.optimizers,opt)
        reg = tf.keras.regularizers.l1_l2(l1=HP_L1_REG, l2=HP_L2_REG)
        long_funnel_count = 0 #keep widest shape for two layers
        model = Sequential()
        input_shape = (set_size,) if snps == 'shuf' else (set_size-1,)
        model.add(Dense(units=units, activation=activation, kernel_regularizer=reg, input_shape=input_shape))
        if rate != 0:
                model.add(Dropout(rate=rate))
        for i in range(HP_NUM_HIDDEN_LAYERS-1):
                if network_shape == 'funnel':
                        units = int(units*0.666)
                elif network_shape == 'long_funnel':
                        if long_funnel_count >= 1: #two wide layers (inclduing previous first layer)
                                units=int(units*0.666)
                        long_funnel_count += 1
                model.add(Dense(units=units, activation=activation, kernel_regularizer=reg, kernel_initializer=kernel_initializer))
                if rate != 0:
                        model.add(Dropout(rate=rate))
        if binary == 'True' :
                model.add(Dense(1, activation='sigmoid'))
                model.compile(loss='binary_crossentropy',metrics=['accuracy', AUC(name='auc')],optimizer=chosen_opt(learning_rate=learning_rate))
        else:
                model.add(Dense(1, activation='linear'))
                model.compile(loss='mean_absolute_error',metrics=['accuracy', 'mae', coeff_determination],optimizer=chosen_opt(learning_rate=learning_rate))
        #new_layer_weights = np.random.rand(x_train.shape[1]-1,units) #(num_inputs,num_units)
        #for i in range(0,x_train.shape[1]-1):
        #       new_Layer_weights[i,:] = beta_weights[i]
        #new_weight_list = []
        #new_weight_list.append(new_layer_weights)
        #new_weight_list.append(np.zeros(num_units)) # biases
        #model.layers[0].set_weights(new_weight_list)
        print(model.summary())
        return model

