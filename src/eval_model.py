from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import np_utils
from keras import backend as K
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def evaluate_model(model, holdout_generator, batch_size):

    '''evaluate model on holdout set to get loss and accuracy'''

    metrics = model.evaluate_generator(holdout_generator,
                                        steps=74//batch_size,
                                        use_multiprocessing=True,
                                        verbose=1)

    print(f"holdout loss: {metrics[0]} accuracy: {metrics[1]}")
    return metrics[0], metrics[1]

def predictions_to_csv(holdout_generator, model, batch_size, name_csv):

    '''Save predictions to csv. Includes file path, predictions, and predicted values
    Note: the holdout generator is reset in the event that the evaluation is executed
    before the predictions'''

    holdout_generator.reset()
    predictions = model.predict_generator(holdout_generator, steps = 74 // batch_size)
    pred_vals = predictions
    vec = np.vectorize(lambda x: 1 if x>0. else 0)
    predicted_class_indices=vec(predictions)
    labels = (holdout_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices.ravel()]
    filenames=holdout_generator.filenames[:len(predictions)]
    results=pd.DataFrame({"Filename":filenames,
                          "Predictions":predictions,
                          "Values":pred_vals.ravel()})
    results.to_csv("/Users/christopherlawton/galvanize/module_2/capstone_2/{}.csv".format(name_csv),index=False)

if __name__=='__main__':

    batch_size = 74
    scale = 65535

    model_name = 'final_run.h5'
    model = load_model('/Users/christopherlawton/galvanize/module_2/capstone_2/save_model/{}'.format(model_name))

    test_datagen = ImageDataGenerator(
                    rescale=1./scale)

    holdout_generator = test_datagen.flow_from_directory(
                        '/Users/christopherlawton/final_test_train_hold/hold',
                        color_mode='grayscale',
                        target_size=(150,150),
                        batch_size=batch_size,
                        class_mode='binary',
                        shuffle=True)

    model_loss, model_accuracy = evaluate_model(model, holdout_generator, batch_size)

    predictions_to_csv(holdout_generator, model, batch_size, 'final_model_full_mam')
