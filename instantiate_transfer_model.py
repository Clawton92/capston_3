from keras.applications import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout
from keras.models import Model

def add_model_head(base_model, n_categories, activation_function):
    """
    Takes a base model and adds a pooling and a softmax output based on the number of categories

    Args:
        base_model (keras Sequential model): model to attach head to
        activation_function (string): the activation of the last dense layer (softmax or sigmoid)

    Returns:
        keras Sequential model: model with new head
        """
    if activation_func = 'sigmoid':
        n_categories = 1
    else:
        n_categories = n_categories

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(n_categories, activation=activation_func)(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def create_transfer_model(input_size, n_categories, activation_function, weights = 'imagenet', model=InceptionV3):
    """
    Creates model without top and attaches new head to it
    Args:
        input_size (tuple(int, int, int)): 3-dimensional size of input to model.

        ***Note that transfer models in keras will only accept an input shape with three channels.
        If grayscale, you must stack the same image 3 times as input.

        n_categories (int): number of categories (labels)
        activation_function (string): activation function for the last dense layer (classification)
        weights (str or arg): weights to use for model
        model (keras Sequential model): model to use for transfer
    Returns:
        keras Sequential model: model with new head
        """
    base_model = model(weights=weights,
                      include_top=False,
                      input_shape=input_size)
    model = add_model_head(base_model, activation_function)
    return model
