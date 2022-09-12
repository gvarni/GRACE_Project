# File created By: Lucien Maman
# Last updated: 12/08/2022
# ---------------------------------------------------------------------------
"""
The TBD architectures (i.e., TBD-S and TBD-T).

Specificities:
- Requires having features computed from both individuals and the group as a whole
- Requires having a pre-trained model dedicated to the prediction of a specific dimension

Designed by: Lucien Maman

It was presented at @ICMI2021.
Please cite @ICMI2021 to refer to the TBDs.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import tensorflow as tf

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

# ---------------------------------------------------------------------------
# Function returning the TBD architecture
# ---------------------------------------------------------------------------

def create_TBD():

    # Define the parameters of the model. Default numbers are the ones used in the paper(s)
    nb_task = 5              # Here, 5 tasks are predicted, following the ones presented in @GAME-ON
    timesteps = 6 * nb_task  # number of timesteps (here 6 segments of 20s) x the number of task (here set to 5)
    nb_indiv_features = 50   # number of features computed from each individuals
    nb_group_features = 41   # number of features computed from the group as a whole
    path_base_model = "./path_to_pre_trained_model" # path to the saved pre-trained model on the "Base" dimension only

    # To store all the outputs and instantiate the model later
    list_outputs = []

    # INPUT MODULE

    # Initiate individual inputs
    input_p0 = tf.keras.Input(
        shape=(timesteps, nb_indiv_features), name="p0"
    )

    input_p1 = tf.keras.Input(
        shape=(timesteps, nb_indiv_features), name="p1"
    )

    input_p2 = tf.keras.Input(
        shape=(timesteps, nb_indiv_features), name="p2"
    )

    # Initiate group input
    input_group = tf.keras.Input(
        shape=(timesteps, nb_group_features), name="Group"
    )

    # Concatenate inputs to instantiate the model later
    list_inputs = [input_p0, input_p1, input_p2, input_group]

    # BASE MODULE

    # Load the saved pre-trained model on a specific dimension (i.e., Social in TBD-T or Task in TBD-S)
    base_model = tf.keras.models.load_model(path_base_model)
    # Make the parameters trainable as a first step to integrate reciprocal impact
    base_model.trainable = True

    # Instantiate the pre-trained model without the layers used to make predictions
    base_model = Model(base_model.inputs, base_model.layers[-6].output, name="Base_model")
    base_model = base_model(list_inputs)

    # TARGET MODULE
    x = Dense(16, activation='relu', name="Dense_group")(base_model)

    # Multitask setting
    for i in range(nb_task):
        x_prev = Dense(8, activation='relu')(x)
        x_prev = Dense(4, activation='relu')(x_prev)

        # OUTPUT MODULE

        # Only one unit in the dense layer as TBDs are designed to predict only one dimension of cohesion
        output = Dense(1, activation='sigmoid', name="Output_t" + str(i + 1))(x_prev)

        # Concatenate outputs to instantiate the model later
        list_outputs.append(output)

    # Create the model instance
    model = Model(
        inputs=list_inputs,
        outputs=list_outputs,
    )
    model._name = "TBD"

    return model
