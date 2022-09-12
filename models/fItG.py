# File created By: Lucien Maman
# Last updated: 12/08/2022
# ---------------------------------------------------------------------------
"""
The fItG architecture.

Specificities: Requires having features computed from both individuals and the group as a whole

Designed by: Lucien Maman

It was initially presented at @ACII2021, and is also used in @ICMI2021 and @IGTD2021.
Please cite @ACII2021 to refer to the fItG.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import tensorflow as tf

from tensorflow.keras.layers import LSTM, Dense, Dropout, concatenate
from tensorflow.keras.models import Model

# ---------------------------------------------------------------------------
# Function returning the fItG architecture
# ---------------------------------------------------------------------------

def create_fItG():

    # Define the parameters of the model. Default numbers are the ones used in the paper(s)
    nb_task = 5              # Here, 5 tasks are predicted, following the ones presented in @GAME-ON
    timesteps = 6 * nb_task  # number of timesteps (here 6 segments of 20s) x the number of task (here set to 5)
    nb_output = 2            # number of outputs predicted: if >1 --> multilabel setting
    nb_indiv_features = 50   # number of features computed from each individuals
    nb_group_features = 41   # number of features computed from the group as a whole

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

    # INDIVIDUAL MODULE

    # Instantiate shared layers
    shared_dense = Dense(nb_indiv_features, activation='relu', name="Shared_Dense")
    shared_lstm = LSTM(nb_indiv_features, return_sequences=True, input_shape=(timesteps, nb_indiv_features),
                       name="Shared_Individual_LSTM")

    # Give individual inputs to shared layers
    p0_dense = shared_dense(input_p0)
    p0_lstm = shared_lstm(p0_dense)

    p1_dense = shared_dense(input_p1)
    p1_lstm = shared_lstm(p1_dense)

    p2_dense = shared_dense(input_p2)
    p2_lstm = shared_lstm(p2_dense)

    # GROUP MODULE

    # Concatenate all the outputs of the shared layers with the group input
    concat_list = [p0_lstm, p1_lstm, p2_lstm, input_group]

    # Input shape of next layers
    group_lstm_cell = nb_indiv_features * 3 + nb_group_features

    x_concat = concatenate(concat_list)
    x = Dense(64, activation='relu', input_shape=(timesteps, group_lstm_cell), name="Dense_group_init")(x_concat)
    x = LSTM(32, return_sequences=False, input_shape=(timesteps, group_lstm_cell), name="Group_LSTM")(x)
    x_drop = Dropout(0.2, name="Dropout_group")(x)
    x = Dense(16, activation='relu', name="Dense_group")(x_drop)

    # OUTPUT MODULE

    # Multitask setting
    for i in range(nb_task):
        output = Dense(nb_output, activation='sigmoid', name="Output_t" + str(i + 1))(x)

        # Concatenate outputs to instantiate the model later
        list_outputs.append(output)

    # Create the model instance
    model = Model(
        inputs=list_inputs,
        outputs=list_outputs,
    )
    model._name = "fItG"

    return model
