# File created By: Lucien Maman
# Last updated: 12/08/2022
# ---------------------------------------------------------------------------
"""
The FI-LSTM architecture.

Specificities: None

Designed by: Lucien Maman

It was presented at @ICMI2021.
Please cite @ICMI2021 to refer to the FI-LSTM.
"""
# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import tensorflow as tf

from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Model

# ---------------------------------------------------------------------------
# Function returning the FI-LSTM architecture
# ---------------------------------------------------------------------------

def create_FILSTM():

    # Define the parameters of the model. Default numbers are the ones used in the paper(s)
    nb_task = 5              # Here, 5 tasks are predicted, following the ones presented in @GAME-ON
    timesteps = 6 * nb_task  # number of timesteps (here 6 segments of 20s) x the number of task (here set to 5)
    nb_output = 2            # number of outputs predicted: if >1 --> multilabel setting
    nb_features = 191        # total number of features (computed from both the group as a whole and individuals)

    # To store all the outputs and instantiate the model later
    list_outputs = []

    # INPUT MODULE

    # Initiate the input
    input_x = tf.keras.Input(
        shape=(timesteps, nb_features), name="Input_x"
    )

    # Create a list containing the input to instantiate the model later
    list_inputs = [input_x]

    # TIME MODULE

    x = LSTM(timesteps, return_sequences=False, input_shape=(timesteps, nb_features), name="LSTM_Time")(input_x)
    x = Dropout(0.2, name="Dropout_Time")(x)
    x = Dense(16, activation='relu', name="Dense_Time_1")(x)
    x = Dense(8, activation='relu', name="Dense_Time_2")(x)

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
    model._name = "FI-LSTM"

    return model
