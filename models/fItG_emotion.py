# File created By: Lucien Maman
# Last updated: 12/08/2022
# ---------------------------------------------------------------------------
"""
The fItG with emotions architecture.

Argument: The strategy to predict emotions. Accepted values are "Bottom-up" (default) and "Top-down"

Specificities:
- Requires having emotion labels
- Requires having features computed from both individuals and the group as a whole

Designed by: Lucien Maman

It was presented at @ACII2021.
Please cite @ACII2021 to refer to the fItG with emotions.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import tensorflow as tf

from tensorflow.keras.layers import LSTM, Dense, Dropout, concatenate, Flatten
from tensorflow.keras.models import Model

# ---------------------------------------------------------------------------
# Function returning the fItG_emotion architecture
# ---------------------------------------------------------------------------

def create_fItG_emotion(strategy="Bottom-up"):

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

    # BOTTOM-UP MODULE

    if strategy == "Bottom-up":
        # concatenate all the outputs from the Individual module
        x_concat_emo_indiv = concatenate([p0_lstm, p1_lstm, p2_lstm])
        dense_emo_1 = Dense(64, activation='sigmoid')(x_concat_emo_indiv)
        dense_emo_2 = Dense(16, activation='sigmoid')(dense_emo_1)
        flatten_bu_output = Flatten()(dense_emo_2)

    # GROUP MODULE

    # Concatenate all the outputs of the shared layers with the group input
    concat_list = [p0_lstm, p1_lstm, p2_lstm, input_group]

    # Input shape of next layers
    group_lstm_cell = nb_indiv_features * 3 + nb_group_features

    x_concat = concatenate(concat_list)
    x = Dense(64, activation='relu', input_shape=(timesteps, group_lstm_cell), name="Dense_group_init")(x_concat)
    x = LSTM(32, return_sequences=False, input_shape=(timesteps, group_lstm_cell), name="Group_LSTM")(x)
    x_drop = Dropout(0.2, name="Dropout_group")(x)
    dense_td_output = Dense(16, activation='relu', name="Dense_group")(x_drop)

    # OUTPUT MODULE

    # Multitask setting
    for i in range(nb_task):

        output_cohesion = Dense(nb_output, activation='sigmoid', name="Output_cohesion_t" + str(i+1))(dense_td_output)
        # Concatenate cohesion outputs to instantiate the model later
        list_outputs.append(output_cohesion)

        if strategy == "Bottom-up":
            # Prediction of emotion from the Bottom-up module
            output_emotion = Dense(1, activation='sigmoid', name="Output_emotion_t" + str(i+1))(flatten_bu_output)

        # TOP-DOWN MODULE
        else:
            # Prediction of emotion from the Top-Down module
            output_emotion = Dense(1, activation='sigmoid', name="Output_emotion_t" + str(i+1))(dense_td_output)

        # Concatenate emotion outputs to instantiate the model later
        list_outputs.append(output_emotion)

    # Create the model instance
    model = Model(
        inputs=list_inputs,
        outputs=list_outputs,
    )
    model._name = "fItG_emotion_" + strategy

    return model
