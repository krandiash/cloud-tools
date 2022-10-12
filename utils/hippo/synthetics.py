from ..config import *


def arima_s4_sweep():

    sweep = prod(
        [
            flag("experiment", ["s4-synthetic-arima"]),
            lzip([
                flag("dataset.p", [0, 0, 0, 0, 0, 1, 2, 3, 5, 10, 20]),
                flag("dataset.d", [0, 1, 1, 2, 1, 0, 0, 0, 0, 0, 0]),
                flag("dataset.q", [0, 0, 1, 2, 2, 0, 0, 0, 0, 0, 0]),
                flag("dataset.lag", [1, 1, 1, 2, 2, 1, 2, 3, 5, 10, 20]),
            ]),
            flag("dataset.horizon", [10]),
        ]
    )

    return sweep

def arima_s4_sweep_01k():

    sweep = prod(
        [
            flag("experiment", ["s4-synthetic-arima"]),
            flag("dataset.p", [0]),
            flag("dataset.d", [1]),
            lzip([
                flag("dataset.q", [0, 1, 2, 3, 5, 10, 20]),
                flag("dataset.lag", [0, 1, 2, 3, 5, 10, 20]),
            ]),
            flag("dataset.horizon", [10]),
        ]
    )

    return sweep

def arima_s4_sweep_k00():

    sweep = prod(
        [
            flag("experiment", ["s4-synthetic-arima"]),
            flag("dataset.d", [0]),
            flag("dataset.q", [0]),
            lzip([
                flag("dataset.p", [1, 2, 3, 5, 10, 20]),
                flag("dataset.lag", [1, 2, 3, 5, 10, 20]),
            ]),
            flag("dataset.horizon", [10]),
        ]
    )

    return sweep

def single_arima_s4_sweep_k00():

    sweep = prod(
        [
            flag("experiment", ["s4-synthetic-arima"]),
            flag("dataset.d", [0]),
            flag("dataset.q", [0]),
            lzip([
                flag("dataset.p", [1, 2, 3, 5, 10, 20]),
                flag("dataset.lag", [1, 2, 3, 5, 10, 20]),
            ]),
            flag("dataset.n_ts", [1]),
            flag("dataset.nobs_per_ts", [1000]),
            flag("dataset.horizon", [10]),
        ]
    )

    return sweep

def single_arima_s4_sweep_0dq():
    # Can we learn differencing?
    sweep = prod(
        [
            flag("experiment", ["s4-synthetic-arima"]),
            flag("dataset.p", [0]),
            flag("dataset.d", [1, 2]),
            flag("model.n_layers", [1, 2, 4]),
            lzip([
                flag("dataset.q", [0, 1, 2, 3, 5, 10, 20]),
                flag("dataset.lag", [0, 1, 2, 3, 5, 10, 20]),
            ]),
            flag("dataset.n_ts", [1]),
            flag("dataset.nobs_per_ts", [1000]),
            flag("dataset.horizon", [10]),
        ]
    )

    return sweep

def single_arima_s4_sweep_k00_20():

    sweep = prod(
        [
            flag("experiment", ["s4-synthetic-arima"]),
            flag("dataset.d", [0]),
            flag("dataset.q", [0]),
            flag("dataset.p", [20]),
            lzip([
                flag("dataset.lag", [20]),
            ]),
            flag("dataset.n_ts", [1]),
            flag("dataset.nobs_per_ts", [1000]),
            flag("dataset.horizon", [1]), # set horizon to 1
        ]
    )

    return sweep

def single_arima_s4_sweep_k00_20_big():

    sweep = prod(
        [
            flag("experiment", ["s4-synthetic-arima"]),
            flag("model.n_layers", [1, 2, 4]), # may need a bigger model to learn for longer horizon
            flag("dataset.d", [0]),
            flag("dataset.q", [0]),
            flag("dataset.p", [20]),
            lzip([
                flag("dataset.lag", [20]),
            ]),
            flag("dataset.n_ts", [1]),
            flag("dataset.nobs_per_ts", [1000]),
            flag("dataset.horizon", [10]),
        ]
    )
    return sweep

def single_arima_s4_sweep_0dq_020():
    # Can we learn differencing?
    sweep = prod(
        [
            flag("experiment", ["s4-synthetic-arima"]),
            flag("dataset.p", [0]),
            flag("dataset.d", [2]),
            flag("dataset.q", [0]),
            flag("dataset.lag", [1, 2, 4, 8, 16, 32]),
            flag("model.n_layers", [1, 2, 4]),
            flag("dataset.n_ts", [1]),
            flag("dataset.nobs_per_ts", [1000]),
            flag("dataset.horizon", [1]),
        ]
    )

    return sweep

def single_arima_s4_sweep_0dq_020_measures():
    # Can we learn differencing?
    sweep = prod(
        [
            flag("experiment", ["s4-synthetic-arima"]),
            flag("dataset.p", [0]),
            flag("dataset.d", [2]),
            flag("dataset.q", [0]),
            flag("dataset.lag", [1, 2, 4]),
            flag("model.n_layers", [1]),
            flag("dataset.n_ts", [1]),
            flag("dataset.nobs_per_ts", [1000]),
            flag("dataset.horizon", [1]),
            flag("model.layer.measure", ['random_lin', 'random_inv', 'fourier', 'legs']),
        ]
    )

    return sweep

# Layer norm and activation messes up perofrmance

def single_arima_s4_sweep_0dq_sweep_1():

    sweep = prod(
        [
            flag("experiment", ["s4-synthetic-arima"]),
            flag("dataset.p", [0]),
            flag("dataset.d", [1, 2, 3]),
            flag("dataset.q", [0]),
            flag("dataset.lag", [1, 2, 3]),
            flag("model.n_layers", [1]),
            flag("dataset.n_ts", [1]),
            flag("dataset.nobs_per_ts", [1000]),
            flag("dataset.horizon", [1]),
            flag("model.layer.activation", ['null', 'gelu']),
            flag("model.norm", ['null', 'layer']),
            flag("model.layer.measure", ['random-linear']),
        ]
    )

    return sweep

def single_arima_s4_sweep_0dq_sweep_2():

    sweep = prod(
        [
            flag("experiment", ["s4-synthetic-arima"]),
            flag("dataset.p", [0]),
            lzip([
                flag("dataset.d", [1, 2, 3]),
                flag("dataset.lag", [1, 2, 3]),
            ]),
            flag("dataset.q", [0]),
            flag("model.n_layers", [1]),
            flag("dataset.n_ts", [1]),
            flag("dataset.nobs_per_ts", [1000]),
            flag("dataset.horizon", [1]),
            flag("model.layer.activation", ['null']),
            flag("model.norm", ['null']),
            flag("model.layer.measure", ['random-linear', 'fourier', 'random-inv', 'legs', 'legt']),
        ]
    )

    return sweep



def single_arima_s4_sweep_0dq_sweep_3():

    sweep = prod(
        [
            flag("dataset.seed", [0, 1, 2]),
            flag("experiment", ["s4-synthetic-arima"]),
            flag("dataset.p", [0]),
            lzip([
                flag("dataset.d", [1, 2, 3]),
                flag("dataset.lag", [1, 2, 3]),
            ]),
            flag("dataset.q", [0]), 
            flag("model.n_layers", [1]),
            flag("dataset.n_ts", [1]),
            flag("dataset.nobs_per_ts", [1000]),
            flag("dataset.horizon", [1]),
            flag("model.layer.activation", ['null', 'gelu']),
            flag("model.norm", ['null', 'layer']),
            flag("model.layer.measure", ['random-linear', 'legs']),
            flag("+task.norm", ['mean', 'null']),
        ]
    )

    return sweep

def single_arima_s4_sweep_p00_sweep_1():

    sweep = prod(
        [
            flag("experiment", ["s4-synthetic-arima"]),
            lzip([
                flag("dataset.p", [0, 1, 2, 3, 5, 10, 20]),
                flag("dataset.lag", [0, 1, 2, 3, 5, 10, 20]),
            ]),
            flag("dataset.d", [0]),
            flag("dataset.q", [0]),
            flag("model.n_layers", [1]),
            flag("dataset.n_ts", [1]),
            flag("dataset.nobs_per_ts", [1000]),
            flag("dataset.horizon", [1]),
            flag("model.layer.activation", ['null', 'gelu']),
            flag("model.norm", ['null', 'layer']),
            flag("model.layer.measure", ['random-linear']),
        ]
    )

    return sweep

def single_arima_s4_sweep_00q_sweep_1():

    sweep = prod(
        [
            flag("experiment", ["s4-synthetic-arima"]),
            lzip([
                flag("dataset.q", [0, 1, 2, 3, 5, 10, 20]),
                flag("dataset.lag", [1, 1, 2, 3, 5, 10, 20]),
            ]),
            flag("dataset.d", [0]),
            flag("dataset.p", [0]),
            flag("model.n_layers", [1]),
            flag("dataset.n_ts", [1]),
            flag("dataset.nobs_per_ts", [1000]),
            flag("dataset.horizon", [1]),
            flag("model.layer.activation", ['null', 'gelu']),
            flag("model.norm", ['null', 'layer']),
            flag("model.layer.measure", ['random-linear']),
        ]
    )

    return sweep


def single_arima_s4_sweep_p0q_sweep_1():

    sweep = prod(
        [
            flag("experiment", ["s4-synthetic-arima"]),
            lzip([
                flag("dataset.p", [1, 2, 3, 5, 10, 20]),
                flag("dataset.q", [1, 2, 3, 5, 10, 20]),
                flag("dataset.lag", [1, 2, 3, 5, 10, 20]),
            ]),
            flag("dataset.d", [0]),
            flag("model.n_layers", [1]),
            flag("dataset.n_ts", [1]),
            flag("dataset.nobs_per_ts", [1000]),
            flag("dataset.horizon", [1]),
            flag("model.layer.activation", ['null', 'gelu']),
            flag("model.norm", ['null', 'layer']),
            flag("model.layer.measure", ['random-linear']),
        ]
    )

    return sweep


def single_arima_s4_sweep_p1q_sweep_1():

    sweep = prod(
        [
            flag("experiment", ["s4-synthetic-arima"]),
            lzip([
                flag("dataset.p", [1, 2, 3, 5, 10, 20]),
                flag("dataset.q", [1, 2, 3, 5, 10, 20]),
                flag("dataset.lag", [1, 2, 3, 5, 10, 20]),
            ]),
            flag("dataset.d", [1]),
            flag("model.n_layers", [1]),
            flag("dataset.n_ts", [1]),
            flag("dataset.nobs_per_ts", [1000]),
            flag("dataset.horizon", [1]),
            flag("model.layer.activation", ['null', 'gelu']),
            flag("model.norm", ['null', 'layer']),
            flag("model.layer.measure", ['random-linear']),
        ]
    )

    return sweep

def single_arima_s4_sweep_arima_ets_sweep_1():

    sweep = prod(
        [
            flag("experiment", ["s4-synthetic-arima"]),
            lzip([
                flag("dataset.p", [0, 0, 1]),
                flag("dataset.q", [1, 2, 1]),
                flag("dataset.lag", [1, 2, 2]),
            ]),
            flag("dataset.d", [1]),
            flag("model.n_layers", [1]),
            flag("dataset.n_ts", [1]),
            flag("dataset.nobs_per_ts", [1000]),
            flag("dataset.horizon", [1]),
            flag("model.layer.activation", ['null', 'gelu']),
            flag("model.norm", ['null', 'layer']),
            flag("model.layer.measure", ['random-linear']),
        ]
    )

    return sweep

def single_arima_s4_sweep_arima_seasonal_p1q_sweep_1():
    # Non-stationary, seasonal data

    sweep = prod(
        [
            flag("dataset.seed", [0, 1, 2]),
            flag("experiment", ["s4-synthetic-arima"]),
            lzip([
                flag("dataset.p", [1, 2, 3, 5, 10, 20]),
                flag("dataset.q", [1, 2, 3, 5, 10, 20]),
                flag("dataset.lag", [1, 2, 3, 5, 10, 20]),
            ]),
            flag("dataset.seasonal", [
                '"{W:{p:1,d:0,q:0,seed:42,scale:0.1}}"',
                # '"{W:{p:0,d:1,q:0,seed:42,scale:0.1}}"',
                # '"{W:{p:0,d:1,q:1,seed:42,scale:0.1}}"',
            ]),
            flag("dataset.d", [1]),
            flag("model.n_layers", [1]),
            flag("dataset.n_ts", [1]),
            flag("dataset.nobs_per_ts", [1000]),
            flag("dataset.horizon", [1]),
            flag("model.layer.activation", ['null', 'gelu']),
            flag("model.norm", ['null', 'layer']),
            flag("+task.norm", ['null']),
            lzip([
                flag("model.layer.n_ssm", [1, 1, 4, 4]),
                flag("model.layer.measure", ['legs', 'random-linear', 'hippo', 'all']),
            ]),
        ]
    )

    return sweep


# Sweep ideas
# Effect of d_state
# Sweep lag vs. order of ARIMA
# Effect of differencing
# Effect of constant


# ARIMA(0, 0, 0) (white noise) and ARIMA(1, 0, 0) are fit perfectly (error equals the std of the noise)
# The fit for ARIMA(0, 1, k) models improves from 0 -> 1 -> 2, and seems rather high for ARIMA(0, 1, 0)
    # TODO: try differencing 
# The fit for ARIMA(0, 2, 2) seems pretty bad, gap is increasing from train -> val -> test
    # TODO: compare to best fit ARIMA(0, 2, 2) model
# The fit for ARIMA(2, 0, 0) has error around 2x the std of the noise
    # TODO: check that this is the expected error -- yes

####################################
######## Final ARIMA Sweeps ########
####################################

# AR Models

def arima_AR_final_sweep_1():

    sweep = prod(
        [
            flag("dataset.seed", [0, 1, 2]),
            flag("experiment", ["s4-synthetic-arima"]),
            flag("model.n_layers", [1]),
            flag("dataset.n_ts", [1]),
            flag("dataset.nobs_per_ts", [1000]),
            flag("dataset.horizon", [1]),

            flag("dataset.d", [0]),
            flag("dataset.q", [0]), 
            lzip([
                flag("dataset.p", [1, 2, 3]),
                flag("dataset.lag", [1, 2, 3]),
            ]),
            flag("model.layer.activation", ['gelu']),
            flag("model.norm", ['layer']),
            lzip([
                flag("model.layer.measure", ['random-linear', 'legs', 'hippo', 'all']),
                flag("model.layer.n_ssm", [1, 1, 4, 4]),
            ]),
            flag("+task.norm", ['mean', 'null']),
        ]
    )
    return sweep


def arima_AR_final_sweep_conv_1():

    sweep = prod(
        [
            flag("dataset.seed", [0, 1, 2]),
            flag("experiment", ["conv1d-synthetic-arima"]),
            flag("model/layer", ['conv1d']),
            flag("model.d_model", [64]),
            flag("model.n_layers", [1]),
            flag("model.dropout", [0.0]),
            flag("dataset.n_ts", [1]),
            flag("dataset.nobs_per_ts", [1000]),
            flag("dataset.horizon", [1]),
            flag("dataset.d", [0]),
            flag("dataset.q", [0]), 
            lzip([
                flag("dataset.p", [1, 2, 3]),
                flag("dataset.lag", [1, 2, 3]),
            ]),
            flag("model.layer.activation", ['gelu']),
            flag("model.norm", ['layer']),
            flag("+task.norm", ['mean', 'null']),
        ]
    )
    return sweep

def arima_AR_final_sweep_lstm_1():

    sweep = prod(
        [
            flag("dataset.seed", [0, 1, 2]),
            flag("experiment", ["lstm-synthetic-arima"]),
            flag("model", ['rnn/lstm']),
            flag("model.d_model", [64]),
            flag("model.n_layers", [1]),
            flag("model.dropout", [0.0]),
            flag("dataset.n_ts", [1]),
            flag("dataset.nobs_per_ts", [1000]),
            flag("dataset.horizon", [1]),
            flag("dataset.d", [0]),
            flag("dataset.q", [0]), 
            lzip([
                flag("dataset.p", [1, 2, 3]),
                flag("dataset.lag", [1, 2, 3]),
            ]),
            flag("model.layer.d_hidden", [64]),
            flag("model.norm", ['layer']),
            flag("+task.norm", ['mean', 'null']),
        ]
    )
    return sweep

def arima_AR_final_sweep_transformer_1():

    sweep = prod(
        [
            flag("dataset.seed", [0, 1, 2]),
            flag("experiment", ["transformer-synthetic-arima"]),
            flag("model", ['transformer']),
            flag("model.d_model", [64]),
            flag("model.n_layers", [1]),
            flag("model.dropout", [0.0]),
            flag("dataset.n_ts", [1]),
            flag("dataset.nobs_per_ts", [1000]),
            flag("dataset.horizon", [1]),

            flag("dataset.d", [0]),
            flag("dataset.q", [0]), 
            lzip([
                flag("dataset.p", [1, 2, 3]),
                flag("dataset.lag", [1, 2, 3]),
            ]),
            flag("model.norm", ['layer']),
            flag("+task.norm", ['mean', 'null']),
        ]
    )
    return sweep

# MA Models
def arima_MA_final_sweep_1():

    sweep = prod(
        [
            flag("dataset.seed", [0, 1, 2]),
            flag("experiment", ["s4-synthetic-arima"]),
            flag("model.n_layers", [1]),
            flag("dataset.n_ts", [1]),
            flag("dataset.nobs_per_ts", [1000]),
            flag("dataset.horizon", [1]),

            flag("dataset.d", [0]),
            flag("dataset.p", [0]), 
            lzip([
                flag("dataset.q", [1, 2, 3]),
                flag("dataset.lag", [1, 2, 3]),
            ]),
            flag("model.layer.activation", ['gelu']),
            flag("model.norm", ['layer']),
            lzip([
                flag("model.layer.measure", ['random-linear', 'legs', 'hippo', 'all']),
                flag("model.layer.n_ssm", [1, 1, 4, 4]),
            ]),
            flag("+task.norm", ['mean', 'null']),
        ]
    )
    return sweep


def arima_MA_final_sweep_conv_1():

    sweep = prod(
        [
            flag("dataset.seed", [0, 1, 2]),
            flag("experiment", ["conv1d-synthetic-arima"]),
            flag("model/layer", ['conv1d']),
            flag("model.d_model", [64]),
            flag("model.n_layers", [1]),
            flag("model.dropout", [0.0]),
            flag("dataset.n_ts", [1]),
            flag("dataset.nobs_per_ts", [1000]),
            flag("dataset.horizon", [1]),
            flag("dataset.d", [0]),
            flag("dataset.p", [0]), 
            lzip([
                flag("dataset.q", [1, 2, 3]),
                flag("dataset.lag", [1, 2, 3]),
            ]),
            flag("model.layer.activation", ['gelu']),
            flag("model.norm", ['layer']),
            flag("+task.norm", ['mean', 'null']),
        ]
    )
    return sweep

def arima_MA_final_sweep_lstm_1():

    sweep = prod(
        [
            flag("dataset.seed", [0, 1, 2]),
            flag("experiment", ["lstm-synthetic-arima"]),
            flag("model", ['rnn/lstm']),
            flag("model.d_model", [64]),
            flag("model.n_layers", [1]),
            flag("model.dropout", [0.0]),
            flag("dataset.n_ts", [1]),
            flag("dataset.nobs_per_ts", [1000]),
            flag("dataset.horizon", [1]),
            flag("dataset.d", [0]),
            flag("dataset.p", [0]), 
            lzip([
                flag("dataset.q", [1, 2, 3]),
                flag("dataset.lag", [1, 2, 3]),
            ]),
            flag("model.layer.d_hidden", [64]),
            flag("model.norm", ['layer']),
            flag("+task.norm", ['mean', 'null']),
        ]
    )
    return sweep

def arima_MA_final_sweep_transformer_1():

    sweep = prod(
        [
            flag("dataset.seed", [0, 1, 2]),
            flag("experiment", ["transformer-synthetic-arima"]),
            flag("model", ['transformer']),
            flag("model.d_model", [64]),
            flag("model.n_layers", [1]),
            flag("model.dropout", [0.0]),
            flag("dataset.n_ts", [1]),
            flag("dataset.nobs_per_ts", [1000]),
            flag("dataset.horizon", [1]),

            flag("dataset.d", [0]),
            flag("dataset.p", [0]), 
            lzip([
                flag("dataset.q", [1, 2, 3]),
                flag("dataset.lag", [1, 2, 3]),
            ]),
            flag("model.norm", ['layer']),
            flag("+task.norm", ['mean', 'null']),
        ]
    )
    return sweep



# ARMA Models
def arima_ARMA_final_sweep_1():

    sweep = prod(
        [
            flag("dataset.seed", [0, 1, 2]),
            flag("experiment", ["s4-synthetic-arima"]),
            flag("model.n_layers", [1]),
            flag("dataset.n_ts", [1]),
            flag("dataset.nobs_per_ts", [1000]),
            flag("dataset.horizon", [1]),

            flag("dataset.d", [0]),
            lzip([
                flag("dataset.p", [1, 2, 3]), 
                flag("dataset.q", [1, 2, 3]),
                flag("dataset.lag", [1, 2, 3]),
            ]),
            flag("model.layer.activation", ['gelu']),
            flag("model.norm", ['layer']),
            lzip([
                flag("model.layer.measure", ['random-linear', 'legs', 'hippo', 'all']),
                flag("model.layer.n_ssm", [1, 1, 4, 4]),
            ]),
            flag("+task.norm", ['mean', 'null']),
        ]
    )
    return sweep


def arima_ARMA_final_sweep_conv_1():

    sweep = prod(
        [
            flag("dataset.seed", [0, 1, 2]),
            flag("experiment", ["conv1d-synthetic-arima"]),
            flag("model/layer", ['conv1d']),
            flag("model.d_model", [64]),
            flag("model.n_layers", [1]),
            flag("model.dropout", [0.0]),
            flag("dataset.n_ts", [1]),
            flag("dataset.nobs_per_ts", [1000]),
            flag("dataset.horizon", [1]),
            flag("dataset.d", [0]),
            lzip([
                flag("dataset.p", [1, 2, 3]), 
                flag("dataset.q", [1, 2, 3]),
                flag("dataset.lag", [1, 2, 3]),
            ]),
            flag("model.layer.activation", ['gelu']),
            flag("model.norm", ['layer']),
            flag("+task.norm", ['mean', 'null']),
        ]
    )
    return sweep

def arima_ARMA_final_sweep_lstm_1():

    sweep = prod(
        [
            flag("dataset.seed", [0, 1, 2]),
            flag("experiment", ["lstm-synthetic-arima"]),
            flag("model", ['rnn/lstm']),
            flag("model.d_model", [64]),
            flag("model.n_layers", [1]),
            flag("model.dropout", [0.0]),
            flag("dataset.n_ts", [1]),
            flag("dataset.nobs_per_ts", [1000]),
            flag("dataset.horizon", [1]),
            flag("dataset.d", [0]),
            lzip([
                flag("dataset.p", [1, 2, 3]), 
                flag("dataset.q", [1, 2, 3]),
                flag("dataset.lag", [1, 2, 3]),
            ]),
            flag("model.layer.d_hidden", [64]),
            flag("model.norm", ['layer']),
            flag("+task.norm", ['mean', 'null']),
        ]
    )
    return sweep

def arima_ARMA_final_sweep_transformer_1():

    sweep = prod(
        [
            flag("dataset.seed", [0, 1, 2]),
            flag("experiment", ["transformer-synthetic-arima"]),
            flag("model", ['transformer']),
            flag("model.d_model", [64]),
            flag("model.n_layers", [1]),
            flag("model.dropout", [0.0]),
            flag("dataset.n_ts", [1]),
            flag("dataset.nobs_per_ts", [1000]),
            flag("dataset.horizon", [1]),

            flag("dataset.d", [0]),
            lzip([
                flag("dataset.p", [1, 2, 3]), 
                flag("dataset.q", [1, 2, 3]),
                flag("dataset.lag", [1, 2, 3]),
            ]),
            flag("model.norm", ['layer']),
            flag("+task.norm", ['mean', 'null']),
        ]
    )
    return sweep



# ARIMA Models
def arima_ARIMA_unitroot_final_sweep_1():

    sweep = prod(
        [
            flag("dataset.seed", [0, 1, 2]),
            flag("experiment", ["s4-synthetic-arima"]),
            flag("model.n_layers", [1]),
            flag("dataset.n_ts", [1]),
            flag("dataset.nobs_per_ts", [1000]),
            flag("dataset.horizon", [1]),

            flag("dataset.p", [0]),
            flag("dataset.q", [0]),
            lzip([
                flag("dataset.d", [1, 2, 3]), 
                flag("dataset.lag", [1, 2, 3]),
            ]),
            flag("model.layer.activation", ['gelu']),
            flag("model.norm", ['layer', 'null']),
            flag("+task.norm", ['mean', 'null']),
            lzip([
                flag("model.layer.measure", ['random-linear', 'legs', 'hippo', 'all']),
                flag("model.layer.n_ssm", [1, 1, 4, 4]),
            ]),
        ]
    )
    return sweep



def arima_ARIMA_unitroot_final_sweep_conv_1():

    sweep = prod(
        [
            flag("dataset.seed", [0, 1, 2]),
            flag("experiment", ["conv1d-synthetic-arima"]),
            flag("model/layer", ['conv1d']),
            flag("model.d_model", [64]),
            flag("model.n_layers", [1]),
            flag("model.dropout", [0.0]),
            flag("dataset.n_ts", [1]),
            flag("dataset.nobs_per_ts", [1000]),
            flag("dataset.horizon", [1]),
            flag("dataset.p", [0]),
            flag("dataset.q", [0]),
            lzip([
                flag("dataset.d", [1, 2, 3]), 
                flag("dataset.lag", [1, 2, 3]),
            ]),
            flag("model.layer.activation", ['gelu']),
            flag("model.norm", ['layer', 'null']),
            flag("+task.norm", ['mean', 'null']),
        ]
    )
    return sweep

def arima_ARIMA_unitroot_final_sweep_lstm_1():

    sweep = prod(
        [
            flag("dataset.seed", [0, 1, 2]),
            flag("experiment", ["lstm-synthetic-arima"]),
            flag("model", ['rnn/lstm']),
            flag("model.d_model", [64]),
            flag("model.n_layers", [1]),
            flag("model.dropout", [0.0]),
            flag("dataset.n_ts", [1]),
            flag("dataset.nobs_per_ts", [1000]),
            flag("dataset.horizon", [1]),
            flag("dataset.p", [0]),
            flag("dataset.q", [0]),
            lzip([
                flag("dataset.d", [1, 2, 3]), 
                flag("dataset.lag", [1, 2, 3]),
            ]),
            flag("model.layer.d_hidden", [64]),
            flag("model.norm", ['layer', 'null']),
            flag("+task.norm", ['mean', 'null']),
        ]
    )
    return sweep

def arima_ARIMA_unitroot_final_sweep_transformer_1():

    sweep = prod(
        [
            flag("dataset.seed", [0, 1, 2]),
            flag("experiment", ["transformer-synthetic-arima"]),
            flag("model", ['transformer']),
            flag("model.d_model", [64]),
            flag("model.n_layers", [1]),
            flag("model.dropout", [0.0]),
            flag("dataset.n_ts", [1]),
            flag("dataset.nobs_per_ts", [1000]),
            flag("dataset.horizon", [1]),

            flag("dataset.p", [0]),
            flag("dataset.q", [0]),
            lzip([
                flag("dataset.d", [1, 2, 3]), 
                flag("dataset.lag", [1, 2, 3]),
            ]),
            flag("model.norm", ['layer', 'null']),
            flag("+task.norm", ['mean', 'null']),
        ]
    )
    return sweep

# ETS Models
def arima_ARIMA_ets_final_sweep_1():

    sweep = prod(
        [
            flag("dataset.seed", [0, 1, 2]),
            flag("experiment", ["s4-synthetic-arima"]),
            flag("model.n_layers", [1]),
            flag("dataset.n_ts", [1]),
            flag("dataset.nobs_per_ts", [1000]),
            flag("dataset.horizon", [1]),

            lzip([
                flag("dataset.p", [0, 0, 0, 1]),
                flag("dataset.d", [1, 2, 1, 1]),
                flag("dataset.q", [1, 2, 2, 2]),
                flag("dataset.lag", [1, 2, 2, 2]),
            ]),
            
            flag("model.layer.activation", ['gelu']),
            flag("model.norm", ['layer', 'null']),
            flag("+task.norm", ['mean', 'null']),
            lzip([
                flag("model.layer.measure", ['random-linear', 'legs', 'hippo', 'all']),
                flag("model.layer.n_ssm", [1, 1, 4, 4]),
            ]),
        ]
    )
    return sweep


def arima_ARIMA_ets_final_sweep_conv_1():

    sweep = prod(
        [
            flag("dataset.seed", [0, 1, 2]),
            flag("experiment", ["conv1d-synthetic-arima"]),
            flag("model/layer", ['conv1d']),
            flag("model.d_model", [64]),
            flag("model.n_layers", [1]),
            flag("model.dropout", [0.0]),
            flag("dataset.n_ts", [1]),
            flag("dataset.nobs_per_ts", [1000]),
            flag("dataset.horizon", [1]),
            lzip([
                flag("dataset.p", [0, 0, 0, 1]),
                flag("dataset.d", [1, 2, 1, 1]),
                flag("dataset.q", [1, 2, 2, 2]),
                flag("dataset.lag", [1, 2, 2, 2]),
            ]),
            flag("model.layer.activation", ['gelu']),
            flag("model.norm", ['layer', 'null']),
            flag("+task.norm", ['mean', 'null']),
        ]
    )
    return sweep

def arima_ARIMA_ets_final_sweep_lstm_1():

    sweep = prod(
        [
            flag("dataset.seed", [0, 1, 2]),
            flag("experiment", ["lstm-synthetic-arima"]),
            flag("model", ['rnn/lstm']),
            flag("model.d_model", [64]),
            flag("model.n_layers", [1]),
            flag("model.dropout", [0.0]),
            flag("dataset.n_ts", [1]),
            flag("dataset.nobs_per_ts", [1000]),
            flag("dataset.horizon", [1]),
            lzip([
                flag("dataset.p", [0, 0, 0, 1]),
                flag("dataset.d", [1, 2, 1, 1]),
                flag("dataset.q", [1, 2, 2, 2]),
                flag("dataset.lag", [1, 2, 2, 2]),
            ]),
            flag("model.layer.d_hidden", [64]),
            flag("model.norm", ['layer', 'null']),
            flag("+task.norm", ['mean', 'null']),
        ]
    )
    return sweep

def arima_ARIMA_ets_final_sweep_transformer_1():

    sweep = prod(
        [
            flag("dataset.seed", [0, 1, 2]),
            flag("experiment", ["transformer-synthetic-arima"]),
            flag("model", ['transformer']),
            flag("model.d_model", [64]),
            flag("model.n_layers", [1]),
            flag("model.dropout", [0.0]),
            flag("dataset.n_ts", [1]),
            flag("dataset.nobs_per_ts", [1000]),
            flag("dataset.horizon", [1]),

            lzip([
                flag("dataset.p", [0, 0, 0, 1]),
                flag("dataset.d", [1, 2, 1, 1]),
                flag("dataset.q", [1, 2, 2, 2]),
                flag("dataset.lag", [1, 2, 2, 2]),
            ]),
            flag("model.norm", ['layer', 'null']),
            flag("+task.norm", ['mean', 'null']),
        ]
    )
    return sweep


# ARIMA Models
def arima_ARIMA_final_sweep_1():

    sweep = prod(
        [
            flag("dataset.seed", [0, 1, 2]),
            flag("experiment", ["s4-synthetic-arima"]),
            flag("model.n_layers", [1]),
            flag("dataset.n_ts", [1]),
            flag("dataset.nobs_per_ts", [1000]),
            flag("dataset.horizon", [1]),

            lzip([
                flag("dataset.p", [1, 2, 3]),
                flag("dataset.d", [1, 1, 1]),
                flag("dataset.q", [1, 2, 3]),
                flag("dataset.lag", [2, 3, 4]),
            ]),
            
            flag("model.layer.activation", ['gelu']),
            flag("model.norm", ['layer', 'null']),
            flag("+task.norm", ['mean', 'null']),
            lzip([
                flag("model.layer.measure", ['random-linear', 'legs', 'hippo', 'all']),
                flag("model.layer.n_ssm", [1, 1, 4, 4]),
            ]),
        ]
    )
    return sweep



def arima_ARIMA_final_sweep_conv_1():

    sweep = prod(
        [
            flag("dataset.seed", [0, 1, 2]),
            flag("experiment", ["conv1d-synthetic-arima"]),
            flag("model/layer", ['conv1d']),
            flag("model.d_model", [64]),
            flag("model.n_layers", [1]),
            flag("model.dropout", [0.0]),
            flag("dataset.n_ts", [1]),
            flag("dataset.nobs_per_ts", [1000]),
            flag("dataset.horizon", [1]),
            lzip([
                flag("dataset.p", [1, 2, 3]),
                flag("dataset.d", [1, 1, 1]),
                flag("dataset.q", [1, 2, 3]),
                flag("dataset.lag", [2, 3, 4]),
            ]),
            flag("model.layer.activation", ['gelu']),
            flag("model.norm", ['layer', 'null']),
            flag("+task.norm", ['mean', 'null']),
        ]
    )
    return sweep

def arima_ARIMA_final_sweep_lstm_1():

    sweep = prod(
        [
            flag("dataset.seed", [0, 1, 2]),
            flag("experiment", ["lstm-synthetic-arima"]),
            flag("model", ['rnn/lstm']),
            flag("model.d_model", [64]),
            flag("model.n_layers", [1]),
            flag("model.dropout", [0.0]),
            flag("dataset.n_ts", [1]),
            flag("dataset.nobs_per_ts", [1000]),
            flag("dataset.horizon", [1]),
            lzip([
                flag("dataset.p", [1, 2, 3]),
                flag("dataset.d", [1, 1, 1]),
                flag("dataset.q", [1, 2, 3]),
                flag("dataset.lag", [2, 3, 4]),
            ]),
            flag("model.layer.d_hidden", [64]),
            flag("model.norm", ['layer', 'null']),
            flag("+task.norm", ['mean', 'null']),
        ]
    )
    return sweep

def arima_ARIMA_final_sweep_transformer_1():

    sweep = prod(
        [
            flag("dataset.seed", [0, 1, 2]),
            flag("experiment", ["transformer-synthetic-arima"]),
            flag("model", ['transformer']),
            flag("model.d_model", [64]),
            flag("model.n_layers", [1]),
            flag("model.dropout", [0.0]),
            flag("dataset.n_ts", [1]),
            flag("dataset.nobs_per_ts", [1000]),
            flag("dataset.horizon", [1]),

            lzip([
                flag("dataset.p", [1, 2, 3]),
                flag("dataset.d", [1, 1, 1]),
                flag("dataset.q", [1, 2, 3]),
                flag("dataset.lag", [2, 3, 4]),
            ]),
            flag("model.norm", ['layer', 'null']),
            flag("+task.norm", ['mean', 'null']),
        ]
    )
    return sweep

# SARIMA Models
# single_arima_s4_sweep_arima_seasonal_p1q_sweep_1