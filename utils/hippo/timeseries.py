from utils.config import *

###############################################################################
#                                                                             #
#                              Informer Datasets                              #
#                                                                             #
###############################################################################

def ett1h_longest_common_flags():

    sweep = prod(
        [
            flag("experiment", ["s4-informer-etth"]),
            flag(
                "dataset.size",
                [
                    "[720,336,720]",
                ],
            ),
            flag("dataset.timeenc", [1]),
            flag("dataset.variant", [0]),
            flag("dataset.features", ["S"]),
            flag("task.metrics", ["[mse,mae]"]),
            flag("dataset.target", ["OT"]),
            flag("trainer.max_epochs", [5]),
            flag("loader.batch_size", [50]),
            flag("scheduler", ["timm_cosine"]),
            flag("scheduler.t_initial", [5]),
            flag("scheduler.warmup_t", [0]),
        ]
    )
    return sweep

def ett2h_longest_common_flags():

    sweep = prod(
        [
            flag("experiment", ["s4-informer-etth"]),
            flag(
                "dataset.size",
                [
                    "[336,336,720]",
                ],
            ),
            flag("dataset.timeenc", [1]),
            flag("dataset.variant", [1]),
            flag("dataset.features", ["S"]),
            flag("task.metrics", ["[mse,mae]"]),
            flag("dataset.target", ["OT"]),
            flag("trainer.max_epochs", [10]),
            flag("loader.batch_size", [50]),
            flag("scheduler", ["timm_cosine"]),
            flag("scheduler.t_initial", [10]),
            flag("scheduler.warmup_t", [0]),
        ]
    )
    return sweep

def ettm_longest_common_flags():
    
    sweep = prod(
        [
            flag("experiment", ["s4-informer-ettm"]),
            flag(
                "dataset.size",
                [
                    "[672,672,672]",
                ],
            ),
            flag("dataset.timeenc", [1]),
            flag("dataset.variant", [0]),
            flag("dataset.features", ["S"]),
            flag("task.metrics", ["[mse,mae]"]),
            flag("dataset.target", ["OT"]),
            flag("trainer.max_epochs", [5]),
            flag("loader.batch_size", [50]),
            flag("scheduler", ["timm_cosine"]),
            flag("scheduler.t_initial", [5]),
            flag("scheduler.warmup_t", [0]),
        ]
    )
    return sweep


def ett1h_shorter_common_flags():

    sweep = prod(
        [
            flag("experiment", ["s4-informer-etth"]),
            flag(
                "dataset.size",
                [
                    "[720,336,336]",
                    "[720,336,168]",
                    "[720,168,48]",
                    "[720,168,24]",
                ],
            ),
            flag("dataset.timeenc", [1]),
            flag("dataset.variant", [0]),
            flag("dataset.features", ["S"]),
            flag("task.metrics", ["[mse,mae]"]),
            flag("dataset.target", ["OT"]),
            flag("trainer.max_epochs", [5]),
            flag("loader.batch_size", [50]),
            flag("scheduler", ["timm_cosine"]),
            flag("scheduler.t_initial", [5]),
            flag("scheduler.warmup_t", [0]),
        ]
    )
    return sweep


def etth1_global_1():
    # bidirectional sweep
    # bidirectional vs. unidirectional is very close
    sweep = prod(
        [
            flag("train.seed", [0, 1, 2, 3, 4]),
            flag("model.n_layers", [4]),
            flag("model.dropout", [0.30]),
            flag("optimizer.lr", [0.01]),
            flag("model.layer.postact", ["glu"]),
            flag("model.layer.bidirectional", [True, False]),
            flag("model.layer.n_ssm", [128]),
            flag("optimizer.weight_decay", [0.0]),
            lzip(
                [
                    flag(
                        "model.layer.measure",
                        [
                            "legs",
                            "fourier_diag",
                        ],
                    ),
                    flag("+model.layer.rank_weight", [1, 1]),
                ]
            ),
        ]
    )
    return prod([sweep, ett1h_longest_common_flags()])

def etth1_global_2():
    # lr_dt sweep
    sweep = prod(
        [
            flag("train.seed", [0, 1, 2, 3, 4]),
            flag("model.n_layers", [4]),
            flag("model.dropout", [0.30]),
            flag("optimizer.lr", [0.01]),
            flag("model.layer.postact", ["glu"]),
            flag("model.layer.bidirectional", [False]),
            flag("+model.layer.lr_dt", [0.001, 0.01]),
            flag("model.layer.n_ssm", [128]),
            flag("optimizer.weight_decay", [0.0]),
            lzip(
                [
                    flag(
                        "model.layer.measure",
                        [
                            "legs",
                            "fourier_diag",
                        ],
                    ),
                    flag("+model.layer.rank_weight", [1, 1]),
                ]
            ),
        ]
    )
    return prod([sweep, ett1h_longest_common_flags()])

def etth1_global_3():
    # Wd + dropout sweep
    # need dropout >= 0.2 (lower dropout generalizes worse)
    # Best combinations (trends on both measures are very similar):
    # dropout 0.2, wd 0.1, 0.2
    # dropout 0.3, wd 0.05
    # best val: dropout 0.2, wd 0.2, fourier_diag
    sweep = prod(
        [
            flag("train.seed", [0, 1, 2, 3, 4]),
            flag("model.n_layers", [4]),
            flag("optimizer.lr", [0.01]),
            flag("model.layer.postact", ["glu"]),
            flag("model.layer.bidirectional", [False]),
            flag("+model.layer.lr_dt", [0.01]),
            flag("model.layer.n_ssm", [128]),
            flag("model.dropout", [0.1, 0.2, 0.3]),
            flag("optimizer.weight_decay", [0.05, 0.10, 0.20]),
            lzip(
                [
                    flag(
                        "model.layer.measure",
                        [
                            "legs",
                            "fourier_diag",
                        ],
                    ),
                    flag("+model.layer.rank_weight", [1, 1]),
                ]
            ),
        ]
    )
    return prod([sweep, ett1h_longest_common_flags()])


def etth1_global_4():
    # d_state sweep
    # d_state 64 still works the best, although 16 is not bad (256 seems worse)
    sweep = prod(
        [
            flag("train.seed", [0, 1, 2, 3, 4]),
            flag("model.n_layers", [4]),
            flag("optimizer.lr", [0.01]),
            flag("model.layer.postact", ["glu"]),
            flag("model.layer.bidirectional", [False]),
            flag("+model.layer.lr_dt", [0.01]),
            flag("model.layer.n_ssm", [128]),
            flag("model.layer.d_state", [16, 64, 256]),
            flag("model.dropout", [0.2]),
            flag("optimizer.weight_decay", [0.20]),
            lzip(
                [
                    flag(
                        "model.layer.measure",
                        [
                            "legs",
                            "fourier_diag",
                        ],
                    ),
                    flag("+model.layer.rank_weight", [1, 1]),
                ]
            ),
        ]
    )
    return prod([sweep, ett1h_longest_common_flags()])

def etth1_global_5():
    # sweep normalization, prenorm
    # prenorm False works better
    # batch norm is really bad, layer norm is better
    sweep = prod(
        [
            flag("train.seed", [0, 1, 2, 3, 4]),
            flag("model.n_layers", [4]),
            flag("optimizer.lr", [0.01]),
            flag("model.layer.postact", ["glu"]),
            flag("model.layer.bidirectional", [False]),
            flag("+model.layer.lr_dt", [0.01]),
            flag("model.layer.n_ssm", [128]),
            flag("model.norm", ['batch', 'layer']),
            flag("model.prenorm", [True, False]),
            flag("model.dropout", [0.2]),
            flag("optimizer.weight_decay", [0.20]),
            lzip(
                [
                    flag(
                        "model.layer.measure",
                        [
                            "legs",
                            "fourier_diag",
                        ],
                    ),
                    flag("+model.layer.rank_weight", [1, 1]),
                ]
            ),
        ]
    )
    return prod([sweep, ett1h_longest_common_flags()])


def etth1_global_6():
    # sweep measures and best (dropout, wd) settings
    sweep = prod(
        [
            flag("train.seed", [5, 6, 7, 8, 9]),
            flag("model.n_layers", [4]),
            flag("optimizer.lr", [0.01]),
            flag("model.layer.postact", ["glu"]),
            flag("model.layer.bidirectional", [False]),
            flag("+model.layer.lr_dt", [0.01]),
            flag("model.layer.n_ssm", [128]),
            lzip([
                flag("model.dropout", [0.2, 0.2, 0.3]),
                flag("optimizer.weight_decay", [0.20, 0.10, 0.05]),
            ]),
            lzip(
                [
                    flag(
                        "model.layer.measure",
                        [
                            "legs",
                            "legsd",
                            "hippo",
                            "fourier",
                            "fourier_diag",
                            "fourier_decay",
                            "fourier_old",
                            "random",
                        ],
                    ),
                    flag("+model.layer.rank_weight", [1, 1, 1, 1, 1, 1, 1, 0]),
                ]
            ),
        ]
    )
    return prod([sweep, ett1h_longest_common_flags()])

def etth1_global_7():
    # sweep measures and a few other high dropout settings
    # dropout 0.4 seems good on test, but val is higher
    sweep = prod(
        [
            flag("train.seed", [5, 6, 7, 8, 9]),
            flag("model.n_layers", [4]),
            flag("optimizer.lr", [0.01]),
            flag("model.layer.postact", ["glu"]),
            flag("model.layer.bidirectional", [False]),
            flag("+model.layer.lr_dt", [0.01]),
            flag("model.layer.n_ssm", [128]),
            lzip([
                flag("model.dropout", [0.3, 0.4]),
                flag("optimizer.weight_decay", [0.0, 0.0]),
            ]),
            lzip(
                [
                    flag(
                        "model.layer.measure",
                        [
                            "legs",
                            "legsd",
                            "hippo",
                            "fourier",
                            "fourier_diag",
                            "fourier_decay",
                            "fourier_old",
                            "random",
                        ],
                    ),
                    flag("+model.layer.rank_weight", [1, 1, 1, 1, 1, 1, 1, 0]),
                ]
            ),
        ]
    )
    return prod([sweep, ett1h_longest_common_flags()])


def etth1_shorter_sweep():
    # sweep measures and a few other high dropout settings
    sweep = prod(
        [
            flag("train.seed", [5, 6, 7, 8, 9]),
            flag("model.n_layers", [4]),
            flag("optimizer.lr", [0.01]),
            flag("model.layer.postact", ["glu"]),
            flag("model.layer.bidirectional", [False]),
            flag("+model.layer.lr_dt", [0.01]),
            flag("model.layer.n_ssm", [128]),
            lzip([
                flag("model.dropout", [0.2, 0.2, 0.3]),
                flag("optimizer.weight_decay", [0.20, 0.10, 0.05]),
            ]),
            lzip(
                [
                    flag(
                        "model.layer.measure",
                        [
                            "legs",
                            "legsd",
                            "hippo",
                            "fourier",
                            "fourier_diag",
                            "fourier_decay",
                            "fourier_old",
                            "random",
                        ],
                    ),
                    flag("+model.layer.rank_weight", [1, 1, 1, 1, 1, 1, 1, 0]),
                ]
            ),
        ]
    )
    return prod([sweep, ett1h_shorter_common_flags()])

def etth2_global_1():
    # sweep measures and a few other high dropout settings
    # 0.3 dropout / 0.2 wd seems best
    sweep = prod(
        [
            flag("train.seed", [5, 6, 7, 8, 9]),
            flag("model.n_layers", [4]),
            flag("optimizer.lr", [0.01]),
            flag("model.layer.postact", ["glu"]),
            flag("model.layer.bidirectional", [False]),
            flag("+model.layer.lr_dt", [0.01]),
            flag("model.layer.n_ssm", [128]),
            flag("model.dropout", [0.1, 0.2, 0.3]),
            flag("optimizer.weight_decay", [0.05, 0.10, 0.20]),
            lzip(
                [
                    flag(
                        "model.layer.measure",
                        [
                            "legs",
                            "fourier_diag",
                        ],
                    ),
                    flag("+model.layer.rank_weight", [1, 1]),
                ]
            ),
        ]
    )
    return prod([sweep, ett2h_longest_common_flags()])


def etth2_global_2():
    sweep = prod(
        [
            flag("train.seed", [5, 6, 7, 8, 9]),
            flag("model.n_layers", [4]),
            flag("optimizer.lr", [0.01]),
            flag("model.layer.postact", ["glu"]),
            flag("model.layer.bidirectional", [False]),
            flag("+model.layer.lr_dt", [0.01]),
            flag("model.layer.n_ssm", [128]),
            flag("model.dropout", [0.3]),
            flag("optimizer.weight_decay", [0.50, 1.0]),
            lzip(
                [
                    flag(
                        "model.layer.measure",
                        [
                            "legs",
                            "fourier_diag",
                        ],
                    ),
                    flag("+model.layer.rank_weight", [1, 1]),
                ]
            ),
        ]
    )
    return prod([sweep, ett2h_longest_common_flags()])

def etth2_global_3():
    # hippo does best
    sweep = prod(
        [
            flag("train.seed", [0, 1, 2, 3, 4]),
            flag("model.n_layers", [4]),
            flag("optimizer.lr", [0.01]),
            flag("model.layer.postact", ["glu"]),
            flag("model.layer.bidirectional", [False]),
            flag("+model.layer.lr_dt", [0.01]),
            flag("model.layer.n_ssm", [128]),
            flag("model.dropout", [0.3]),
            flag("optimizer.weight_decay", [0.2]),
            lzip(
                [
                    flag(
                        "model.layer.measure",
                        [
                            "legs",
                            "legsd",
                            "hippo",
                            "fourier",
                            "fourier_diag",
                            "fourier_decay",
                            "fourier_old",
                            "random",
                        ],
                    ),
                    flag("+model.layer.rank_weight", [1, 1, 1, 1, 1, 1, 1, 0]),
                ]
            ),
        ]
    )
    return prod([sweep, ett2h_longest_common_flags()])


def ettm_global_1():
    sweep = prod(
        [
            flag("train.seed", [0, 1, 2, 3, 4]),
            flag("model.n_layers", [4]),
            flag("optimizer.lr", [0.01]),
            flag("model.layer.postact", ["glu"]),
            flag("model.layer.bidirectional", [False]),
            flag("+model.layer.lr_dt", [0.01]),
            flag("model.layer.n_ssm", [128]),
            flag("model.dropout", [0.1, 0.2, 0.3]),
            flag("optimizer.weight_decay", [0.05, 0.10, 0.20]),
            lzip(
                [
                    flag(
                        "model.layer.measure",
                        [
                            "legs",
                            "fourier_diag",
                        ],
                    ),
                    flag("+model.layer.rank_weight", [1, 1]),
                ]
            ),
        ]
    )
    return prod([ettm_longest_common_flags(), sweep])

def ettm_global_2():
    sweep = prod(
        [
            flag("train.seed", [0, 1, 2, 3, 4]),
            flag("model.n_layers", [4]),
            flag("optimizer.lr", [0.01]),
            flag("model.layer.postact", ["glu"]),
            flag("model.layer.bidirectional", [False]),
            flag("+model.layer.lr_dt", [0.01]),
            flag("model.layer.n_ssm", [128]),
            flag("model.dropout", [0.1, 0.2, 0.3]),
            flag("optimizer.weight_decay", [0.05, 0.10, 0.20]),
            flag("model.norm", [None, 'layer']),
            flag("task", ['forecasting']),
            lzip(
                [
                    flag(
                        "model.layer.measure",
                        [
                            "legs",
                            "random-linear",
                        ],
                    ),
                ]
            ),
        ]
    )
    return prod([ettm_longest_common_flags(), sweep])

def ettm_ablation_1():
    sweep = prod(
        [
            flag("train.seed", [5, 6, 7, 8, 9]),
            flag("model.n_layers", [4]),
            flag("optimizer.lr", [0.01]),
            flag("model.layer.postact", ["glu"]),
            flag("model.layer.bidirectional", [False]),
            flag("+model.layer.lr_dt", [0.01]),
            flag("model.layer.n_ssm", [128]),
            flag("model.dropout", [0.3]),
            flag("optimizer.weight_decay", [0.10]),
            flag("model.norm", [None, 'layer']),
            flag("+task.norm", [None, 'mean']), 
            flag("task", ['forecasting']),
            lzip(
                [
                    flag(
                        "model.layer.measure",
                        [
                            "legs",
                        ],
                    ),
                ]
            ),
        ]
    )
    return prod([ettm_longest_common_flags(), sweep])


###############################################################################
#
# Monash
#
###############################################################################

def monash_sweep_1():

    sweep = prod(
        [
            flag("experiment", ["forecasting/s4-monash"]),
            flag(
                "dataset.dataset_name",
                [
                    "hospital",  # val check interval needs to less than 814
                    "fred_md",
                    "traffic_weekly",  # val check interval needs to be less than 431
                    "traffic_hourly",
                    # "dominick", # loss is nan
                    "solar_10_minutes",
                    "kdd_cup",
                    "melbourne_pedestrian_counts",
                    "aus_elecdemand",
                    # "rideshare", # needs really short val check interval
                    "electricity_weekly",
                    "electricity_hourly",
                ],
            ),
            flag("dataset.weighted_sampler", [True, False]),
            flag("model.dropout", [0.0]),
            flag("model.d_model", [128]),
            flag("model.n_layers", [1]),
            flag("optimizer.weight_decay", [0.0]),
            flag("model.norm", ["layer", "null"]),
            flag("task.norm", ["mean", "revnorm", "null"]),
            flag("task.loss", ["mse", "mae"]),
            flag("+trainer.val_check_interval", [50]),
            flag("scheduler.num_training_steps", [20000]),
            # flag("model.layer.n_ssm", [1]),
            # flag("model.layer.measure", ['all', 'legs', 'hippo']),
        ]
    )

    return sweep



def monash_sweep_test():
    # ONLY FOR TESTING WHICH DATASETS ERROR OUT
    dataset_params = {
        # loss is nan
        # "dominick": 50,
        # "m4_daily": 50,
        # "m4_monthly": 50,
        "m1_quarterly": 50,
        "m3_yearly": 50,
        "m3_quarterly": 50,
        "m3_monthly": 50,
        "m3_other": 50,
        "m1_yearly": 50,
        "m1_monthly": 50,
        "m4_hourly": 50,
        "m4_weekly": 50,
        "m4_quarterly": 50,

        "car_parts": 50,
        "hospital": 50,
        "fred_md": 50,
        "traffic_weekly": 50,
        "traffic_hourly": 50,
        "solar_10_minutes": 50,
        "kdd_cup": 50,
        "melbourne_pedestrian_counts": 50,
        "aus_elecdemand": 50,
        "rideshare": 50,
        "electricity_weekly": 50,
        "electricity_hourly": 50,
        "nn5_daily": 50,
        "nn5_weekly": 50,

        "tourism_yearly": 50,
        "tourism_quarterly": 50,
        "tourism_monthly": 50,
        "sunspot": 50,
        "us_births": 50,
        "saugeen_river_flow": 50, 
        "kaggle_web_traffic_weekly": 50,
        "temperature_rain": 50,
        "vehicle_trips": 50,
        "bitcoin": 50,
        "weather": 50,
        "covid_deaths": 50,
        "solar_weekly": 50,   
    }

    sweep = prod(
        [
            flag("train.seed", [1]),
            flag("experiment", ["forecasting/s4-monash"]),
            flag("dataset.dataset_name", list(dataset_params.keys())),
            flag("dataset.weighted_sampler", [True]),
            flag("model.dropout", [0.1]),
            flag("model.d_model", [128]),
            flag("model.n_layers", [2]),
            flag("optimizer.weight_decay", [0.0]),
            flag("model.norm", ["layer"]),
            flag("task.norm", ["mean"]),
            flag("task.loss", ["mse"]),
            flag("+trainer.val_check_interval", [50]),
            flag("trainer.max_steps", [100]),
            flag("scheduler.num_training_steps", [100]),
            
        ]
    )

    return sweep

        

def monash_sweep_2():

    dataset_params = {
        # loss is nan
        # "dominick": 50,
        # "m4_daily": 50,
        # "m4_monthly": 50,
        "m1_quarterly": 50,
        "m3_yearly": 50,
        "m3_quarterly": 50,
        "m3_monthly": 50,
        "m3_other": 50,
        "m1_yearly": 25,
        "m1_monthly": 50,
        "m4_hourly": 50,
        "m4_weekly": 50,
        "m4_quarterly": 50,
    }

    sweep = prod(
        [
            flag("train.seed", [1, 2, 3]),
            flag("experiment", ["forecasting/s4-monash"]),
            lzip([
                flag("dataset.dataset_name", list(dataset_params.keys())),
                flag("+trainer.val_check_interval", list(dataset_params.values())),
            ]),
            flag("dataset.weighted_sampler", [True, False]),
            flag("model.dropout", [0.1]),
            flag("model.d_model", [128]),
            flag("model.n_layers", [2]),
            flag("optimizer.weight_decay", [0.0]),
            flag("model.norm", ["layer", "null"]),
            flag("task.norm", ["mean", "null"]),
            flag("task.loss", ["mse"]),
            flag("trainer.max_steps", [20000]),
            flag("scheduler.num_training_steps", [20000]),
            lzip([
                flag("model.layer.n_ssm", [4, 4, 1]),
                flag("model.layer.measure", ['all', 'hippo', 'legs']),
            ]),
            
        ]
    )

    return sweep


        

def monash_sweep_3():

    dataset_params = {
        # loss is nan
        # "dominick": 50,
        # "m4_daily": 50,
        # "m4_monthly": 50,
    
        "car_parts": 50,
        "hospital": 50,
        "fred_md": 50,
        "traffic_weekly": 50,
        "traffic_hourly": 50,
        "solar_10_minutes": 50,
        "kdd_cup": 50,
        "melbourne_pedestrian_counts": 50,
        "aus_elecdemand": 50,
        "rideshare": 50,
        "electricity_weekly": 50,
        "electricity_hourly": 50,
        "nn5_daily": 50,
        "nn5_weekly": 50,
    }

    sweep = prod(
        [
            flag("train.seed", [1, 2, 3]),
            flag("experiment", ["forecasting/s4-monash"]),
            flag("dataset.dataset_name", list(dataset_params.keys())),
            flag("dataset.weighted_sampler", [True, False]),
            flag("model.dropout", [0.1]),
            flag("model.d_model", [128]),
            flag("model.n_layers", [2]),
            flag("optimizer.weight_decay", [0.0]),
            flag("model.norm", ["layer", "null"]),
            flag("task.norm", ["mean", "null"]),
            flag("task.loss", ["mse"]),
            flag("+trainer.val_check_interval", [50]),
            flag("trainer.max_steps", [20000]),
            flag("scheduler.num_training_steps", [20000]),
            lzip([
                flag("model.layer.n_ssm", [4, 4, 1]),
                flag("model.layer.measure", ['all', 'hippo', 'legs']),
            ]),
            
        ]
    )

    return sweep


        

def monash_sweep_4():

    dataset_params = {
        "tourism_yearly": 50,
        "tourism_quarterly": 50,
        "tourism_monthly": 50,
        "sunspot": 50,
        "us_births": 50,
        "saugeen_river_flow": 50, 
        "kaggle_web_traffic_weekly": 50,
        "temperature_rain": 50,
        "vehicle_trips": 50,
        "bitcoin": 50,
        "weather": 50,
        "covid_deaths": 50,
        "solar_weekly": 50,   
    }

    sweep = prod(
        [
            flag("train.seed", [1, 2, 3]),
            flag("experiment", ["forecasting/s4-monash"]),
            flag("dataset.dataset_name", list(dataset_params.keys())),
            flag("dataset.weighted_sampler", [True, False]),
            flag("model.dropout", [0.1]),
            flag("model.d_model", [128]),
            flag("model.n_layers", [2]),
            flag("optimizer.weight_decay", [0.0]),
            flag("model.norm", ["layer", "null"]),
            flag("task.norm", ["mean", "null"]),
            flag("task.loss", ["mse"]),
            flag("+trainer.val_check_interval", [50]),
            flag("trainer.max_steps", [20000]),
            flag("scheduler.num_training_steps", [20000]),
            lzip([
                flag("model.layer.n_ssm", [4, 4, 1]),
                flag("model.layer.measure", ['all', 'hippo', 'legs']),
            ]),
            
        ]
    )

    return sweep