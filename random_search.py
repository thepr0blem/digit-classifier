import random as rd
import numpy as np

from load_data import create_model

parameters_dct = {"no_of_filters": [8, 16, 32, 48, 64],
                  "kern_size": [3, 4, 5],
                  "max_pool": [2, 3],
                  "dropout_perc": [0.05, 0.1, 0.2, 0.3, 0.4],
                  "dense_size": [64, 128, 192, 256, 320],
                  "optimizers": ["adam", "adamax", "nadam", "RMSProp"]
                  }


def run_random_search(params, no_of_searches=1):
    """Perform random search on hyper parameters list, saves models and validation accuracies.

    Args:
        params: Dictionary with hyperparameters for CNN random search.
        no_of_searches: How many times random search is executed.

    Returns:
        List of accuracies performed on validation set for each iteration.
    """
    val_accs_list = []

    for i in range(no_of_searches):
        # Creating a tuple for each iteration of random search with selected parameters

        params_dict = {"iteration": i + 1,
                       "no_of_filters": rd.choice(params["no_of_filters"]),
                       "kern_size": rd.choice(params["kern_size"]),
                       "max_pool": rd.choice(params["max_pool"]),
                       "dropout_perc_conv": rd.choice(params["dropout_perc"]),
                       "dropout_perc_dens": rd.choice(params["dropout_perc"]),
                       "dense_size": rd.choice(params["dense_size"]),
                       "optimizer": rd.choice(params["optimizers"]),
                       }

        np.save(r".\random_search\params\params_dict_{}.npy".format(i), params_dict)

        hist_dict = create_model(it=i + 1,
                                 no_of_filters=params_dict["no_of_filters"],
                                 kern_size=params_dict["kern_size"],
                                 max_p_size=params_dict["max_pool"],
                                 drop_perc_conv=params_dict["dropout_perc_conv"],
                                 drop_perc_dense=params_dict["dropout_perc_dens"],
                                 dens_size=params_dict["dense_size"],
                                 optimizer=params_dict["optimizer"]
                                 )

        val_accs_list.append(hist_dict['val_acc'][-1])

    np.save(r".\random_search\val_accs_list.npy", val_accs_list)

    return val_accs_list
