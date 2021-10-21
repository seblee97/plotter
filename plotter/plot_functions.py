import os
from typing import List, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm

TIME_UNIT = "time_unit"
SUMMARY_FIGSIZE = (9, 6)
DISCRETE = "discrete"
CONTINUOUS = "continuous"


def smooth_data(data: List[float], window_width: int) -> List[float]:
    """Calculates moving average of list of values

    Args:
        data: raw, un-smoothed data.
        window_width: width over which to take moving averags

    Returns:
        smoothed_values: averaged data
    """

    def _smooth(single_dataset):
        cumulative_sum = np.cumsum(single_dataset, dtype=np.float32)
        cumulative_sum[window_width:] = (
            cumulative_sum[window_width:] - cumulative_sum[:-window_width]
        )
        # explicitly forcing data type to 32 bits avoids floating point errors
        # with constant dat.
        smoothed_values = np.array(
            cumulative_sum[window_width - 1 :] / window_width, dtype=np.float32
        )
        return smoothed_values

    if all(isinstance(d, list) for d in data):
        smoothed_data = []
        for dataset in data:
            smoothed_data.append(_smooth(dataset))
    elif all(
        (isinstance(d, float) or isinstance(d, int) or isinstance(d, np.int64))
        for d in data
    ):
        smoothed_data = _smooth(data)

    return smoothed_data


def _get_cmap(colormap: str):
    cmap = cm.get_cmap(colormap)

    if isinstance(cmap, mpl.colors.ListedColormap):
        cmap_type = DISCRETE
        color_cycle = mpl.cycler(color=cmap.colors)
        mpl.rcParams["axes.prop_cycle"] = color_cycle
        return cmap, cmap_type
    else:
        cmap_type = CONTINUOUS
        return cmap, cmap_type


def plot_multi_seed_multi_run(
    folder_path: str,
    exp_names: List[str],
    window_width: int,
    linewidth: int = 3,
    colormap: Union[str, None] = None,
):
    """Expected structure of folder_path is:

    - folder_path
        |_ run_1
        |   |_ seed_0
        |   |_ seed_1
        |   |_ ...
        |   |_ seed_M
        |
        |_ run_2
        |   |_ seed_0
        |   |_ seed_1
        |   |_ ...
        |   |_ seed_M
        |
        |_ ...
        |_ ...
        |_ run_N
            |_ seed_0
            |_ seed_1
            |_ ...
            |_ seed_M

    with a file called data_logger.csv in each leaf folder.
    """
    experiment_folders = {
        exp_name: os.path.join(folder_path, exp_name) for exp_name in exp_names
    }

    tag_set = {}

    # arbitrarily select one seed's dataframe for each run to find set of column names
    for exp, exp_path in experiment_folders.items():
        ex_seed = [
            f for f in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path, f))
        ][0]

        ex_df = pd.read_csv(os.path.join(exp_path, ex_seed, "data_logger.csv"))
        tag_subset = list(ex_df.columns)
        for tag in tag_subset:
            if tag not in tag_set:
                tag_set[tag] = []
            tag_set[tag].append(exp)

    cmap, cmap_type = _get_cmap(colormap)

    for tag, relevant_experiments in tag_set.items():
        if tag not in []:
            print(tag)

            fig = plt.figure(figsize=(SUMMARY_FIGSIZE))
            for exp_i, exp in enumerate(relevant_experiments):

                if cmap_type == DISCRETE:
                    color = cmap(exp_i / len(relevant_experiments))
                    print(exp_i / len(relevant_experiments))
                    print(color)

                print(cmap_type)

                attribute_data = []
                seed_folders = [
                    f
                    for f in os.listdir(experiment_folders[exp])
                    if os.path.isdir(os.path.join(experiment_folders[exp], f))
                ]

                for seed in seed_folders:
                    df = pd.read_csv(
                        os.path.join(experiment_folders[exp], seed, "data_logger.csv")
                    )
                    tag_data = df[tag].dropna()
                    attribute_data.append(tag_data)

                if len(attribute_data):
                    mean_attribute_data = np.mean(attribute_data, axis=0)
                    std_attribute_data = np.std(attribute_data, axis=0)
                    smooth_mean_data = smooth_data(
                        mean_attribute_data, window_width=window_width
                    )
                    smooth_std_data = smooth_data(
                        std_attribute_data, window_width=window_width
                    )
                    if len(smooth_mean_data):
                        scaled_x = (len(df) / len(smooth_mean_data)) * np.arange(
                            len(smooth_mean_data)
                        )
                        kwargs = {"linewidth": linewidth, "label": exp}
                        if cmap_type == DISCRETE:
                            kwargs["color"] = color
                        plt.plot(scaled_x, smooth_mean_data, **kwargs)
                        kwargs = {"alpha": 0.3}
                        if cmap_type == DISCRETE:
                            kwargs["color"] = color
                        plt.fill_between(
                            scaled_x,
                            smooth_mean_data - smooth_std_data,
                            smooth_mean_data + smooth_std_data,
                            **kwargs,
                        )
            plt.legend(
                bbox_to_anchor=(
                    1.01,
                    1.0,
                ),
                loc="upper left",
                ncol=1,
                borderaxespad=0.0,
            )
            plt.xlabel(TIME_UNIT)
            plt.ylabel(tag)

            fig.tight_layout()

            os.makedirs(os.path.join(folder_path, "figures"), exist_ok=True)
            fig.savefig(
                os.path.join(
                    folder_path, "figures", f"{tag}_plot_multi_seed_multi_run.pdf"
                ),
                dpi=100,
            )
            plt.close()
