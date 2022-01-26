"""Generic approach for scoring runway models
"""

import logging
import mlflow
import mlflow.sklearn
import sklearn.metrics
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import Any, Dict, List, Tuple
from pathlib import Path

from matplotlib.ticker import PercentFormatter

log = logging.getLogger(__name__)

def score_models(
        dat: pd.DataFrame,
        model_pipelines: Dict[str, Dict[str, Any]],
        inputs: Dict[str, Any],
        target: Dict[str, Any],
        airport_name: str,
        categories: Dict[str, List[Any]],
        ):
    """
    Parameters
    ----------
    dat : pd.DataFrame
        Full test/train dataset
    model_pipelines : Dict[str, Dict[str, Any]]
        Dictionary of model data and objects
    inputs : Dict[str, Any]
        Standard input specification
    target : Dict[str, Any]
        Standard target specification

    Returns
    -------
    None.
    """

    for m in model_pipelines:
        log.info(f"Computing metrics for {m}")

        metrics = dict()
        files = list()

        model = model_pipelines[m]["model"]

        # Pre-filter the data so that all subsequent metrics are computed only
        #   on the actual ML model values, not on defaults
        X_test, y_test, drop_fraction_test = _filter_data(
            model,
            dat.loc[(dat["test_sample"] == True), inputs.keys()],
            dat.loc[(dat["test_sample"] == True), target["name"]],
            )
        log.info("Drop fraction for test dataset is {:.3f}".format(
            drop_fraction_test,
            ))
        metrics["drop_fraction_test"] = drop_fraction_test
        metrics["num_testing_samples"] = (dat["test_sample"] == True).sum()

        X_train, y_train, drop_fraction_train = _filter_data(
            model,
            dat.loc[(dat["train_sample"] == True), inputs.keys()],
            dat.loc[(dat["train_sample"] == True), target["name"]],
            )
        log.info("Drop fraction for test dataset is {:.3f}".format(
            drop_fraction_train,
            ))
        metrics["drop_fraction_train"] = drop_fraction_train


        # Compute predictions for each dataset
        y_pred_test = (
            model
            .predict_df(X_test)["pred"]
            )
        y_pred_train = (
            model
            .predict_df(X_train)["pred"]
            )

        # Compute accuracy scores
        accuracy_test = sklearn.metrics.accuracy_score(
            y_test,
            y_pred_test,
            )
        log.info("Accuracy on test data is {:.3f}".format(
            accuracy_test,
            ))
        metrics["accuracy_test"] = accuracy_test

        accuracy_train = sklearn.metrics.accuracy_score(
            y_train,
            y_pred_train,
            )
        log.info("Accuracy on training data is {:.3f}".format(
            accuracy_train,
            ))
        metrics["accuracy_train"] = accuracy_train


        # Compute precision and recall
        precision_test, recall_test, _, _ = (
            sklearn.metrics.precision_recall_fscore_support(
                y_test,
                y_pred_test,
                average="weighted",
                )
            )
        log.info("Precision on test data is {:.3f}".format(
            precision_test,
            ))
        metrics["precision_test"] = precision_test

        log.info("Recall on test data is {:.3f}".format(
            recall_test,
            ))
        metrics["recall_test"] = recall_test

        precision_train, recall_train, _, _ = (
            sklearn.metrics.precision_recall_fscore_support(
                y_train,
                y_pred_train,
                average="weighted",
                )
            )
        log.info("Precision on train data is {:.3f}".format(
            precision_train,
            ))
        metrics["precision_train"] = precision_train

        log.info("Recall on test data is {:.3f}".format(
            recall_train,
            ))
        metrics["recall_train"] = recall_train


        # Compute AUC
        p_test = model.predict_proba(X_test)
        auc_test = sklearn.metrics.roc_auc_score(
            y_test,
            p_test,
            average="weighted",
            multi_class="ovo",
            labels=p_test.columns.tolist(),
            )
        log.info("AUC on test data is {:.3f}".format(
            auc_test,
            ))
        metrics["auc_test"] = auc_test

        p_train = model.predict_proba(X_train)
        auc_train = sklearn.metrics.roc_auc_score(
            y_train,
            p_train,
            average="weighted",
            multi_class="ovo",
            labels=p_train.columns.tolist(),
            )
        log.info("AUC on train data is {:.3f}".format(
            auc_train,
            ))
        metrics["auc_train"] = auc_train


        # Compute fraction of flights misclassificed to parallel runway
        misclass_to_parallel_frac_test = _misclass_to_parallel_runway(
            y_test,
            y_pred_test,
            )
        log.info("Parallel runway misclass for test is {:.3f}".format(
            misclass_to_parallel_frac_test,
            ))
        metrics["misclass_to_parallel_runway_frac_test"] = misclass_to_parallel_frac_test

        misclass_to_parallel_frac_train = _misclass_to_parallel_runway(
            y_train,
            y_pred_train,
            )
        log.info("Parallel runway misclass for train is {:.3f}".format(
            misclass_to_parallel_frac_train,
            ))
        metrics["misclass_to_parallel_runway_frac_train"] = misclass_to_parallel_frac_train


        # Create confusion matrices
        known_runways = categories["target"]
        known_runways.sort()
        log.info("Generating confusion matrices")
        cm_test_path = _confusion_matrix(
            y_test,
            y_pred_test,
            known_runways,
            airport_name,
            "test",
            )
        files = files + cm_test_path
        cm_train_path = _confusion_matrix(
            y_train,
            y_pred_train,
            known_runways,
            airport_name,
            "train",
            )
        files = files + cm_train_path


        # Create accuracy-over-time plots
        aot_test_path = _accuracy_over_time(
            (dat
             .loc[
                 (dat["test_sample"] == True),
                 list(inputs.keys()) + [target["name"]]]
             .rename(columns={
                 target["name"]:"actual",
                 })
             ),
            y_pred_test,
            model._default_response,
            airport_name,
            "Test",
            )
        files = files + aot_test_path
        aot_train_path = _accuracy_over_time(
            (dat
             .loc[
                 (dat["train_sample"] == True),
                 list(inputs.keys()) + [target["name"]]]
             .rename(columns={
                 target["name"]:"actual",
                 })
             ),
            y_pred_train,
            model._default_response,
            airport_name,
            "Training",
            )
        files = files + aot_train_path


        # Create distribution of accuracy plots
        doa_test_path = _distribution_of_accuracy(
            (X_test
             .join(dat["gufi"])
             .join(y_test)
             .rename(columns={
                 target["name"]:"actual",
                 })
             ),
            y_pred_test,
            airport_name,
            "Test",
            )
        files.append(doa_test_path)
        doa_train_path = _distribution_of_accuracy(
            (X_train
             .join(dat["gufi"])
             .join(y_train)
             .rename(columns={
                 target["name"]:"actual",
                 })
             ),
            y_pred_train,
            airport_name,
            "Training",
            )
        files.append(doa_train_path)


        # Analyze runways in / out of config
        inconf_test_act_file, inconf_test_act_frac = _runways_out_of_config(
            X_test.join(dat["gufi"]),
            y_test,
            airport_name,
            "act_test",
        )
        files = files + inconf_test_act_file
        metrics["fraction_in_config_actual_test"] = inconf_test_act_frac

        inconf_train_act_file, inconf_train_act_frac = _runways_out_of_config(
            X_train.join(dat["gufi"]),
            y_train,
            airport_name,
            "act_train",
        )
        files = files + inconf_train_act_file
        metrics["fraction_in_config_actual_train"] = inconf_train_act_frac

        inconf_test_pred_file, inconf_test_pred_frac = _runways_out_of_config(
            X_test.join(dat["gufi"]),
            y_pred_test,
            airport_name,
            "pred_test",
        )
        files = files + inconf_test_pred_file
        metrics["fraction_in_config_pred_test"] = inconf_test_pred_frac

        inconf_train_pred_file, inconf_train_pred_frac = _runways_out_of_config(
            X_train.join(dat["gufi"]),
            y_pred_train,
            airport_name,
            "pred_train",
        )
        files = files + inconf_train_pred_file
        metrics["fraction_in_config_pred_train"] = inconf_train_pred_frac


        # Log metrics to MLflow
        with mlflow.start_run(run_id=model_pipelines[m]["run_id"]):
            log.info("Logging metrics for model {} to MLFlow under run_id {}".format(
                m,
                model_pipelines[m]["run_id"],
                ))
            for key, value in metrics.items():
                mlflow.log_metric(key, value)

            for file in files:
                f = Path(file)
                if f.exists():
                    mlflow.log_artifact(file)
                    f.unlink()
                else:
                    log.error(f"Looked for {file} but could not find it locally")

    # Return true to help enforce pipeline order
    return True

def _filter_data(
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        ) -> Tuple[pd.DataFrame, pd.Series, float]:
    X_keep_idx, _, y_keep_idx, _ = model.filter(
        X=X,
        y=y,
        )
    keep_idx = (
        X_keep_idx
        & y_keep_idx
        )
    X_filt = X[keep_idx]
    y_filt = y[keep_idx]

    drop_fraction = 1-(len(y_filt)/len(y))

    return X_filt, y_filt, drop_fraction

def _misclass_to_parallel_runway(
        y_true: pd.Series,
        y_pred: pd.Series,
        ) -> float:
    """
    Return the fraction of incorrect predictions that are put on a parallel
        runway (i.e., numeric portion of the prediction matches actual)

    Parameters
    ----------
    y_true : pd.Series
        True runways used
    y_pred : pd.Series
        Predicted runways

    Returns
    -------
    res : float
        Fraction of flights that are both:
            1. incorrectly predicted
            2. predicted to use a runway with the same numeric component as the
                actual value
    """

    dir_y = (
        y_true
        .str
        .extract(r"(\d{1,2})+", expand=False)
        .astype(int)
        .reset_index(drop=True)
    )
    
    dir_y_pred = (
        y_pred
        .str
        .extract(r"(\d{1,2})+", expand=False)
        .astype(int)
        .reset_index(drop=True)
    )

    misclass = (y_true.values != y_pred.values)
    parallel = (
        (np.abs(dir_y - dir_y_pred) == 0)
        | (np.abs(dir_y - dir_y_pred) == 1)
        | (np.abs(dir_y - dir_y_pred) == 35)
        )

    res = (misclass & parallel).sum() / y_true.shape[0]

    return res

def _confusion_matrix(
        y: pd.Series,
        y_pred: pd.Series,
        labels: List[str],
        airport_name: str,
        naming_token: str,
        ) -> List[str]:

    y = y.astype(str)
    y_pred = y_pred.astype(str)

    cm0 = pd.DataFrame(
        sklearn.metrics.confusion_matrix(
            y,
            y_pred,
            labels=labels,
            ),
        columns=labels,
        index=labels,
        )

    cm = np.log10(cm0).replace(-1*np.inf, 0)

    num_rwys = len(cm.index)

    fig = plt.figure(figsize=(max(1.5*num_rwys, 8), max(1.5*num_rwys, 8)))
    ax = fig.gca()
    im = ax.imshow(
        cm,
        cmap="plasma",
        )

    ax.set_xticks(np.arange(len(cm.index)))
    ax.set_yticks(np.arange(len(cm.index)))
    ax.set_xticklabels(cm.index, fontsize=24)
    ax.set_yticklabels(cm.index, fontsize=24)

    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(
        np.arange(len(cm.index)+1)-.5,
        minor=True,
        )
    ax.set_yticks(
        np.arange(len(cm.index)+1)-.5,
        minor=True,
        )
    ax.grid(
        which="minor",
        axis="both",
        color="w",
        linestyle='-',
        linewidth=3,
        )
    ax.tick_params(
        which="minor",
        bottom=False,
        left=False,
        )
    ax.tick_params(
        labeltop=True,
        labelright=True,
        )

    textcolors = ["white", "black"]
    threshold = im.norm(cm.max().max())*3./4.

    kw = dict()
    for i in range(len(cm.index)):
        for j in range(len(cm.index)):
            kw.update(color=textcolors[int(im.norm(cm.iloc[j, i]) > threshold)])
            im.axes.text(
                i,
                j,
                "{:.1%}".format(cm0.iloc[j, i]/cm0.sum().sum()),
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=16,
                **kw,
                )

    ax.set_xlabel("Predicted Runway", fontsize=24)
    ax.set_ylabel("True Runway", fontsize=24)
    ax.set_title(f"Confusion Matrix for {airport_name}", fontsize=24)


    fig.tight_layout()

    filename_pattern = f"cm_{airport_name}_{naming_token}"

    filenames = list()

    fig_filename = f"{filename_pattern}.png".lower()
    plt.savefig(
        fname=fig_filename,
        dpi=300,
        )
    filenames.append(fig_filename)

    dat_filename = f"{filename_pattern}.csv".lower()
    cm0.to_csv(
        dat_filename,
        )
    filenames.append(dat_filename)

    return filenames

def _accuracy_over_time(
        X: pd.DataFrame,
        y_pred: pd.Series,
        y_default: Any,
        airport_name: str,
        naming_token: str,
        bin_size: int = 5*60,
        max_lookahead: int = 3*60*60,
        ) -> List[str]:

    df = (
        X
        .join(y_pred)
        )
    # default response can return an array, but fillna can take scalar or indexed list like Serie
    # should work unless pred is not scalar quantity and df has only 1 element
    df["pred"] = df["pred"].fillna(pd.Series(y_default(X), index=df.index)) 
    df["correct"] = (df["pred"] == df["actual"])
    df["bin"] = bin_size*np.floor(df["lookahead"]/bin_size)

    plot_dat = (
        df
        .loc[
            df.index.isin(y_pred.index)
            & df["lookahead"].between(0,max_lookahead),
            :]
        .groupby(["bin"])["correct"]
        .agg(["count", "mean"])
        .reset_index()
        )
    plot_dat_base = (
        df
        .loc[
            df["lookahead"].between(0,max_lookahead),
            :]
        .groupby(["bin"])["correct"]
        .agg(["count", "mean"])
        .reset_index()
        )

    def _make_fig(
            airport_name: str,
            naming_token: str,
            plot_dat: pd.DataFrame,
            plot_dat_base: pd.DataFrame=None,
            max_lookahead: int = 3*60*60,
            ) -> str:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.gca()
        ax.plot(
                plot_dat["bin"],
                plot_dat["mean"],
                label="ML model alone",
                )

        ax.set_xticks(np.arange(0,max_lookahead+1,1800))
        ax.set_xticklabels((np.arange(0,max_lookahead+1,1800)/60).astype(int), fontsize=24)
        ax.set_xlabel("Minutes before expected landing", fontsize=24)
        ax.invert_xaxis()

        ax.set_ylim(0.5, 1.0)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        plt.yticks(fontsize=24)
        ax.set_ylabel("Correct predictions", fontsize=24)

        ax.set_title(f"Aggregated Model Accuracy\n{airport_name}, {naming_token} Data", fontsize=24)

        if plot_dat_base is not None:
            ax.plot(
                plot_dat_base["bin"],
                plot_dat_base["mean"],
                label="ML model &\ndefault values for bad data",
                )
            ax.legend(
                loc=3,
                edgecolor="black",
                )

        fig_filename = f"ts_{airport_name}_{naming_token}_{'ml' if plot_dat_base is None else 'all'}.png".lower()
        plt.savefig(
            fname=fig_filename,
            dpi=300,
            )

        return fig_filename

    fn_ml = _make_fig(
        airport_name,
        naming_token,
        plot_dat,
        )
    fn_all = _make_fig(
        airport_name,
        naming_token,
        plot_dat,
        plot_dat_base,
        )

    return [fn_ml, fn_all]

def _distribution_of_accuracy(
        X: pd.DataFrame,
        y_pred: pd.Series,
        airport_name: str,
        naming_token: str,
        max_lookahead: int = 3*60*60,
        ) -> str:

    df = (
        X
        .join(
            y_pred,
            how="inner",
            )
        )
    df["correct"] = (df["pred"] == df["actual"])

    pct_correct_dat = (
        df
        .loc[
            df["lookahead"].between(0,max_lookahead),
            :]
        .groupby(["gufi"])["correct"]
        .agg(["count", "mean"])
        .reset_index()
        )

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    vals, bins, _ = ax.hist(
        pct_correct_dat["mean"],
        bins=np.linspace(0,1,21),
        )

    ax.set_xticks(np.linspace(0,1,11))
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Percentage of individual predictions\ncorrect for this flight")

    ax.set_ylabel("Number of flights")

    ax.text(
            0.02,
            0.98,
            "Only includes predictions\nfrom the ML model",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox={
                "facecolor":"white",
                "edgecolor":"black",
                },
            )

    ax.set_title(f"Are the metrics skewed by a few \"bad flights\"?\n{airport_name}, {naming_token} Data")

    fig_filename = f"doa_{airport_name}_{naming_token}.png".lower()
    plt.savefig(
        fname=fig_filename,
        dpi=300,
        )

    return fig_filename

def _runways_out_of_config(
    X: pd.DataFrame,
    y: pd.Series,
    airport_name: str,
    naming_token: str,
    operation_type: str="departure_runway",
) -> Tuple[List[str], float]:

    filenames = list()

    patterns = {
        "arrival_runway":'A_(?P<arr>.*)',
        "departure_runway":'D_(?P<dep>.*)_A',
    }

    dat = X.copy()
    dat["rwy"] = y.values

    res = (
        dat
        .groupby(
            [
                "airport_configuration_name",
                "rwy",
            ],
            as_index=False,
        )
        .agg(
            num=pd.NamedAgg("gufi", "count")
        )
    )
    res["num"] = res["num"].astype(int)

    res["op_config"] = (
        res["airport_configuration_name"]
        .str
        .extract(patterns[operation_type])
    )

    res["in_config"] = (
        res
        .apply(
            lambda row: row["rwy"] in row["airport_configuration_name"],
            axis=1,
        )
    )

    config_rwy_pairs = (
        res
        .pivot(
            index="airport_configuration_name",
            columns="rwy",
            values="num",
        )
        .reset_index()
        .fillna(0)
    )

    frac_in_config = (
        res
        .groupby(
            [
                "airport_configuration_name",
                "in_config"
            ],
            as_index=False,
        )
        .agg(
            sum=pd.NamedAgg("num", "sum")
        )
        .pivot(
            index="airport_configuration_name",
            columns="in_config",
            values="sum",
        )
        .reset_index()
        .fillna(0)
    )
    frac_in_config = frac_in_config.rename(columns={
        True:"num_in_config",
        False:"num_not_in_config",
    })

    for c in ["num_in_config", "num_not_in_config"]:
        if c not in frac_in_config.columns:
            frac_in_config[c] = 0.0

    frac_in_config["count"] = frac_in_config["num_in_config"] + frac_in_config["num_not_in_config"]
    frac_in_config["fraction_in_config"] = frac_in_config["num_in_config"] / frac_in_config["count"]

    frac_in_config_overall = frac_in_config["num_in_config"].sum() / frac_in_config["count"].sum()

    filename_pairs = f"config_rwy_pairs_{airport_name}_{naming_token}.html".lower()
    config_rwy_pairs.to_html(
        filename_pairs,
        index=False,
    )
    filenames.append(filename_pairs)

    filename_frac = f"frac_in_config_{airport_name}_{naming_token}.html".lower()
    frac_in_config.to_html(
        filename_frac,
        index=False,
    )
    filenames.append(filename_frac)

    return filenames, frac_in_config_overall
