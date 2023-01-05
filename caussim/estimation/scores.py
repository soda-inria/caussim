import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
    f1_score,
)
from sklearn.utils import check_array, check_consistent_length

# TODO: insert into estimation
def ipw_score(y, a, hat_mu_0, hat_mu_1, hat_e, trimming=None):
    if trimming is not None:
        clipped_hat_e = np.clip(hat_e, trimming, 1 - trimming)
    else:
        clipped_hat_e = hat_e
    ipw_weights = a / clipped_hat_e + (1 - a) / (1 - clipped_hat_e)
    hat_y = hat_mu_1 * a + hat_mu_0 * (1 - a)

    return np.sum(((y - hat_y) ** 2) * ipw_weights) / len(y)


def ipw_R_risk(y, a, hat_mu_0, hat_mu_1, hat_e, hat_m, trimming=None):
    if trimming is not None:
        clipped_hat_e = np.clip(hat_e, trimming, 1 - trimming)
    else:
        clipped_hat_e = hat_e
    ipw_weights = a / clipped_hat_e + (1 - a) / (1 - clipped_hat_e)
    hat_tau = hat_mu_1 - hat_mu_0

    return np.sum((((y - hat_m) - (a - hat_e) * (hat_tau)) ** 2) * ipw_weights) / len(y)


def ipw_R_risk_oracle(y, a, hat_mu_0, hat_mu_1, e, mu_1, mu_0):
    m = mu_0 * (1 - e) + mu_1 * e
    return ipw_R_risk(y=y, a=a, hat_mu_0=hat_mu_0, hat_mu_1=hat_mu_1, hat_e=e, hat_m=m)


# ### metrics Utils ### #


def print_metrics_regression(y_true, predictions, verbose=1, elog=None):

    mad = mean_absolute_error(y_true, predictions)
    mse = mean_squared_error(y_true, predictions)
    mape = mean_absolute_percentage_error(y_true, predictions)
    r2 = r2_score(y_true, predictions)

    if verbose:
        print("Mean absolute deviation (MAD) = {}".format(mad))
        print("Mean squared error (MSE) = {}".format(mse))
        print("Mean absolute percentage error (MAPE) = {}".format(mape))
        print("R^2 Score = {}".format(r2))

    return {
        "mad": mad,
        "mse": mse,
        "mape": mape,
        "r2": r2,
    }


def print_metrics_binary(y_true, prediction_probs, verbose=1, elog=None):
    if verbose:
        print("==> Binary scores:")
    prediction_probs = np.array(prediction_probs)
    prediction_probs = np.transpose(
        np.append([1 - prediction_probs], [prediction_probs], axis=0)
    )
    predictions = prediction_probs.argmax(axis=1)
    cf = confusion_matrix(y_true, predictions, labels=range(2))
    if elog is not None:
        elog.print("Confusion matrix:")
        elog.print(cf)
    elif verbose:
        print("Confusion matrix:")
        print(cf)
    cf = cf.astype(np.float32)

    acc = (cf[0][0] + cf[1][1]) / np.sum(cf)
    prec0 = cf[0][0] / (cf[0][0] + cf[1][0])
    prec1 = cf[1][1] / (cf[1][1] + cf[0][1])
    rec0 = cf[0][0] / (cf[0][0] + cf[0][1])
    rec1 = cf[1][1] / (cf[1][1] + cf[1][0])

    auroc = roc_auc_score(y_true, prediction_probs[:, 1])
    (precisions, recalls, thresholds) = precision_recall_curve(
        y_true, prediction_probs[:, 1]
    )
    auprc = average_precision_score(y_true, prediction_probs[:, 1])
    f1macro = f1_score(y_true, predictions, average="macro")
    # calibration
    brier = brier_score_loss(y_true, prediction_probs[:, 1])
    bss = brier_skill_score(y_true, prediction_probs[:, 1])
    results = {
        "Accuracy": acc,
        "Precision class 0": prec0,
        "Precision class 1": prec1,
        "Recall class 0": rec0,
        "Recall class 1": rec1,
        "Area Under the Receiver Operating Characteristic curve (AUROC)": auroc,
        "Area Under the Precision Recall curve (AUPRC)": auprc,
        "F1 score (macro averaged)": f1macro,
        "Brier score": brier,
        "Brier Skill Score": bss,
    }
    if verbose:
        for key in results:
            print("{} = {}".format(key, results[key]))

    return {
        "acc": acc,
        "prec0": prec0,
        "prec1": prec1,
        "rec0": rec0,
        "rec1": rec1,
        "auroc": auroc,
        "auprc": auprc,
        "f1macro": f1macro,
        "brier": brier,
        "bss": bss,
    }


def brier_skill_score(y_true, y_prob):
    """
    Brier skill score : https://en.wikipedia.org/wiki/Brier_score
    Args:
        y_true ([type]): [description]
        y_prob ([type]): [description]
    """
    brier = brier_score_loss(y_true, y_prob)
    dummy_brier = brier_score_loss(y_true, np.repeat(y_true.mean(), len(y_true)))
    #
    bss = 1 - brier / dummy_brier
    return bss


def get_treatment_metrics(y_true, y_prob):
    """Only for binary treatment

    Args:
        y_true (_type_): _description_
        y_prob (_type_): _description_

    Returns:
        _type_: _description_
    """
    assert y_prob.ndim == 1, "y_prob should be a 1D array, with the score of the positive class."
    return {
        "bss": brier_skill_score(y_true, y_prob),
        "bs": brier_score_loss(y_true, y_prob),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def heterogeneity_score(
    y_0: np.array,
    y_1: np.array,
    ps: np.array,
    n_bins:int=10,
    strategy: str="uniform") -> float:
    """ Compute the variance of the delta between y_0 and y_1 by propensity bin average over the bins.
    Let the ITE, :math:`\tau_i = \mu_{1, i} - \mu_{0, i}` and the mean of the ITE for one bin, :math:`\tau_b = \frac{1}{|b|} \sum_{i \in b}^n \tau_i`.
    The heterogeneity proxy is:
    .. math::
        \mathcal{H}^2 = \frac{1}{nbins} \sum_{b} \frac{1}{|b| - 1} \sum_{i \in b} (\tau_i - \tau_b)^2
    
    Parameters
    ----------
    y_0 : np.array
        _description_
    y_1 : np.array
        _description_
    ps : np.array
        _description_
    n_bins : int, optional
        _description_, by default 10
    strategy : str, optional
        _description_, by default "uniform"
        
    Returns
    -------
    float
        Treatment effect heterogeneity proxy
    """
    check_consistent_length(ps, y_0, y_1)    
    bins_df = pd.DataFrame({"ps": ps, "y_0": y_0, "y_1": y_1})
    if strategy == "quantile":  # Determine bin edges by distribution of data
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(ps, quantiles * 100)
        bins[-1] = bins[-1] + 1e-8
    elif strategy == "uniform":
        bins = np.linspace(0.0, 1.0 + 1e-8, n_bins + 1)
    else:
        raise ValueError(
            "Invalid entry to 'strategy' input. Strategy "
            "must be either 'quantile' or 'uniform'."
        )
    
    bins_df["ps_bin"] = pd.cut(
        x=bins_df["ps"],
        bins=bins, 
        labels=bins[1:])
    bins_df["tau"] = bins_df["y_1"] - bins_df["y_0"]
    var_by_bin = bins_df.groupby("ps_bin").agg(
        **{
            "std_tau": pd.NamedAgg("tau", lambda x: np.nanvar(x, ddof=1)),
        }
    )
    n_by_bins = bins_df.groupby("ps_bin").agg(
        **{
            "count_bin": pd.NamedAgg("tau", len),
        }
    )
    n=bins_df.shape[0]
    return np.multiply(var_by_bin.std_tau, n_by_bins.count_bin/n).sum()
        