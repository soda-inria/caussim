from typing import Tuple
import numpy as np
from sklearn.utils import check_random_state
from caussim.estimation.scores import heterogeneity_score

from caussim.pdistances.divergences import jensen_shannon_divergence
from caussim.pdistances.mmd import total_variation_distance
from caussim.pdistances.effect_size import mean_causal_effect

class CausalDf(object):
    def __init__(
        self,
        df,
        dataset_name: str = None,
        overlap_parameter: str = None,
        random_state: int=None,
        dgp: int=None
    ) -> None:
        self.df = df
        self.df.reset_index(drop=True)
        self.x_cols = [col for col in df.columns if col.lower().startswith("x")]
        if dataset_name is not None:
            self.dataset_name = dataset_name
        else:
            self.dataset_name = ""
        if overlap_parameter is not None:
            self.overlap_parameter = overlap_parameter
        else:
            self.overlap_parameter = ""
        if random_state is not None:
            self.random_state = random_state
        else:
            self.random_state = np.nan
        if dgp is not None:
            self.dgp = dgp
        else:   
            self.dgp = np.nan
            
    def get_aX(self) -> np.ndarray:
        return self.df[["a", *self.x_cols]].values

    def get_X(self) -> np.ndarray:
        return self.df[self.x_cols].values

    def get_a(self) -> np.ndarray:
        return self.df["a"].values

    def get_y(self) -> np.ndarray:
        return self.df["y"].values

    def get_aX_y(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.get_aX(), self.df["y"].values

    def estimate_oracles(self, covariate_mean_balance=False):
        cate = self.df["mu_1"] - self.df["mu_0"]
        cate_obs = self.df["y_1"] - self.df["y_0"]
        treatment_heterogeneity = np.std(cate)
        treatment_heterogeneity_obs = np.std(cate_obs)
        heterogeneity_score_ = heterogeneity_score(
            y_0=self.df["mu_0"],
            y_1=self.df["mu_1"],
            ps=self.df["e"],
            n_bins=10,
        )        
        effect_ratio = mean_causal_effect(self.df["mu_1"], self.df["mu_0"])
        ate = cate.mean()
        mask_treated = self.df["a"] == 1
        att = (
            self.df.loc[mask_treated, "y_1"] - self.df.loc[mask_treated, "y_0"]
        ).mean()
        hat_ate_diff = np.mean(self.df[self.df["a"] == 1]["y"].values) - np.mean(
            self.df[self.df["a"] == 0]["y"].values
        )
        # n_treated = self.df["a"].sum()
        # n_untreated = self.df["a"].shape[0] - n_treated
        oracle_ipw = (
            (self.df["a"] * self.df["y"] / self.df["e"]).sum()
            - ((1 - self.df["a"]) * self.df["y"] / (1 - self.df["e"])).sum()
        ) / self.df["a"].shape[0]

        # oracle covariate balance
        if covariate_mean_balance:
            mean_balance_treated = (
                self.df.loc[mask_treated, self.x_cols] / self.df["e"]
            ).mean()
            mean_balance_untreated = (
                self.df.loc[~mask_treated, self.x_cols] / (1 - self.df["e"])
            ).mean()
        else:
            mean_balance_treated = None
            mean_balance_untreated = None

        results = {
            "cate": cate,
            "treatment_heterogeneity": treatment_heterogeneity,
            "treatment_heterogeneity_obs": treatment_heterogeneity_obs,
            "heterogeneity_score": heterogeneity_score_,
            "ate": ate,
            "att": att,
            "hat_ate_diff": hat_ate_diff,
            "oracle_ipw": oracle_ipw,
            "oracle_mean_balance_treated_X": mean_balance_treated,
            "oracle_mean_balance_untreated_X": mean_balance_untreated,
            "effect_ratio": effect_ratio,
        }
        return results
    
    def bootstrap(self, seed:int=None):
        """Create a bootstrap (with replacement) version of the original dataset.

        Parameters
        ----------
        seed : _type_
            _description_
        """
        gen = check_random_state(seed)
        bs_indices = gen.choice(len(self.df), size= len(self.df), replace=True) 
        self.df = self.df.iloc[bs_indices]
        return self
        
    def describe(self, prefix=""):
        """Describe a causal dataset
        Args:
            prefix (str): Prefix to use for the keys of the returned dataset

        Returns:
            Dict: [description]
        """
        dataset_info = {}
        dataset_info["a_mean"] = self.df["a"].mean()
        dataset_info["e_mean"] = self.df["e"].mean()
        dataset_info["N"] = self.df.shape[0]
        dataset_info["D"] = len(
            [colname for colname in self.df.columns if colname.lower().startswith("x")]
        )

        dataset_info["e_max"] = self.df["e"].max()
        dataset_info["e_min"] = self.df["e"].min()
        # overlap measures with ps distances
        dataset_info["d_js"] = jensen_shannon_divergence(self.df["e"], 1 - self.df["e"])
        dataset_info["d_tv"] = total_variation_distance(self.df["e"], 1 - self.df["e"])
        dataset_info["d_normalized_tv"] = total_variation_distance(
            self.df["e"] / dataset_info["a_mean"],
            (1 - self.df["e"]) / (1 - dataset_info["a_mean"]),
        )

        dataset_info["eps_B_0"] = np.sqrt(
            np.mean((self.df["y_0"] - self.df["mu_0"]) ** 2)
        )
        dataset_info["eps_B_1"] = np.sqrt(
            np.mean((self.df["y_1"] - self.df["mu_1"])) ** 2
        )
        dataset_info["snr_1"] = dataset_info["eps_B_1"] / np.abs(self.df["mu_1"])
        dataset_info["snr_0"] = dataset_info["eps_B_0"] / np.abs(self.df["mu_0"])
        dataset_info["dataset_name"] = self.dataset_name
        dataset_info["overlap_parameter"] = self.overlap_parameter
        ds_info_prefixed = {f"{prefix}{k}": v for k, v in dataset_info.items()}
        return ds_info_prefixed

    def __str__(self):
        return f"{self.dataset_name}__overlap_{self.overlap_parameter}__rs_{self.random_state}__dgp_{self.dgp}"