from typing import Dict

import numpy as np
from numpy.random.mtrand import seed
import pandas as pd
from joblib import Memory
from caussim.data.causal_df import CausalDf
from caussim.pdistances.mmd import mmd_rbf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
from tqdm import tqdm

from caussim.config import DIR2CACHE
from caussim.utils import cantor_pairing

memory = Memory(DIR2CACHE, verbose=0)


def sample_wager(n, p=5, setup: str = "A", y_noise=0, random_state=0, hte:float = 0) -> CausalDf:
    """
    Simulation setting inspired from https://arxiv.org/pdf/1712.04912.pdf
    https://github.com/xnie/rlearner/blob/master/experiments_for_paper/run_simu.R

    """
    rng = check_random_state(random_state)
    if setup == "A":
        x = rng.uniform(low=0, high=1, size=(n, p))
        y_0 = (
            np.sin(np.pi * x[:, 0] * x[:, 1])
            + 2 * (x[:, 2] - 0.5) ** 2
            + x[:, 3]
            + 0.5 * x[:, 4]
        )
        eta = 0.1
        e = np.maximum(eta, np.minimum(np.sin(np.pi * x[:, 0] * x[:, 1]), 1 - eta))
        tau = (x[:, 0] + x[:, 1]+ hte*x[:, 2]) / 2
    elif setup == "B":
        x = rng.normal(size=(n, p))
        y_0 = np.maximum(0, x[:, 0] + x[:, 1], x[:, 2]) + np.maximum(
            0, x[:, 3] + x[:, 4]
        )
        e = 0.5
        tau = x[:, 0] + np.log(1 + np.exp(x[:, 1]))
    elif setup == "C":
        x = rng.normal(size=(n, p))
        y_0 = 2 * np.log(1 + np.exp(x[:, 0] + x[:, 1] + x[:, 2]))
        e = 1 / (1 + np.exp(x[:, 1] + x[:, 2]))
        tau = np.ones(n)
    elif setup == "D":
        x = rng.normal(size=(n, p))
        y_0 = (
            np.maximum(x[:, 0] + x[:, 1] + x[:, 2], 0)
            + np.maximum(x[:, 3] + x[:, 4], 0)
        ) / 2
        e = 1 / (1 + np.exp(-x[:, 0]) + np.exp(-x[:, 1]))
        tau = np.maximum(x[:, 0] + x[:, 1] + x[:, 2], 0) - np.maximum(
            x[:, 3] + x[:, 4], 0
        )
    else:
        raise (NameError("Invalid setup option"))
    a = rng.binomial(1, e, size=n)
    y_1 = y_0 + tau
    # observed response
    y = a * y_1 + (1 - a) * y_0
    y = y + y_noise * rng.normal(size=(n,))
    # formatting into a dataset
    targets = pd.DataFrame(
        np.vstack([y, y_1, y_0, y_1, y_0, a, e]).transpose(),
        columns=["y", "y_1", "y_0", "mu_1", "mu_0", "a", "e"],
    )
    covariates = pd.DataFrame(x, columns=[f"X_{j}" for j in range(x.shape[1])])

    full_X = pd.concat((targets, covariates), axis=1)
    return CausalDf(full_X, dataset_name="wager", dgp=setup, random_state=random_state)


# ### Toy example simulation with sigmoids ### #


def sigmoid(x, alpha=1):
    return 1 / (1 + np.exp(-alpha * x))


def sample_sigmoids(
    n=100,
    alpha_treated=2,
    scale_treated=0.9,
    alpha_untreated=1,
    scale_untreated=0.6,
    untreated_offset=0.1,
    treated_offset=0,
    alpha_intervention=0.5,
    ps_offset=0.2,
    max_overlap=0.95,
    xlim=(0, 20),
    space_radius=5,
    x_noise=0,
    y_noise=0,
    random_state=0,
):
    rng = check_random_state(random_state)
    dim = 1

    # linear transformation of Z
    c = 1
    offset = 0
    Z = rng.uniform(xlim[0], xlim[1], size=(n, dim))
    # observed proxy
    X = c * Z + offset + x_noise * rng.normal(size=(n, dim))
    tmin, tmax = -space_radius, space_radius
    # send to radius
    R = (tmax - tmin) * X / (xlim[1] - xlim[0]) + tmin
    # intervention
    ps = np.clip(
        alpha_intervention * sigmoid(R, 1) + ps_offset,
        1 - max_overlap * np.ones_like(R),
        max_overlap,
    )
    a = rng.binomial(size=(n, 1), n=1, p=ps)
    # oracle response surfaces
    mu_0 = scale_untreated * sigmoid(R, alpha_untreated) + untreated_offset
    mu_1 = scale_treated * sigmoid(R, alpha_treated) + treated_offset
    y_1 = mu_1 + y_noise * rng.normal(size=(n, 1))
    y_0 = mu_0 + y_noise * rng.normal(size=(n, 1))
    # observed response
    y = a * y_1 + (1 - a) * y_0
    y[y>1] = 1
    y[y<0] = 0
    # formatting into a dataset
    cols_Z = [f"z_{j}" for j in range(Z.shape[1])]
    cols_X = [f"x_{j}" for j in range(X.shape[1])]
    full_observations = pd.DataFrame(
        np.hstack([y, y_1, y_0, mu_1, mu_0, a, ps, X, Z]),
        columns=["y", "y_1", "y_0", "mu_1", "mu_0", "a", "e", *cols_X, *cols_Z],
    )
    return full_observations


# ### Causal Simulator ### #
# Original structure comes from  https://github.com/IBM/causallib
# We replaced the generation of covariates by a mixture of gaussian and added featurization-linear links (Nystroem, Splines)

TREATMENT = "treatment"
TREATMENT_ASSIGNMENT = "treatment_assignment"
BASELINE = "baseline"
TREATMENT_EFFECT = "treatment_effect"
COVARIATE = "covariate"


class CausalSimulator(object):
    TREATMENT_METHODS = {
        "random": lambda x, p, params, random_state: CausalSimulator._treatment_random(
            x, p, random_state
        ),
        "logistic": lambda x, p, params, random_state: CausalSimulator._treatment_logistic_dichotomous(
            x, p, params=params, random_state=random_state
        ),
    }
    # G for general - applicable to all types of variables
    G_LINKING_METHODS = {
        "affine": lambda params: AffineTransformer(**params),
        "linear": lambda params: AffineTransformer(**params, intercept=False),
        "rbf": lambda params: RbfLink(**params),
        "spline": lambda params: SplineLink(**params),
        "nystroem": lambda params: NystroemLink(**params),
    }

    def __init__(
        self,
        baseline_link: Dict = None,
        effect_link: Dict = None,
        treatment_link: Dict = None,
        treatment_assignment: Dict = None,
        dim: int = 2,
        treatment_ratio=0.1,
        effect_size: float = 0.5,
        outcome_noise: float = 0,
        alignment: float = 1,
        random_seed: int = 0,
        n_samples_init: int = 1000,
        X_given=None,
        clip: float = 0
    ) -> None:
        super().__init__()
        self.treatment_ratio = treatment_ratio
        self.prob_categories = pd.Series(
            [1 - treatment_ratio, treatment_ratio], index=[0, 1]
        )
        var_types = pd.Series(
            [COVARIATE] * dim + [TREATMENT, TREATMENT_EFFECT, BASELINE]
        )
        self.var_names = var_types.index.to_series().reset_index(drop=True)
        self.var_types = var_types
        self.treatment_index = var_types[var_types == TREATMENT].index[0]
        self.baseline_index = var_types[var_types == BASELINE].index[0]
        self.effect_index = var_types[var_types == TREATMENT_EFFECT].index[0]
        self.covariate_indices = var_types[var_types == COVARIATE].index
        self.params = {
            TREATMENT: treatment_link.get("params", {}),
            TREATMENT_ASSIGNMENT: treatment_assignment.get("params", {}),
            BASELINE: baseline_link.get("params", {}),
            TREATMENT_EFFECT: effect_link.get("params", {}),
        }

        self.dim = dim
        self.treatment_link_type = treatment_link.get("type", "linear")
        self.treatment_assignment_type = treatment_assignment.get("type", "joint")
        self.baseline_link_type = baseline_link.get("type", "spline")
        self.effect_link_type = effect_link.get("type", "spline")
        self.outcome_noise = outcome_noise
        self.effect_size = effect_size
        self.alignment = alignment
        self.rs_rotation = random_seed
        self.rs_gaussian = random_seed
        self.n_samples_init = n_samples_init
        if ((clip >0.5) or (clip<0)):
            raise ValueError("Clip must be between 0 and 0.5")
        self.clip = clip
        
        seed(self.rs_gaussian)
        generator = check_random_state(self.rs_gaussian)
        if self.treatment_assignment_type != "joint":
            if X_given is None:
                X_given = pd.DataFrame(
                    generator.random.randn(self.n_samples_init, self.dim),
                    columns=self.covariate_indices,
                )
            else:
                self.n_samples_init = X_given.shape[0]
                Warning(
                    "X_given not null, so force number of sample to X_given sample size"
                )
        # Treatment can be either post covariate simulation or joint with covariate simulation
        if self.treatment_assignment_type == "joint":
            X_given, propensity, treatment = self._generate_joint_treatment_covariates(
                num_samples=self.n_samples_init,
                dim=self.dim,
                treatment_ratio=self.treatment_ratio,
                random_state_rotation=self.rs_rotation,
                random_state_gaussian=self.rs_gaussian,
                **self.params[TREATMENT_ASSIGNMENT],
            )
        else:
            var_index = self.treatment_index
            var_params = self.params.get(self.var_types[var_index], {})
            X_parents_ix = self.covariate_indices
            X_parents = X_given.loc[:, X_parents_ix]
            # Deprecated, but still supported
            self.treatment_pipeline = self.fit_g_link_pipeline(
                X_parents=X_parents,
                link_type=self.treatment_link_type,
                params=var_params,
            )
            propensity, treatment = self._generate_treatment(
                X_parents=X_parents,
                prob_category=self.prob_categories,
                treatment_pipeline=self.treatment_pipeline,
                treatment_assignment=self.treatment_assignment_type,
            )

        # define variables types
        for var_index in [self.baseline_index, self.effect_index]:
            X_parents_ix = self.covariate_indices
            X_parents = X_given.loc[:, X_parents_ix]
            var_params = self.params.get(self.var_types[var_index], {})
            if self.var_types[var_index] == BASELINE:
                self.baseline_pipeline = self.fit_g_link_pipeline(
                    X_parents=X_parents,
                    link_type=self.baseline_link_type,
                    params=var_params,
                )
            elif self.var_types[var_index] == TREATMENT_EFFECT:
                self.effect_pipeline = self.fit_g_link_pipeline(
                    X_parents=X_parents,
                    link_type=self.effect_link_type,
                    params=var_params,
                )
            else:
                raise ValueError("unkown variable index")

        X_given.columns = [f"x_{i}" for i in self.covariate_indices]
        self.x_cols = X_given.columns
        return None

    def sample(
        self,
        num_samples: int = 1000,
    ):
        generator = check_random_state(self.rs_gaussian)

        if self.treatment_assignment_type != "joint":
            X_given = pd.DataFrame(
                generator.randn(num_samples, self.dim),
                columns=self.covariate_indices,
            )
        # Treatment can be either post covariate simulation or joint with covariate simulation
        if self.treatment_assignment_type == "joint":
            X_given, propensity, treatment = self._generate_joint_treatment_covariates(
                num_samples=num_samples,
                dim=self.dim,
                treatment_ratio=self.treatment_ratio,
                random_state_rotation=self.rs_rotation,
                **self.params[TREATMENT_ASSIGNMENT],
                random_state_gaussian=self.rs_gaussian,
            )
        else:
            raise NotImplementedError

        mu_1 = self.mu(X_given, np.ones(num_samples))
        mu_0 = self.mu(X_given, np.zeros(num_samples))
        # homoscedastic noise
        y_0, cov_std, noise = self._noise_col(
            pd.Series(mu_0), snr=(1 - self.outcome_noise)
        )
        y_1, _, _ = self._noise_col(
            pd.Series(mu_1), snr=(1 - self.outcome_noise), noise=noise
        )
        y = y_1 * treatment + y_0 * (1 - treatment)
        # assembling dataset
        outcomes = pd.DataFrame(
            {
                "mu_0": mu_0,
                "mu_1": mu_1,
                "y_0": y_0,
                "y_1": y_1,
                "y": y,
                "e": propensity,
                "a": treatment,
            }
        )
        X_given.columns = [f"x_{i}" for i in self.covariate_indices]
        df = pd.concat((outcomes, X_given), axis=1) 
        if self.clip is not None:
            df = df.loc[(df["e"]>=self.clip) & (df["e"]<=1-self.clip)]
        causal_df = CausalDf(
            df,
            dataset_name="caussim",
            overlap_parameter=self.overlap,
            random_state=cantor_pairing(self.rs_rotation, self.rs_gaussian),
        )
        return causal_df

    def _generate_joint_treatment_covariates(
        self,
        num_samples: int,
        dim: int = 2,
        treatment_ratio: float = 0.1,
        overlap: float = 1,
        sigma: np.array = None,
        random_state_rotation=None,
        random_state_gaussian=None,
    ):
        # TODO: should separate the init/fit behavior of choosing a rotation from the generation of samples depending on gaussian seed.
        gaussian_generator = check_random_state(random_state_gaussian)
        if sigma is None:
            sigma = np.array([2, 4])
        self.overlap = overlap
        self._center = np.zeros(len(sigma))
        self._center[0] = overlap
        self._sigma = np.diag(sigma)
        self._rotation = self._generate_rotation(
            dim, random_state=random_state_rotation
        )
        center_rotated_1 = np.atleast_1d(self._rotation.dot(self._center))
        center_rotated_0 = np.atleast_1d(self._rotation.dot(-self._center))
        sigma_rotated = np.atleast_2d(self._rotation.dot(self._sigma).dot(self._rotation.T))
    
        X_control = gaussian_generator.multivariate_normal(
            mean=center_rotated_0,
            cov=sigma_rotated,
            size=int((1 - treatment_ratio) * num_samples),
        )
        X_treated = gaussian_generator.multivariate_normal(
            mean=center_rotated_1,
            cov=sigma_rotated,
            size=num_samples - X_control.shape[0],
        )

        X = np.concatenate((X_control, X_treated), axis=0)
        treatment = np.concatenate(
            (
                np.zeros(X_control.shape[0]),
                np.ones(X_treated.shape[0]),
            ),
            axis=0,
        )

        # compute propensity score
        p_x_a_1 = self._gaussian_density(X, center_rotated_1, sigma_rotated)
        p_x_a_0 = self._gaussian_density(X, center_rotated_0, sigma_rotated)
        propensity_density = (
            treatment_ratio
            * p_x_a_1
            / ((1 - treatment_ratio) * p_x_a_0 + treatment_ratio * p_x_a_1)
        )

        return pd.DataFrame(X), propensity_density, treatment

    def propensity_density(self, x):
        center_rotated_1 = self._rotation.dot(self._center)
        center_rotated_0 = self._rotation.dot(-self._center)
        sigma_rotated = self._rotation.dot(self._sigma).dot(self._rotation.T)
        p_x_a_1 = self._gaussian_density(x, center_rotated_1, sigma_rotated)
        p_x_a_0 = self._gaussian_density(x, center_rotated_0, sigma_rotated)

        return (
            self.treatment_ratio
            * p_x_a_1
            / ((1 - self.treatment_ratio) * p_x_a_0 + self.treatment_ratio * p_x_a_1)
        )

    @staticmethod
    def _gaussian_density(x, mu, sigma):
        dim = x.shape[1]
        return (
            (2 * np.pi) ** (-dim / 2)
            * np.linalg.det(sigma) ** (-dim / 2)
            * np.exp(-0.5 * ((x - mu).dot(np.linalg.inv(sigma)) * (x - mu)).sum(axis=1))
        )

    def _generate_treatment(
        self,
        X_parents,
        prob_category,
        treatment_pipeline,
        treatment_assignment,
    ):

        x_continuous = pd.Series(
            treatment_pipeline.transform(X_parents), index=X_parents.index
        )

        generation_method = self.TREATMENT_METHODS.get(treatment_assignment)
        assignment_params = self.params.get("treatment_assignment")

        self.overlap = assignment_params.get("overlap", 1)
        propensity, treatment = generation_method(
            x_continuous,
            prob_category,
            params=assignment_params,
            random_state=self.rs_gaussian,
        )
        return (propensity.astype(float).loc[:, 1].values, treatment.astype(int).values)

    def fit_g_link_pipeline(
        self,
        X_parents,
        link_type,
        params,
    ):
        if link_type in list(self.G_LINKING_METHODS.keys()):
            link_pipeline = self.G_LINKING_METHODS[link_type](
                params,
            )
        else:
            raise ValueError(f"unknown link type {link_type}")
        link_pipeline.fit(X_parents)
        return link_pipeline

    def mu(self, X: pd.DataFrame, a: pd.Series):
        """Simulation response surfaces : $\mu(x, a)$

        Args:
            X (pd.DataFrame): [description]
            a (pd.Series): [description]

        Returns:
            [type]: [description]
        """
        if X.shape[1] != len(self.covariate_indices):
            raise ValueError(
                f"Bad shape for x, got {X.shape[1]} instead of {len(self.covariate_indices)}"
            )
        X_given = pd.DataFrame(X, columns=self.covariate_indices)
        if a.size != X.shape[0]:
            raise ValueError(
                f"Bad shape for a ({a.shape[0]}), shoudl match x shape ({X.shape[0]})"
            )

        for var_index in [self.baseline_index, self.effect_index]:
            X_parents_ix = self.covariate_indices
            X_parents = X_given.loc[:, X_parents_ix]
            if self.var_types[var_index] == BASELINE:
                mu_0 = self.baseline_pipeline.transform(X_parents)
            elif self.var_types[var_index] == TREATMENT_EFFECT:
                tau_x = self.effect_pipeline.transform(X_parents)
        y = mu_0 * (1 - self.effect_size) + tau_x * (self.effect_size) * a

        return y

    def to_dict(self) -> Dict:
        """
        Simple converter to dict
        """

        return {
            "baseline_link_type": self.baseline_link_type,
            "effect_link_type": self.effect_link_type,
            "treatment_assignment_type": self.treatment_assignment_type,
            "treatment_link_type": self.treatment_link_type,
            "effect_size": self.effect_size,
            "outcome_noise": self.outcome_noise,
            "treatment_ratio": self.treatment_ratio,
            "rs_rotation": self.rs_rotation,
            "rs_gaussian": self.rs_gaussian,
            "n_samples_init": self.n_samples_init,
            "params": self.params,
            "components": self.baseline_pipeline.pipeline.named_steps[
                "featurization"
            ].components_.tolist(),
        }

    @staticmethod
    def _noise_col(X_signal, snr, cov_std=None, noise=None):
        """
        Noising the given signal according to its size and the given snr and normalizing it to have standard-deviation
        of 1.
        Args:
            X_signal (pd.Series): Covariate column to noise.
            snr (float): Signal to noise ratio that controls the amount of noise to add (value of 1.0 will not generate
                        noise)
            cov_std (float): a given standard deviation
            noise (pd.Series): Gaussian white noise vector.
        Returns:
            (pd.Series, float, pd.Series): 3-element tuple containing:

            - **X_noised** (*pd.Series*): The signal after the noising process
            - **cov_std** (*float*): Standard deviation of the original un-noised signal
            - **noise** (*pd.Series*): The additive noise randomly generated.
        """
        n = X_signal.index.size
        cov_std = cov_std or X_signal.std(ddof=1)  # type: float
        X_signal *= np.sqrt(snr)
        # X_signal /= cov_std
        noise = (
            np.random.normal(loc=0, scale=1, size=n) * np.sqrt(1 - snr)
            if noise is None
            else noise
        )
        X_noised = X_signal + noise
        return (
            pd.Series(X_noised, index=X_signal.index),
            cov_std,
            pd.Series(noise, index=X_signal.index),
        )

    @staticmethod
    def _sigmoid(x, slope=1):
        return 1.0 / (1.0 + np.exp(-slope * x))

    # ### G link methods ### #
    @staticmethod
    def _treatment_random(x_continuous, prob_category, random_state=None):
        """
        Assign treatment to samples completely at random.
        Args:
            x_continuous (pd.Series): Aggregated signal (a scalar per sample) based on the variable's predecessor
                                      variables.
            prob_category (pd.Series): Probability vector the size of number of treatment categories with every entry is
                                       the corresponding probability of that category.

        Returns:
            (pd.DataFrame, pd.DataFrame): 2-element tuple containing:

            - **treatment** (*pd.Series*): Treatment assignment for each sample.
            - **propensity** (*pd.DataFrame*): The marginal conditional probability of treatment given covariates.
                                               A DataFrame shaped (num_samples x num_of_possible_treatment_categories).
        """
        if random_state is None:
            random_state = check_random_state(None)
        index_names = x_continuous.index
        columns_names = prob_category.index
        propensity = pd.DataFrame(
            data=np.tile(prob_category, (len(index_names), 1)),
            index=index_names,
            columns=columns_names,
        )
        treatment = pd.Series(
            data=random_state.choice(
                a=prob_category.index,
                size=len(index_names),
                replace=True,
                p=prob_category,
            ),
            index=index_names,
        )
        return propensity, treatment

    @staticmethod
    def _treatment_logistic_dichotomous(
        x_continuous, prob_category, params=None, random_state=None
    ):
        """
        Assign treatment to samples using a logistic model.
        Args:
            x_continuous (pd.Series): Aggregated signal (a scalar per sample) based on the variable's predecessor
                                      variables.
            prob_category (pd.Series): Probability vector the size of number of treatment categories with every entry is
                                       the corresponding probability of that category.
            params (dict | None): Parameters that will be used in the generation function, e.g. sigmoid slope.

        Returns:
            (pd.DataFrame, pd.DataFrame): 2-element tuple containing:

            - **treatment** (*pd.Series*): Treatment assignment for each sample.
            - **propensity** (*pd.DataFrame*): The marginal conditional probability of treatment given covariates.
                                               A DataFrame shaped (num_samples x num_of_possible_treatment_categories).

        Raises:
            ValueError: If given more than to categories. This method supports dichotomous treatment only.
        """
        if prob_category.size != 2:  # this method suited for dichotomous outcome only
            raise ValueError(
                "logistic method supports only binary treatment. Got the distribution vector "
                "{p_vec} of length {n_cat}".format(
                    n_cat=prob_category.size, p_vec=prob_category
                )
            )
        index_names = x_continuous.index
        columns_names = prob_category.index
        propensity = pd.DataFrame(index=index_names, columns=columns_names)
        # compute propensities:
        t = x_continuous.quantile(prob_category.iloc[1], interpolation="higher")
        slope = params.get("overlap", 1.0) if params is not None else 1.0
        cur_propensity = 1.0 / (
            1 + np.exp(slope * (x_continuous - np.repeat(t, x_continuous.size)))
        )
        # assign the propensity values:
        propensity.loc[:, columns_names[1]] = cur_propensity
        propensity.loc[:, columns_names[0]] = (
            np.ones(cur_propensity.size) - cur_propensity
        )
        treatment = CausalSimulator._sample_from_row_stochastic_matrix(
            propensity, random_state=random_state
        )

        return propensity, treatment

    # ### HELPER FUNCTIONS ### #
    @staticmethod
    def _sample_from_row_stochastic_matrix(propensity, random_state=None):
        """
        Given a row-stochastic matrix (DataFrame) sample one support from each row.
        Args:
            propensity (pd.DataFrame): A row-stochastic DataFrame (i.e. all rows sums to one and non negative).

        Returns:
            treatment (pd.Series): A vector (length of propensity.index) of the resulted sampling.
        """
        if random_state is None:
            random_state = check_random_state(None)
        categories_names = propensity.columns
        prop_cdf = propensity.cumsum(axis="columns")
        r = random_state.uniform(low=0, high=1, size=(propensity.index.size, 1))
        categories = prop_cdf.le(np.tile(r, (1, categories_names.size))).sum(
            axis="columns"
        )
        treatment = pd.Series(
            categories_names[categories].values, index=propensity.index
        )
        # treatment = pd.Series(index=index_names)
        # for i in treatment.index:
        #     treatment[i] = np.random.choice(prob_category.index, [propensity.loc[i, :]])
        return treatment

    @staticmethod
    def _generate_rotation(dim: int = 2, theta: float = None, random_state=None):
        """Generate a random rotation matrix for the given dimensionality and angle.

        Parameters
        ----------
        dim : int
            Dimensionality of the rotation matrix.
        theta : float
            Angle of rotation in radians.
        random_state : int or RandomState
            Random state for the random number generator.

        Returns
        -------
        rotation : ndarray
            Rotation matrix.
        """
        generator = check_random_state(random_state)
        if theta is None:
            theta = generator.uniform(0, 2 * np.pi)
        if dim == 2:
            c, s = np.cos(theta), np.sin(theta)
            rotation = np.array(((c, -s), (s, c)))
        elif dim == 1:
            return np.ones(1)
        else:
            raise NotImplementedError(
                "Only 2D rotation is implemented, in higher dimension, we should consider a combination of rotation on different planes. eg. [block matrix with successives rotations]()."
            )
        return rotation


# ### LINKING PARENTS TO VARIABLE ### #
class AffineTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, *, beta=None, intercept=True, random_state=None):
        self.beta = beta
        self.random_state = random_state
        self.intercept = intercept

    def fit(self, X, y=None):
        X = self._validate_data(X, accept_sparse="csr")
        random_state = check_random_state(self.random_state)
        n_features = X.shape[1]
        if self.intercept:
            n_features += 1

        if self.beta is None:
            self.beta = random_state.normal(
                loc=0.0, scale=1.0, size=n_features
            ) / np.sqrt(n_features)
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = self._validate_data(X, accept_sparse="csr", reset=False)
        if self.intercept:
            X = np.hstack((X, np.ones((X.shape[0], 1))))

        return X.dot(self.beta)


class RbfLink(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        *,
        gamma=1.0,
        n_components=10,
        beta=None,
        intercept=True,
        random_state_feat=None,
        random_state_beta=None,
    ):
        self.gamma = gamma
        self.n_components = n_components
        self.beta = beta
        self.intercept = intercept
        self.random_state_feat = random_state_feat
        self.random_state_beta = random_state_beta
        self.pipeline = Pipeline(
            [
                (
                    "featurization",
                    RBFSampler(
                        random_state=self.random_state_feat,
                        n_components=self.n_components,
                        gamma=self.gamma,
                    ),
                ),
                (
                    "affine",
                    AffineTransformer(
                        random_state=self.random_state_beta,
                        beta=self.beta,
                        intercept=self.intercept,
                    ),
                ),
            ]
        )

    def fit(self, X, y=None):
        self.pipeline.fit(X, y)
        return self

    def transform(self, X):
        return self.pipeline.transform(X)


class SplineLink(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        *,
        degree=3,
        n_knots=3,
        beta=None,
        intercept=True,
        random_state_feat=None,
        random_state_beta=None,
    ):
        self.degree = degree
        self.n_knots = n_knots
        self.beta = beta
        self.intercept = intercept
        self.random_state_beta = random_state_beta
        self.pipeline = Pipeline(
            [
                (
                    "featurization",
                    SplineTransformer(
                        n_knots=self.n_knots,
                        degree=self.degree,
                        knots="uniform",
                        extrapolation="constant",
                    ),
                ),
                (
                    "affine",
                    AffineTransformer(
                        random_state=self.random_state_beta,
                        beta=self.beta,
                        intercept=self.intercept,
                    ),
                ),
            ]
        )

    def fit(self, X, y=None):
        self.pipeline.fit(X, y)
        return self

    def transform(self, X):
        return self.pipeline.transform(X)


class NystroemLink(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        *,
        gamma=1.0,
        n_components=10,
        beta=None,
        intercept=True,
        random_state_feat=None,
        random_state_beta=None,
    ):
        self.gamma = gamma
        self.n_components = n_components
        self.beta = beta
        self.intercept = intercept
        self.random_state_feat = random_state_feat
        self.random_state_beta = random_state_beta
        self.pipeline = Pipeline(
            [
                (
                    "featurization",
                    Nystroem(
                        random_state=self.random_state_feat,
                        n_components=self.n_components,
                        gamma=self.gamma,
                    ),
                ),
                (
                    "affine",
                    AffineTransformer(
                        random_state=self.random_state_beta,
                        beta=self.beta,
                        intercept=self.intercept,
                    ),
                ),
            ]
        )

    def fit(self, X, y=None):
        self.pipeline.fit(X, y)
        return self

    def transform(self, X):
        return self.pipeline.transform(X)


# ### UTILS ### #
# TODO: include into CausalSimulator object ?
def get_transformed_space_from_simu(sim, df):
    """
    Return the latent space features given a simulation and data

    Latent space of the simulation is defined by the featurization performed by both the baseline and the effect pipelines.

    The basis of these two latent spaces is either :
    - common if both pipeline featurization share a random seed
    - fully distinct if the random seeds are different for the two pipelines.

    Args:
        sim ([type]): [description]
        df ([type]): [description]

    Returns:
        [Tuple]: Control and treated latent features, dimension is (baseline_n_components+ effect_n_components)
    """
    x_cols = [col for col in df.columns if col.lower().startswith("x")]
    is_common_basis = sim.baseline_pipeline.get_params().get(
        "random_state_feat", True
    ) == sim.effect_pipeline.get_params().get("random_state_feat", True)

    Z_control = sim.baseline_pipeline.pipeline.named_steps["featurization"].transform(
        df.query("a==0")[x_cols].values
    )
    Z_treated = sim.baseline_pipeline.pipeline.named_steps["featurization"].transform(
        df.query("a==1")[x_cols].values
    )

    if not is_common_basis:
        Z_control = np.concatenate(
            [
                Z_control,
                sim.effect_pipeline.pipeline.named_steps["featurization"].transform(
                    df.query("a==0")[x_cols].values
                ),
            ],
            axis=1,
        )
        Z_treated = np.concatenate(
            [
                Z_treated,
                sim.effect_pipeline.pipeline.named_steps["featurization"].transform(
                    df.query("a==1")[x_cols].values
                ),
            ],
            axis=1,
        )
    return Z_control, Z_treated


# @memory.cache : not working because of can't pickle CausalSimulator
def sample_reject_simulations_on_mmd(
    sim: CausalSimulator,
    mmd_mean,
    mmd_ci,
    n_iter=10,
    random_state=None,
    N_samples=10000,
):
    """
    Iterate over a simulation random seed (thus on the choice of the basis components) to get a simulation with a treated/control mmd within a prespecified range.

    Args:
        sim (CausalSimulator): [description]
        mmd_mean ([type]): [description]
        mmd_ci ([type]): [description]
        n_iter (int, optional): number of seeds to try before stopping. Defaults to 10.
        random_state ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    generator = check_random_state(random_state)
    mmd_min = mmd_mean - mmd_ci
    mmd_max = mmd_mean + mmd_ci
    mmds = []
    seeds = []
    for i in tqdm(range(n_iter)):
        seed = generator.randint(0, 10000000)
        sim.rs_gaussian = seed
        sim.rs_rotation = seed
        seeds.append(seed)

        df = sim.sample(num_samples=N_samples)
        Z_control, Z_treated = get_transformed_space_from_simu(sim, df)
        latent_space_mmd = mmd_rbf(Z_control, Z_treated)
        mmds.append(latent_space_mmd)
        if (latent_space_mmd >= mmd_min) and (latent_space_mmd <= mmd_max):
            print(
                "Simulation found with mmd={:.2E}for {:.2E}target".format(
                    latent_space_mmd, mmd_mean
                )
            )
            return sim, latent_space_mmd
    mmds = np.sort(mmds)
    print(
        f"No simulation found with latent space mmd in [{mmd_min}, {mmd_max}], got following mmds:\n {mmds}"
    )
    return seeds, mmds


# ### SIMULATIONS CONFIG ### #

SIMU_RBF_1 = {
    "type": "rbf",
    "params": {
        "n_components": 2,
        "random_state_feat": 76,
        "gamma": 1
        # "beta": pd.Series({"rbf_0": 1, "rbf_1": 0, "intercept": 0}),
    },
}

SIMU_SPLINE_1 = {
    "type": "spline",
    "params": {
        "degree": 2,
        "n_knots": 4,
        # "beta": pd.Series({"rbf_0": 1, "rbf_1": 0, "intercept": 0}),
    },
}

SIMU_SPLINE_2 = {
    "type": "spline",
    "params": {
        "degree": 3,
        "n_knots": 3,
        # "beta": pd.Series({"rbf_0": 1, "rbf_1": 0, "intercept": 0}),
    },
}

SIMU_NYSTROEM_1 = {
    "type": "nystroem",
    "params": {
        "n_components": 2,
        "gamma": 0.1,
        "random_state_feat": 45,
        "random_state_beta": 45,
    },
}

SIMU_NYSTROEM_2 = {
    "type": "nystroem",
    "params": {
        "n_components": 2,
        "gamma": 0.1,
        "random_state_feat": 45,
        "random_state_beta": 3,
    },
}
