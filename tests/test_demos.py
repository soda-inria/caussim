
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from caussim.config import DIR2FIGURES_T

from caussim.data.loading import load_dataset
from caussim.demos.utils import plot_models_comparison
from caussim.estimation.estimators import CateEstimator, SLearner, TLearner
from caussim.experiences.base_config import CAUSSIM_1D_CONFIG


def test_plot_models_comparison():
    simu_config = CAUSSIM_1D_CONFIG.copy()
    sim, causal_df = load_dataset(simu_config)
    
    model_ref = CateEstimator(
        meta_learner=TLearner(
            final_estimator=RandomForestRegressor(n_estimators=10, max_depth=10),
        )
    )
    # model_compared = CateEstimator(
    #     meta_learner=SLearner(
    #     final_estimator=RandomForestRegressor(n_estimators=10, max_depth=10),
    # ))
    model_compared = CateEstimator(
        meta_learner=SLearner(
        final_estimator=Ridge(),
    ))
    x_train = causal_df.get_aX()
    y_train = causal_df.get_y()
    model_ref.fit(x_train, y_train)
    model_compared.fit(x_train, y_train)
    sim.rs_gaussian = 42
    causal_df_test = sim.sample(num_samples=500)
    
    predictions_model_ref = model_ref.predict(causal_df_test.get_aX())
    predictions_model_compared = model_compared.predict(causal_df_test.get_aX())
    # Only for test, we would set annotation to 0 if we had no nuisance estimations
    predictions_model_ref["check_e"] = 0.5
    predictions_model_ref["check_m"] = 0.5
    predictions_model_compared["check_e"] = 0.5
    predictions_model_compared["check_m"] = 0.5
    
    plot_models_comparison(
        df=causal_df_test.df,
        predictions_model_ref=predictions_model_ref,
        predictions_model_compared=predictions_model_compared,
    )
    plt.savefig(DIR2FIGURES_T / "test_plot_model_comparison.png", bbox_inches="tight")
    
    
    