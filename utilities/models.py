from __future__ import annotations
import pandas as pd
import numpy as np
import pymc as pm
import arviz
from typing import Optional
from scipy.special import logit


def hierarchical_binomial_regression(
        y: np.array,
        decision_values: np.array,
        subjects_indices: np.array,
        coords: dict,
        b_prior_mean: Optional[float]=0,
        b_prior_sigma: Optional[float]=2,
        s_prior_sigma: Optional[float]=2,
        n_draws: Optional[int]=1000,
        n_tuning_draws: Optional[int]=1000
) -> tuple[pm.Model, arviz.InferenceData]:
    """
    This function fits a hierarchical logistic regression of decision values onto y, 
    with random slopes for each subject.

    Parameters
    ----------
    y : np.array [N, ]
        Observed binary data (accept, reject...)
    decision_values : np.array [N, ]
        Decision values to regress onto the observed data
    subjects_indices : np.array  [N, ]
        Index of each participant to which the slopes are fitted separately
    coords : dict  
        "subject": subj_labels, 
        "coef": ["intercept", "slope"],
        The subject maps the data to each subject, the coef are for the coefficients
    b_prior_mean : Optional[float], optional
        Prior mean of each beta parameters, by default 0
    b_prior_sigma : Optional[float], optional
        Prior variance of the population level distribution of the beta, by default 2
    s_prior_sigma : Optional[float], optional
        Prior between subjects variance, by default 2
    n_drawss : Optional[int], optional
        Number of draws for the posterior, by default 1000
    n_tuning_draws : Optional[int], optional
        Number of tuning draws, by default 1000
    Returns
    -------
    tuple[pm.Model, arviz.InferenceData]
        pm.model : pymc model object
        idata : arviz inference data
    """
    with pm.Model(coords=coords) as model:
        y_obs = pm.Data("y_obs", y)
        dv = pm.Data("decision_values", decision_values)
        subj_idx = pm.Data("subj_idx", subjects_indices.astype("int32"))

        # Hyperpriors:
        beta_pop = pm.Normal("beta_pop", mu=b_prior_mean, sigma=b_prior_sigma, dims="coef")
        sigma_pop = pm.HalfNormal("sigma_pop", sigma=s_prior_sigma, dims="coef")

        # Non-centered random interecept and slope:
        z = pm.Normal("z", 0, 1, dims=("subject", "coef"))
        beta_sub = pm.Deterministic("beta_sub", beta_pop + z * sigma_pop, dims=("subject", "coef"))

        # Expected value
        eta = beta_sub[subj_idx, 0] + beta_sub[subj_idx, 1] * dv
        p = pm.Deterministic("p", pm.math.sigmoid(eta))

        # Likelihood:
        pm.Bernoulli("y", p=p, observed=y_obs)

        # Sample:
        idata = pm.sample(
            draws=n_draws,
            tune=n_tuning_draws,
            idata_kwargs={"log_likelihood": True},
        )

    return model, idata


def hierarchical_policy_compression(
        y: np.array,
        decision_values: np.array,
        p_a: np.array,
        subjects_indices: np.array,
        coords: dict,
        b_prior_mean: Optional[float]=0,
        b_prior_sigma: Optional[float]=2,
        s_prior_sigma: Optional[float]=2,
        n_draws: Optional[int]=1000,
        n_tuning_draws: Optional[int]=1000
) -> tuple[pm.Model, arviz.InferenceData]:
    """
    This function fits a hierarchical logistic regression of decision values onto y, 
    with random slopes for each subject.

    Parameters
    ----------
    y : np.array [N, ]
        Observed binary data (accept, reject...)
    decision_values : np.array [N, ]
        Decision values to regress onto the observed data
    subjects_indices : np.array  [N, ]
        Index of each participant to which the slopes are fitted separately
    coords : dict  
        "subject": subj_labels, 
        "coef": ["intercept", "slope"],
        The subject maps the data to each subject, the coef are for the coefficients
    b_prior_mean : Optional[float], optional
        Prior mean of each beta parameters, by default 0
    b_prior_sigma : Optional[float], optional
        Prior variance of the population level distribution of the beta, by default 2
    s_prior_sigma : Optional[float], optional
        Prior between subjects variance, by default 2
    n_drawss : Optional[int], optional
        Number of draws for the posterior, by default 1000
    n_tuning_draws : Optional[int], optional
        Number of tuning draws, by default 1000
    Returns
    -------
    tuple[pm.Model, arviz.InferenceData]
        pm.model : pymc model object
        idata : arviz inference data
    """
    # Clipping action probability for numerical stability:
    p_a = np.clip(p_a, 1e-6, 1-1e-6)
    with pm.Model(coords=coords) as model:
        y_obs = pm.Data("y_obs", y)
        dv = pm.Data("decision_values", decision_values)
        pa_data = pm.Data("p_a", p_a)
        subj_idx = pm.Data("subj_idx", subjects_indices.astype("int32"))

        # Hyperpriors:
        beta_pop = pm.Normal("beta_pop", mu=b_prior_mean, sigma=b_prior_sigma, dims="coef")
        sigma_pop = pm.HalfNormal("sigma_pop", sigma=s_prior_sigma, dims="coef")

        # Non-centered random interecept and slope:
        z = pm.Normal("z", 0, 1, dims=("subject", "coef"))
        beta_sub = pm.Deterministic("beta_sub", beta_pop + z * sigma_pop, dims=("subject", "coef"))

        # Expected values:
        eta = beta_sub[subj_idx, 0] + beta_sub[subj_idx, 1] * dv + pm.math.logit(pa_data)
        p = pm.Deterministic("p", pm.math.sigmoid(eta))

        # Likelihood 
        pm.Bernoulli("y", p=p, observed=y_obs)

        # Sampling:
        idata = pm.sample(
            draws=n_draws,
            tune=n_tuning_draws,
            idata_kwargs={"log_likelihood": True},
        )

    return model, idata


def policy_compression_model(
        y: np.array,
        coords: dict,
        X: Optional[np.array]=None,
        Z: Optional[np.array]=None,
        X_prior: Optional[np.array]=None,
        Z_prior: Optional[np.array]=None,
        k_prior: Optional[np.array]=None,
        random_effects: Optional[np.array]=None,
        b_prior_mean: Optional[float]=0,
        b_prior_sigma: Optional[float]=2,
        s_prior_sigma: Optional[float]=2,
        n_draws: Optional[int]=1000,
        n_tuning_draws: Optional[int]=1000
) -> tuple[pm.Model, arviz.InferenceData]:
    """
    This function fits a hierarchical logistic regression of decision values onto y, 
    with random slopes for each subject.

    Parameters
    ----------
    y : np.array [N, ]
        Observed binary data (accept, reject...)
    decision_values : np.array [N, ]
        Decision values to regress onto the observed data
    subjects_indices : np.array  [N, ]
        Index of each participant to which the slopes are fitted separately
    coords : dict  
        "subject": subj_labels, 
        "coef": ["intercept", "slope"],
        The subject maps the data to each subject, the coef are for the coefficients
    b_prior_mean : Optional[float], optional
        Prior mean of each beta parameters, by default 0
    b_prior_sigma : Optional[float], optional
        Prior variance of the population level distribution of the beta, by default 2
    s_prior_sigma : Optional[float], optional
        Prior between subjects variance, by default 2
    n_drawss : Optional[int], optional
        Number of draws for the posterior, by default 1000
    n_tuning_draws : Optional[int], optional
        Number of tuning draws, by default 1000
    Returns
    -------
    tuple[pm.Model, arviz.InferenceData]
        pm.model : pymc model object
        idata : arviz inference data
    """
    # Clipping action probability for numerical stability:
    if X_prior is not None:
        X_prior = np.clip(X_prior, 1e-6, 1-1e-6)

    subj_idx_raw, subj_labels = pd.factorize(data["vpn"])
    coords = {
        "groups": subj_labels,
        "fixed_effects": ['is_full_energy', 'is_low_energy_LC', 'is_low_energy_HC'],
        "random_effects": ['b_23', 'b_14', 'theta_01', 'theta_02', 'theta_03', 'theta_04'],
        "fixed_effects_priors": ['b_offer_prior'],
        "random_effects_prior": ['b_23', 'b_14', 'theta_01', 'theta_02', 'theta_03', 'theta_04'],
    }
    with pm.Model(coords=coords) as model:
        y_obs = pm.Data("y_obs", y)

        # Specify the regressors
        X = pm.Data("X", X)  # Fixed effects (i.e. kept constant across participants)
        Z = pm.Data("Z", Z)  # Random effects (i.e. varying across participants)
        X_prior = pm.Data("X_prior", X_prior)  # Fixed effects related to priors over actions (i.e. kept constant across participants)
        Z_prior = pm.Data("X_prior", Z_prior)  # Random effects related to priors over actions (i.e. varying across participants)
        k_prior = pm.Data("k_prior", k_prior)  # Constant effect of priors, i.e. not fitted to the data 
        random_effects = pm.Data("random_effects", random_effects)

        # ========================================================
        # Hyperpriors
        # Hyperpriors on the fixed effects (i.e. kept constant across participants):
        beta_fixed = pm.Normal("beta_fixed", mu=b_prior_mean, sigma=b_prior_sigma, dims="fixed_effects")
        beta_prior_fixed = pm.Normal("beta_fixed", mu=b_prior_mean, sigma=b_prior_sigma, dims="fixed_effects_priors")

        # Hyperpriors on the random effects (i.e. varying across participants):
        beta_random = pm.Normal("beta_random", mu=b_prior_mean, sigma=b_prior_sigma, dims="random_effects")
        sigma_pop = pm.HalfNormal("sigma_pop", sigma=s_prior_sigma, dims="random_effects")
        beta_prior_random = pm.Normal("beta_random", mu=b_prior_mean, sigma=b_prior_sigma, dims="random_effects_prior")
        sigma_prior_pop = pm.HalfNormal("sigma_pop", sigma=s_prior_sigma, dims="random_effects_prior")

        # Non-centered random interecept and slope:
        z = pm.Normal("z", 0, 1, dims=("random_effects", "fixed_effects"))
        beta_grp = pm.Deterministic("beta_sub", beta_fixed + z * sigma_pop, dims=("random_effects", "fixed_effects"))
        z_prior = pm.Normal("z_prior", 0, 1, dims=("random_effects", "prior_fixed_effects"))
        beta_grp = pm.Deterministic("beta_sub", beta_fixed + z * sigma_pop, dims=("random_effects", "fixed_effects"))

        # Expected values:
        if X_prior is not None and X_prior_fixed is not None:
            eta = (beta_sub[Z, 0:n_planning] * X_plan).sum(axis=-1) + \
                (beta_sub[Z, n_planning:] * pm.math.logit(X_prior)).sum(axis=-1) + pm.math.logit(X_prior_fixed)
        elif X_prior is not None:
            eta = (beta_sub[Z, 0:n_planning] * X_plan).sum(axis=-1) + \
                (beta_sub[Z, n_planning:] * pm.math.logit(X_prior)).sum(axis=-1)
        elif X_prior_fixed is not None:
            eta = (beta_sub[Z, 0:n_planning] * X_plan).sum(axis=-1) + pm.math.logit(X_prior_fixed)
        else:
            eta = (beta_sub[Z, 0:n_planning] * X_plan).sum(axis=-1)
        p = pm.Deterministic("p", pm.math.sigmoid(eta))

        # Likelihood 
        pm.Bernoulli("y", p=p, observed=y_obs)

        # Sampling:
        idata = pm.sample(
            draws=n_draws,
            tune=n_tuning_draws,
            idata_kwargs={"log_likelihood": True},
        )

    return model, idata


def ott_model(
        y: np.array,
        decision_values: np.array,
        is_basic: np.array,
        is_full_energy: np.array,
        is_low_energy_LC: np.array,
        is_low_energy_HC: np.array,
        subjects_indices: np.array,
        coords: dict,
        b_prior_mean: Optional[float]=0,
        b_prior_sigma: Optional[float]=2,
        s_prior_sigma: Optional[float]=2,
        n_draws: Optional[int]=1000,
        n_tuning_draws: Optional[int]=1000
) -> tuple[pm.Model, arviz.InferenceData]:
    """
    Ott's hybrid model from https://www.sciencedirect.com/science/article/pii/S1053811922003469:

    Parameters
    ----------
    y : np.array [N, ]
        Observed binary data (accept, reject...)
    decision_values : np.array [N, ]
        Decision values to regress onto the observed data
    is_basic : np.array [N, ]
        Regressor indicating basic trials (i.e. enough energy and energy below max)
    is_full_energy : np.array [N, ]
        Regressor indicating trials where energy is max
    is_low_energy_LC : np.array [N, ]
        Regressor indicating trials where energy is low and cost is low
    is_low_energy_HC : np.array [N, ]
        Regressor indicating trials where energy is low and cost is high
    subjects_indices : np.array  [N, ]
        Index of each participant to which the slopes are fitted separately
    coords : dict  
        "subject": subj_labels, 
        "coef": ["intercept", "slope"],
        The subject maps the data to each subject, the coef are for the coefficients
    b_prior_mean : Optional[float], optional
        Prior mean of each beta parameters, by default 0
    b_prior_sigma : Optional[float], optional
        Prior variance of the population level distribution of the beta, by default 2
    s_prior_sigma : Optional[float], optional
        Prior between subjects variance, by default 2
    n_drawss : Optional[int], optional
        Number of draws for the posterior, by default 1000
    n_tuning_draws : Optional[int], optional
        Number of tuning draws, by default 1000
    Returns
    -------
    tuple[pm.Model, arviz.InferenceData]
        pm.model : pymc model object
        idata : arviz inference data
    """
    with pm.Model(coords=coords) as mdl:
        y_obs = pm.Data("y_obs", y)
        dv = pm.Data("Decision values", decision_values)
        is_basic = pm.Data("is_basic", is_basic)
        is_maxE = pm.Data("is_maxE", is_full_energy)
        is_minE_LC = pm.Data("is_minE_LC", is_low_energy_LC)
        is_minE_HC = pm.Data("is_minE_HC", is_low_energy_HC)
        subj_idx = pm.Data("subj_idx", subjects_indices)

        # Population level priors:
        beta_pop = pm.Normal("beta_pop", mu=b_prior_mean, sigma=b_prior_sigma, dims="coef")
        sigma_pop = pm.HalfNormal('sigma_pop', sigma=s_prior_sigma, dims="coef")
        # Non centered parametrization of within subject coefficients
        z = pm.Normal("z", 0, 1, dims=("subject", "coef"))
        beta_sub = pm.Deterministic("beta_sub", beta_pop + z * sigma_pop, dims=("subject","coef"))

        # Likelihood
        p = pm.Deterministic("p", 
                            pm.math.sigmoid(
                                beta_sub[subj_idx, 0] * dv +
                                beta_sub[subj_idx, 1] * is_basic +
                                beta_sub[subj_idx, 2] * is_maxE +
                                beta_sub[subj_idx, 3] * is_minE_LC +
                                beta_sub[subj_idx, 4] * is_minE_HC
                                ))
        pm.Bernoulli("y", p=p, observed=y_obs)

        idata = pm.sample(n_draws, tune=n_tuning_draws, target_acceptance=0.95, idata_kwargs={"log_likelihood": True})

    return mdl, idata



def ott_model_with_action_prior(
        y: np.array,
        decision_values: np.array,
        is_basic: np.array,
        is_full_energy: np.array,
        is_low_energy_LC: np.array,
        is_low_energy_HC: np.array,
        p_a: np.array,
        subjects_indices: np.array,
        coords: dict,
        b_prior_mean: Optional[float]=0,
        b_prior_sigma: Optional[float]=2,
        s_prior_sigma: Optional[float]=2,
        n_draws: Optional[int]=1000,
        n_tuning_draws: Optional[int]=1000
) -> tuple[pm.Model, arviz.InferenceData]:
    """
    Ott's hybrid model from https://www.sciencedirect.com/science/article/pii/S1053811922003469:

    Parameters
    ----------
    y : np.array [N, ]
        Observed binary data (accept, reject...)
    decision_values : np.array [N, ]
        Decision values to regress onto the observed data
    is_basic : np.array [N, ]
        Regressor indicating basic trials (i.e. enough energy and energy below max)
    is_full_energy : np.array [N, ]
        Regressor indicating trials where energy is max
    is_low_energy_LC : np.array [N, ]
        Regressor indicating trials where energy is low and cost is low
    is_low_energy_HC : np.array [N, ]
        Regressor indicating trials where energy is low and cost is high
    p_a : np.array [N, ]
        Action prior
    subjects_indices : np.array  [N, ]
        Index of each participant to which the slopes are fitted separately
    coords : dict  
        "subject": subj_labels, 
        "coef": ["intercept", "slope"],
        The subject maps the data to each subject, the coef are for the coefficients
    b_prior_mean : Optional[float], optional
        Prior mean of each beta parameters, by default 0
    b_prior_sigma : Optional[float], optional
        Prior variance of the population level distribution of the beta, by default 2
    s_prior_sigma : Optional[float], optional
        Prior between subjects variance, by default 2
    n_drawss : Optional[int], optional
        Number of draws for the posterior, by default 1000
    n_tuning_draws : Optional[int], optional
        Number of tuning draws, by default 1000
    Returns
    -------
    tuple[pm.Model, arviz.InferenceData]
        pm.model : pymc model object
        idata : arviz inference data
    """
    # Clipping action probability for numerical stability:
    p_a = np.clip(p_a, 1e-6, 1-1e-6)
    with pm.Model(coords=coords) as mdl:
        y_obs = pm.Data("y_obs", y)
        dv = pm.Data("Decision values", decision_values)
        is_basic = pm.Data("is_basic", is_basic)
        is_maxE = pm.Data("is_maxE", is_full_energy)
        is_minE_LC = pm.Data("is_minE_LC", is_low_energy_LC)
        is_minE_HC = pm.Data("is_minE_HC", is_low_energy_HC)
        pa_data = pm.Data("p_a", p_a)
        subj_idx = pm.Data("subj_idx", subjects_indices)

        # Population level priors:
        beta_pop = pm.Normal("beta_pop", mu=b_prior_mean, sigma=b_prior_sigma, dims="coef")
        sigma_pop = pm.HalfNormal('sigma_pop', sigma=s_prior_sigma, dims="coef")
        # Non centered parametrization of within subject coefficients
        z = pm.Normal("z", 0, 1, dims=("subject", "coef"))
        beta_sub = pm.Deterministic("beta_sub", beta_pop + z * sigma_pop, dims=("subject","coef"))

        # Likelihood
        p = pm.Deterministic("p", 
                            pm.math.sigmoid(
                                beta_sub[subj_idx, 0] * dv +
                                beta_sub[subj_idx, 1] * is_basic +
                                beta_sub[subj_idx, 2] * is_maxE +
                                beta_sub[subj_idx, 3] * is_minE_LC +
                                beta_sub[subj_idx, 4] * is_minE_HC +
                                pm.math.logit(pa_data)
                                ))
        pm.Bernoulli("y", p=p, observed=y_obs)

        idata = pm.sample(n_draws, tune=n_tuning_draws, target_acceptance=0.95, idata_kwargs={"log_likelihood": True})

    return mdl, idata


def preference_model(
        y: np.array,
        decision_values: np.array,
        pref_regressors: pd.DataFrame,
        subject_index: np.array,
        subject_labels: np.array      
):
    '''
    Parameters
    ----------
    y : np.array [N samples, ]
        Observed binary data (1, 0...)
    decision_values : np.array [N samples, ]
        Decision values to regress onto the observed data
    pref_regressors : np.array [N samples, M regressors]
        Regressor to fit participants preference for. We can have M regressors
    subject_index : np.array  [N samples, ]
        Index of the subject associated with each observation
    subject_labels : np.array  [N subjects, ]
        Single identifier of each subject
    coords : dict  
        "subject": subj_labels, 
        "coef": ["intercept", "slope"],
        The subject maps the data to each subject, the coef are for the coefficients
    b_prior_mean : Optional[float], optional
        Prior mean of each beta parameters, by default 0
    b_prior_sigma : Optional[float], optional
        Prior variance of the population level distribution of the beta, by default 2
    s_prior_sigma : Optional[float], optional
        Prior between subjects variance, by default 2
    n_drawss : Optional[int], optional
        Number of draws for the posterior, by default 1000
    n_tuning_draws : Optional[int], optional
        Number of tuning draws, by default 1000
    Returns
    -------
    tuple[pm.Model, arviz.InferenceData]
        pm.model : pymc model object
        idata : arviz inference data
    """
    '''
    # Get dimensions:
    n_obs = y.shape[0]
    n_groups = subject_labels.shape[0]
    n_pref = pref_regressors.shape[1]

    # Create intercept:
    intercept = np.ones(n_obs)

    # Set coordinates:
    coords = {
        "subject": subject_labels,
        "coef_intercept": ["B_intercept"],
        "coef_planning": ["B_plan"],
        "coef_pref": ["B_" + col for col in pref_regressors.columns],
        "coef_interaction": ["slope"],
    }


    # Model:
    with pm.Model(coords=coords) as planning_preferences_interaction_model:
        # Data:
        y_obs = pm.Data("y_obs", y)
        intercept = pm.Data("intercept", intercept)
        planning = pm.Data("planning", decision_values)
        preferences = pm.Data("preferences", pref_regressors)
        subj_idx = pm.Data("subj_idx", subject_index.astype("int32"))

        # Hyperpriors:
        # Intercept term
        beta_intercept = pm.Normal("beta_intercept", mu=0, sigma=2, dims="coef_intercept")
        sigma_intercept = pm.HalfNormal("sigma_intercept", sigma=2, dims="coef_intercept")
        # Planning term:
        beta_planning = pm.Normal("beta_planning", mu=0, sigma=2, dims="coef_planning")
        sigma_planning = pm.HalfNormal("sigma_planning", sigma=2, dims="coef_planning")
        # Preference terms:
        beta_pref = pm.Normal("beta_pref", mu=0, sigma=2, dims="coef_pref")
        sigma_pref = pm.HalfNormal("sigma_pref", sigma=2, dims="coef_pref")
        # Interaction term:
        beta_interaction = pm.Normal("beta_interaction", mu=0, sigma=2, dims="coef_interaction")
        sigma_interaction = pm.HalfNormal("sigma_interaction", sigma=2, dims="coef_interaction")

        # Offset parameters:
        z_intercept = pm.Normal("z_intercept", 0, 1, dims=("subject", "coef_intercept"))
        z_planning = pm.Normal("z_planning", 0, 1, dims=("subject", "coef_planning"))
        z_biases = pm.Normal("z_biases", 0, 1, dims=("subject", "coef_pref"))
        z_interaction = pm.Normal("z_interaction", 0, 1, dims=("subject", "coef_interaction"))

        # Centered parameters:
        beta_intercept_sub = pm.Deterministic("beta_intercept_sub", beta_intercept + z_intercept * sigma_intercept, 
                                              dims=("subject", "coef_intercept"))
        beta_planning_sub = pm.Deterministic("beta_planning_sub", beta_planning + z_planning * sigma_planning, 
                                             dims=("subject", "coef_planning"))
        beta_pref_sub = pm.Deterministic("beta_pref_sub", beta_pref + z_biases * sigma_pref, 
                                         dims=("subject", "coef_pref"))
        beta_interaction_sub = pm.Deterministic("beta_interaction_sub", beta_interaction + z_interaction * sigma_interaction, 
                                                dims=("subject", "coef_interaction"))
        
        # Estimate the score of the bias (i.e. weighted sum of each of the biases regressors):
        preference = pm.Deterministic('preference', (beta_pref_sub[subj_idx] * preferences).sum(axis=-1))
        
        # Convert the bias back onto probability space:
        pi_prior = pm.Deterministic("pi_prior", pm.math.sigmoid(preference))

        # Compute the entropy:
        entropy = pm.Deterministic("entropy", -pi_prior * pm.math.log(pi_prior) - (1-pi_prior) * pm.math.log(1 - pi_prior))
        
        # Eta parameter is the weighted sum of the intercept, the bias, the planning values and 
        # the interaction between the entropy of the bias and the planning
        eta = (
            beta_intercept_sub[subj_idx, 0] * intercept
            + preference
            + beta_planning_sub[subj_idx, 0] * planning
            + beta_interaction_sub[subj_idx, 0] * (entropy * planning)
        )
        
        # Expected values:
        p = pm.Deterministic("p", pm.math.sigmoid(eta))

        # Likelihood 
        pm.Bernoulli("y", p=p, observed=y_obs)

        # Sampling:
        idata = pm.sample(
            draws=1000,
            tune=1000,
            chains=4,
            target_accept=0.85,
            idata_kwargs={"log_likelihood": True},
        )

    return idata
