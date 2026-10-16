"""
This module contains code for the Bayesian nested sampling routine
Code adapted from: https://github.com/CarlosSMWolff/ParamEst-NN
"""

import numpy as np
from numpy import exp, cos, cosh, sqrt
from scipy.integrate import cumulative_trapezoid


def compute_waiting_time(tau: float, Omega: float, Delta: float):
    """
    Calculates the waiting time distribution of a coherently driven two-level system based on Omega and Delta.

    Parameters:
    ----------
    tau (float): Time delay.
    Omega (float): Parameter Omega.
    Delta (float): Parameter Delta.

    Returns:
    ----------
    float: Calculated w(tau) value.
    """
    w_tau = (
        8
        * exp(-tau / 2)
        * Omega**2
        / sqrt(-64 * Omega**2 + (1 + 4 * Delta**2 + 16 * Omega**2) ** 2)
        * (
            -cos(
                tau
                * sqrt(
                    -1
                    + 4 * Delta**2
                    + 16 * Omega**2
                    + sqrt(-64 * Omega**2 + (1 + 4 * Delta**2 + 16 * Omega**2) ** 2)
                )
                / (2 * sqrt(2))
            )
            + cosh(
                tau
                * sqrt(
                    1
                    - 4 * Delta**2
                    - 16 * Omega**2
                    + sqrt(-64 * Omega**2 + (1 + 4 * Delta**2 + 16 * Omega**2) ** 2)
                )
                / (2 * sqrt(2))
            )
        )
    )

    return w_tau


def compute_waiting_time_list(params, taulist):
    """
    Calculate the waiting time distribution of a coherently driven two-level system for a list of time delays.

    Parameters:
    ----------
    params (list): List containing two parameters - Omega and Delta.
    taulist (list): List of time delay values.

    Returns:
    ----------
    numpy.ndarray: An array containing the waiting time distribution.
    """
    Omega, Delta = params
    wtau = np.asarray([compute_waiting_time(tau, Omega, Delta) for tau in taulist])
    return wtau


def compute_log_likelihood_analytical(
    Delta: float, data: np.array, gamma: float = 1.0, Omega: float = 1.0
) -> np.array:
    """
    Returns the log likelihood function log[P(D|Delta)] that will tell us how probable the data D is, given our model with parameters Delta.
    This calculation uses analytical formulas for the likelihood.

    Parameters:
    -----------
    Delta (float): Model parameter Delta. This is the parameter being estimated.
    data (np.array): The trajectory data in the form of an array of time delays
    gamma (float): Model parameter gamma (asssumed known)
    Omega (float): Model parameter Omega (assumed known)

    Returns:
    -----------
    np.array: The log-likelihood function for each observation (we compute it after each jump)
    """
    tau_list = data
    wlist = np.array([compute_waiting_time(tau, Omega, Delta) for tau in tau_list])

    # Jumps are independent, therefore, the likelihood is the product of the probabilities, and the log-likelihood, the sum of the log-probabilities.
    log_likelihood = np.sum(np.log(wlist)) + 1e-300

    return log_likelihood


def compute_likelihood_analytical(
    Delta: float, data: np.array, gamma: float = 1.0, Omega: float = 1.0
) -> np.array:
    """
    Returns the likelihood function P(D|Delta) that will tell us how probable the data D is, given our model with parameters Delta.
    This calculation uses analytical formulas for the likelihood.

    Parameters:
    -----------
    Delta (float): Model parameter Delta. This is the parameter being estimated.
    data (np.array): The trajectory data in the form of an array of time delays
    gamma (float): Model parameter gamma (asssumed known)
    Omega (float): Model parameter Omega (assumed known)

    Returns:
    -----------
    np.array: The likelihood function for each observation (we compute it after each jump)
    """

    log_likelihood = compute_log_likelihood_analytical(Delta, data, gamma, Omega)

    return np.exp(log_likelihood)


def get_estimates_Bayesian(
    data: np.array,
    DeltaMin: float = 0.0,
    DeltaMax: float = 5.0,
    nDeltaGrid: int = 500,
    output_probabilities=False,
    only_mean=False,
):
    """
    Calculate estimates of Delta (1D parameter estimation) in a coherently driven TLS by computing the probability distribution over a grid of Delta values.
    Then, estimates are computed by taking the mean, median and maximum.
    We use analytical computations of the likelihood for this.

    Parameters:
    -----------
    data (np.array): List of time delays representing quantum jumps.
    output_probabilities (bool): Whether to output the probability distribution.
    DeltaMin (float): Minimum value of delta for the analysis.
    DeltaMax (float): Maximum value of delta for the analysis.
    nDeltaGrid (int): Number of points in the grid for delta values.

    Returns:
    -----------
    tuple: A tuple containing the estimated delta values (mean, median, and max).
    If output_probabilities is True:
        The tuple also includes the probability distribution for delta values.

    """
    # Generate a fine grid of delta values
    DeltaBayesListFine = np.linspace(DeltaMin, DeltaMax, nDeltaGrid)

    # Calculate the likelihood grid using analytical computation
    likelihood_grid = np.array(
        [compute_likelihood_analytical(Delta, data) for Delta in DeltaBayesListFine]
    )

    # Compute the probability distribution by normalizing the likelihood
    prob_grid = likelihood_grid / np.trapz(likelihood_grid, DeltaBayesListFine)
    cdf_grid = cumtrapz(prob_grid, DeltaBayesListFine)

    # Calculate mean, median, and maximum estimates for delta
    deltaMean = np.trapz(prob_grid * DeltaBayesListFine, DeltaBayesListFine)

    if only_mean == True:
        return deltaMean

    deltaMedian = DeltaBayesListFine[np.argmin(np.abs(cdf_grid - 0.5))]
    deltaMax = DeltaBayesListFine[np.argmax(prob_grid)]

    # Check if output_probabilities is requested
    if output_probabilities == False:
        return deltaMean, deltaMedian, deltaMax
    else:
        return deltaMean, deltaMedian, deltaMax, prob_grid


def compute_bayes_posterior(
    data: list[float],
    x_vals: np.ndarray,
    prior_delta_min: float = 0.0,
    prior_delta_max: float = 5.0,
    n_grid: int = 500,
    omega: float = 1.0,
    gamma: float = 1.0,
):
    """
    Compute Bayesian posterior P(Delta|D) with a uniform prior on
    [prior_delta_min, prior_delta_max], evaluated on x_vals.

    Returns
    -------
    posterior : np.ndarray
        Posterior evaluated on the grid
    """

    x = x_vals

    # Likelihood on grid
    likelihood = np.array([
        compute_likelihood_analytical(Delta, data, gamma, omega) for Delta in x
    ])

    # Uniform prior
    prior = np.where(
        (x >= prior_delta_min) & (x <= prior_delta_max),
        1.0 / (prior_delta_max - prior_delta_min),
        0.0,
    )

    # Unnormalized posterior
    posterior = likelihood * prior

    # Normalize
    Z = np.trapz(posterior, x)
    if Z > 0:
        posterior /= Z

    return posterior
