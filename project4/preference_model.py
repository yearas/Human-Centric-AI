# The preference model is a linear model that predicts the score of an item based on its features and a set of weights.
# The weights are learned from observed rankings of items and the model assumes that the probability of an item being chosen over others follows a Luce choice model.

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


# Strength of the Gaussian prior on w. With 102 features and only a handful of
# comparisons per participant the likelihood alone has infinitely many perfect
# solutions, so this term is what makes the estimate well-defined at all.
DEFAULT_PRIOR_STRENGTH = 1.0


def compute_item_scores(weights, feature_matrix, ranking):
    scores = []
    for item_index in ranking:
        item_features = feature_matrix[item_index]
        scores.append(np.dot(item_features, weights))
    return np.array(scores)


def negative_log_posterior(weights, rankings, feature_matrix, prior_strength):
    total = 0.0
    for ranking in rankings:
        scores = compute_item_scores(weights, feature_matrix, ranking)
        for position in range(len(ranking)):
            # At each position the winner competes only against the items that have not been placed yet, which is what makes the whole ranking a chain of Luce choices rather than independent comparisons
            remaining_scores = scores[position:]
            total = total - scores[position] + logsumexp(remaining_scores)

    total = total + 0.5 * prior_strength * np.dot(weights, weights)
    return total


def negative_log_posterior_gradient(weights, rankings, feature_matrix, prior_strength):
    number_of_features = len(weights)
    gradient = np.zeros(number_of_features)

    for ranking in rankings:
        scores = compute_item_scores(weights, feature_matrix, ranking)
        for position in range(len(ranking)):
            remaining_scores = scores[position:]
            remaining_probabilities = np.exp(remaining_scores - logsumexp(remaining_scores))

            expected_features = np.zeros(number_of_features)
            # The expected features are computed as a weighted sum of the features of the remaining items, where the weights are the probabilities of each item being chosen at that position in the ranking
            for offset in range(len(remaining_scores)):
                item_index = ranking[position + offset]
                expected_features = expected_features + remaining_probabilities[offset] * feature_matrix[item_index]

            chosen_features = feature_matrix[ranking[position]]
            gradient = gradient - chosen_features + expected_features

    gradient = gradient + prior_strength * weights
    return gradient


def fit_preference_weights(rankings, feature_matrix, prior_strength=DEFAULT_PRIOR_STRENGTH):
    number_of_features = feature_matrix.shape[1]
    initial_weights = np.zeros(number_of_features)

    # The optimization is performed using the L-BFGS-B algorithm, which is a quasi-Newton method that approximates the inverse Hessian matrix to find the minimum of the negative log-posterior function.
    # The gradient of the negative log-posterior is also provided to speed up convergence.
    result = minimize(
        negative_log_posterior,
        initial_weights,
        args=(rankings, feature_matrix, prior_strength),
        jac=negative_log_posterior_gradient,
        method='L-BFGS-B',
    )
    return result.x

# The posterior covariance matrix is computed as the inverse of the Hessian matrix of the negative log-posterior function evaluated at the estimated weights
def posterior_covariance(weights, rankings, feature_matrix, prior_strength=DEFAULT_PRIOR_STRENGTH):
    number_of_features = len(weights)
    hessian = np.zeros((number_of_features, number_of_features))

    for ranking in rankings:
        scores = compute_item_scores(weights, feature_matrix, ranking)
        for position in range(len(ranking)):
            remaining_scores = scores[position:]
            remaining_probabilities = np.exp(remaining_scores - logsumexp(remaining_scores))

            expected_features = np.zeros(number_of_features)
            for offset in range(len(remaining_scores)):
                item_index = ranking[position + offset]
                expected_features = expected_features + remaining_probabilities[offset] * feature_matrix[item_index]

            for offset in range(len(remaining_scores)):
                item_index = ranking[position + offset]
                difference = feature_matrix[item_index] - expected_features
                hessian = hessian + remaining_probabilities[offset] * np.outer(difference, difference)

    for feature_position in range(number_of_features):
        hessian[feature_position, feature_position] = hessian[feature_position, feature_position] + prior_strength

    return np.linalg.inv(hessian)
