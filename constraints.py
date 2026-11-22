"""Projection et optimisation sous contrainte de norme L2."""

from regularization import parameter_norm


def project_l2_ball(parameters, radius):
    """Projette les poids sur une boule L2 ; les biais ne sont pas contraints."""
    if radius < 0:
        raise ValueError("Le rayon doit être positif ou nul")

    projected = {key: value.copy() for key, value in parameters.items()}
    norm = parameter_norm(parameters)

    if norm > radius and norm > 0:
        scale = radius / norm
        for key in projected:
            if key.startswith("W"):
                projected[key] *= scale

    return projected


def projected_gradient_step(parameters, gradients, learning_rate, radius):
    """Effectue un pas de gradient puis projette les poids sur la boule."""
    tentative = {
        key: parameters[key] - learning_rate * gradients[key]
        for key in parameters
    }
    return project_l2_ball(tentative, radius)
