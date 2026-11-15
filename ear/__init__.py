"""Emotion-aware allocation and review with piecewise-constant actions."""
from .model import Model, make_model
from .integrate import integrate, objective_gradient
from .optimize import optimize
