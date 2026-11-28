"""Finite trigger dynamics with an unobserved record of the two orders."""
from pulse import PulseSurvey


class Model(PulseSurvey):
    pulse_coherence = False
