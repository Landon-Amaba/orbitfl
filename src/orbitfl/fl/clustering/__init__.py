"""Clustering strategies: abstract base class and four implementations.

    ephemeris     clusters known in advance from orbital geometry (GTRFL in the thesis)
    similarity    clusters discovered from model-update similarity (MCFL)
    plane         static clusters, one per orbital plane
    random_churn  random clusters that change at a matched rate (control)
"""
