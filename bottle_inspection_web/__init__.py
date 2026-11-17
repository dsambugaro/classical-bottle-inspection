"""Local web interface for the bottle inspection pipeline.

The web app consumes the models trained and validated by the CLI (`make run`); it never
trains. It calls the same preprocessing and feature functions as the CLI, passing the
parameters of each profile (see bottle_inspection.params).
"""

__version__ = "1.0.0"
