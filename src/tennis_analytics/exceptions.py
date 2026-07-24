"""Project-specific exception hierarchy."""


class TennisAnalyticsError(Exception):
    """Base exception for expected application failures."""


class ConfigurationError(TennisAnalyticsError):
    """Raised when configuration is missing or invalid."""


class DataDownloadError(TennisAnalyticsError):
    """Raised when no usable source data can be obtained."""


class DataValidationError(TennisAnalyticsError):
    """Raised when an input dataset is incomplete or malformed."""


class EvaluationError(TennisAnalyticsError):
    """Raised when a model evaluation cannot be completed safely."""
