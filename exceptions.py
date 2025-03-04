"""
Custom exceptions for the plant image dataset framework.
"""


class DirEmptyError(Exception):
    """Exception raised when a directory is empty or doesn't contain expected files."""
    pass


class InvalidModalityError(Exception):
    """Exception raised when an invalid modality type is requested."""
    pass


class ConfigurationError(Exception):
    """Exception raised when there's an issue with the dataset configuration."""
    pass
