class CUDAOutOfMemoryException(Exception):
    """Thrown if system detects that CUDA is out of memory
    """


class StopTrainingException(Exception):
    """Thrown if training should be stopped
    """


class ModelListEmptyException(Exception):
    """Exception that is thrown from the ModelAverager class if an average is requested, but the
    model list is empty
    """


class CallbackFunctionNotFoundException(Exception):
    """Exception that is thrown when a callback function could not be found in a Callback class
    """


class NoCallbackInstanceException(Exception):
    """Exception that is thrown when something which is not an instance of
    a Callback-Class is passed to a CallbackHandler
    """
