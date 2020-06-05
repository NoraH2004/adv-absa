import clsframework
from clsframework.exceptions import CallbackFunctionNotFoundException, NoCallbackInstanceException
import functools
import logging
from collections import defaultdict


def handlecallback(func):
    """Decorator that updates the state dict with all infos passed as
    keyword arguments and returns required elements of the state dict
    """
    @functools.wraps(func)
    def wrapper_handlecallback(self, *args, **kwargs):
        logging.debug(f"{func.__name__} got keys [{' '.join(kwargs.keys())}]")
        # Integrate received information
        self.state.update(kwargs)
        # Call decorated function
        func(self, *args, **kwargs)
        # Return the internal state of variables that were updated during the call
        retvals = [self.state[key] for key in kwargs.keys()]
        return retvals[0] if len(retvals) == 1 else retvals
    return wrapper_handlecallback


class CallbackHandler():
    """A class managing multiple callback classes.

    Manages calls to functions of callback classes, distributes and updates a state dictionary, etc.
    """

    def __init__(self, callbacks):
        self.callbacks = callbacks
        self.state = {}         # Dictionary containing all managed information
        # Register callback functions to be called when certain values of state dict are updated
        self.val_callbacks = defaultdict(lambda: [])
        for callback in self.callbacks:
            if isinstance(callback, clsframework.Callback):
                for name, func in callback.val_callbacks.items():
                    self.val_callbacks[name].append(func)
            else:
                raise NoCallbackInstanceException(
                    f"{callback} is not an instance of a Callback-Class!")

    def _perform_val_callbacks(self, newvals, depth=0):
        """Calls value callbacks on updated values and integrates returned information

        depth records the number of recursive calls of the function.
        If depth < 2 the recursion is stopped. We do not allow deeper
        recursions because it might happen that a val callback updates
        the value it reacts to, which would lead to an infinite
        circular callback loop.
        """
        if depth < 3:
            allinnernew = {}        # Dict of all values updated by value callbacks
            for key in newvals.keys():
                if key in self.val_callbacks.keys():
                    for callback in self.val_callbacks[key]:
                        inner_new = callback(**self.state)
                        if inner_new is not None:
                            self.state.update(inner_new)
                            allinnernew.update(inner_new)
            # Call value callbacks on newly updated values if they exist
            if len(allinnernew) != 0:
                self._perform_val_callbacks(allinnernew, depth=depth+1)

    def __call__(self, funcname, reverse=False):
        """Calls the member function named funcname of all registered callbacks
        and updates state dict using the returned information

        If reverse is True, the callbacks are used in the reverse order
        """
        for callback in reversed(self.callbacks) if reverse else self.callbacks:
            if hasattr(callback, funcname):
                # Execute callback function
                new = getattr(callback, funcname)(**self.state)
                # If necessary, integrate returned information in managed state
                if new is not None:
                    logging.debug(
                        f"callback {funcname} of {type(callback)} returned [{' '.join(new.keys())}]")
                    # Update state dict with new information
                    self.state.update(new)
                    # Call registered value callbacks if necessary
                    self._perform_val_callbacks(new)
            else:
                raise CallbackFunctionNotFoundException(funcname)

    def __getitem__(self, key):
        """Returns requested value of the managed state"""
        return self.state[key]

    @handlecallback
    def on_model_load(self, **kwargs):
        self("on_model_load")

    @handlecallback
    def on_inference_begin(self, **kwargs):
        self("on_inference_begin")

    @handlecallback
    def on_epoch_begin(self, **kwargs):
        self("on_epoch_begin")

    @handlecallback
    def on_batch_begin(self, **kwargs):
        self("on_batch_begin")

    @handlecallback
    def after_forward_pass(self, **kwargs):
        self("after_forward_pass")

    @handlecallback
    def on_batch_end(self, **kwargs):
        self("on_batch_end", reverse=True)

    @handlecallback
    def on_epoch_end(self, **kwargs):
        self("on_epoch_end", reverse=True)

    @handlecallback
    def on_inference_end(self, **kwargs):
        self("on_inference_end", reverse=True)

    @handlecallback
    def on_model_save(self, **kwargs):
        self("on_model_save")
