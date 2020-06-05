from clsframework import Callback


class TorchOptimizer(Callback):
    """
    Encapsulates arbitrary optimizers from torch and torch_optimizer as a module for our system
    """

    def __init__(self, optclass, grad_accumulation_steps=1, **kwargs):
        super().__init__()
        self.optargs = kwargs       # Pass all arguments to optimizer
        self.optclass = optclass    # Class to instantiate to make optimizer
        self.optimizer = None
        # Set up stuff for gradient accumulation
        self.grad_accumulation_steps = grad_accumulation_steps
        self.next_grad = grad_accumulation_steps  # How many samples until next gradient and weight update

    def on_model_load(self, **kwargs):
        pass

    def on_inference_begin(self, model, classifier, **kwargs):
        # Set up Optimizer if this has not been done already
        if self.optimizer is None:
            # This is a bit complicated to match the optimizer from the BERT paper
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [{
                'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01
            }, {
                'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }]
            self.optimizer = self.optclass(optimizer_grouped_parameters, **self.optargs)

            # Set model and optimizer to FP16 if that has not yet happened and is requested
            if classifier.fp16 is not None and classifier.set_fp16 is None:
                try:
                    import apex
                    classifier.model, self.optimizer = apex.amp.initialize(classifier.model, self.optimizer,
                                                                           opt_level=classifier.fp16)
                    classifier.set_fp16 = classifier.fp16
                except ModuleNotFoundError:
                    print("Not able to set model to FP16 since apex is not installed")
                    classifier.fp16 = None
            # Register optimizer in managed state
            return {"optimizer": self.optimizer}

    def on_epoch_begin(self, **kwargs):
        pass

    def on_batch_begin(self, **kwargs):
        pass

    def after_forward_pass(self, y, nbseensamples, **kwargs):
        if y is not None:
            if nbseensamples > self.next_grad:
                self.next_grad = nbseensamples + self.grad_accumulation_steps
                # Change model parameters
                self.optimizer.step()
                # Reset gradient to zero since backprop in pytorch is cumulative
                self.optimizer.zero_grad()

    def on_batch_end(self, **kwargs):
        pass

    def on_epoch_end(self, **kwargs):
        pass

    def on_inference_end(self, **kwargs):
        pass

    def on_model_save(self, **kwargs):
        pass
