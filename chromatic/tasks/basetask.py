class BaseTask:
    def __init__(self,
                 model,
                 dataset_generator,
                 logdir,
                 *args,
                 **kwargs):
        
        self._model = model
        self._dataset_generator = dataset_generator
        self._logdir = logdir

    def run(self):
        raise NotImplementedError
