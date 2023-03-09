from nvidia.dali.plugin.pytorch import DALIGenericIterator


class CatalystDALIGenericIterator(DALIGenericIterator):
    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)
