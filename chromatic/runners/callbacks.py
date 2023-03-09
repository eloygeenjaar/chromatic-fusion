from catalyst.core.callback import Callback, CallbackNode, CallbackOrder


class DALICallback(Callback):
    """Logs pipeline execution time."""

    def __init__(self):
        """Initialisation for TimerCallback."""
        super().__init__(order=CallbackOrder.metric + 1, node=CallbackNode.all)

    def on_loader_end(self, runner: "IRunner"):
        runner.loader.reset()