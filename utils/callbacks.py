import wandb
from pytorch_lightning.callbacks import Callback


class LogArtifactCallback(Callback):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def on_train_start(self, trainer, pl_module):
        # Access the wandb instance from the logger
        wandb_run = trainer.logger.experiment

        # Create and log the artifact using the provided arguments
        artifact = wandb.Artifact(name="config", type="model")
        artifact.add_file(self.file_path)
        wandb_run.log_artifact(artifact)
