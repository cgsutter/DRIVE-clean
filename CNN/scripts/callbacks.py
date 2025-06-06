# Portions of this code were writen with the assistance of AI tools (e.g., ChatGPT).

# Necessary imports
import _config as config
import custom_keras_callbacks
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


from wandb.keras import WandbCallback, WandbMetricsLogger, WandbModelCheckpoint


def create_callbacks_list(savebestweights, earlystop_patience = config.earlystop_patience, evid = config.evid):
    """
    Creates a list of Keras callbacks for model training.

    Parameters
    ----------
    savebestweights : str
        File path to save the best model weights based on validation accuracy.
    earlystop_patience : int, optional
        Number of epochs with no improvement after which training will be stopped (default from config).
    evid : bool, optional
        If True, includes ReduceLROnPlateau to reduce learning rate when a metric has stopped improving (default from config).

    Returns
    -------
    list
        A list of Keras callbacks to be used during model training.
    """

    print("started model_fitting")
    print(savebestweights)

    checkpoint = ModelCheckpoint(
        savebestweights,
        monitor="val_accuracy",
        verbose=0,
        save_best_only=True,
        mode="max",
    )  # saves out best model locally


    es = custom_keras_callbacks.EarlyStopping(
        monitor="val_accuracy", mode="max", patience=earlystop_patience
    )

    # wb_metricslog = WandbMetricsLogger(
    #     log_freq="epoch"
    # )  # this plots loss and accuracy curves to w&b UI (under "epochs" dropdown). It also logs learning rate. These are plots in the UI dropdown where tables are, but also in the "Overview" part of that run. Can adjust log freq if want to do by batch rather than epoch
    # wb_checkpoint = WandbModelCheckpoint(filepath = savebestweights, monitor = "val_acc", save_best_only=True, Mode = "max") # this and "checkpoint" above essentially do the same thing but saving out best model also as an artifact on w&b too

    print("got through es")
    callbacks_list = [
        checkpoint,
        es,
        # wb_metricslog,
    ]
    if evid:
        print("evidential learing, setting learning rate reduction on plateau")
        reducelr = ReduceLROnPlateau(
            factor=0.1,
            min_lr=1.0e-15,
            mode="max",
            monitor="val_accuracy",
            patience=3,
            verbose=0,
        )
        # logger.info("... loaded ReduceLROnPlateau")
        callbacks_list.append(reducelr)
        print(
            "added another callback to reduce learning rate on plateau for adam loss function"
        )
    return callbacks_list