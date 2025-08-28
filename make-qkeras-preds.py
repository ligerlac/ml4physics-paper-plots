import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from qkeras import QActivation, QConv2D, QDense, QDenseBatchnorm, quantized_bits

from tensorflow.keras.models import load_model


if __name__ == "__main__":
    from utils import get_training_data, get_input_data
    import math

    X_train, y_train, is_outlier_train, X_val, y_val, is_outlier_val, X_test, y_test, is_outlier_test = get_training_data("saved_inputs_targets", zb_frac=5)

    cicada_v1 = load_model("data/models/cicada-v1", custom_objects={"QDense": QDense, "quantized_bits": quantized_bits, "QActivation": QActivation})
    cicada_v2 = load_model("data/models/cicada-v2", custom_objects={"QConv2D": QConv2D, "quantized_bits": quantized_bits, "QActivation": QActivation})

    preds_v1 = cicada_v1.predict(X_test.reshape((-1, 252, 1)))
    preds_v2 = cicada_v2.predict(X_test)

    np.save("data/predictions/cicada-v1/x_test.npy", preds_v1.flatten())
    np.save("data/predictions/cicada-v2/x_test.npy", preds_v2.flatten())

    exit(0)

    # Calculate steps for more frequent logging
    total_samples = len(X_train)
    batch_size = 256
    total_steps = math.ceil(total_samples / batch_size)

    # Log every N batches by setting steps_per_epoch to N
    log_every_n_batches = 100  # Adjust this value
    steps_per_epoch = log_every_n_batches

    # Calculate how many "epochs" to cover your full dataset
    epochs_to_cover_full_dataset = math.ceil(total_steps / steps_per_epoch)
    total_epochs_needed = 2 * epochs_to_cover_full_dataset

    print(f"Total samples: {total_samples}")
    print(f"Batch size: {batch_size}")
    print(f"Total steps for full dataset: {total_steps}")
    print(f"Logging every {log_every_n_batches} batches")
    print(f"Total 'epochs' needed: {total_epochs_needed}")

    cicada_v1 = CicadaV1((252,)).get_model()
    cicada_v1.compile(optimizer=Adam(learning_rate=0.001), loss="mae")

    cicada_v2 = CicadaV2((18,14)).get_model()
    cicada_v2.compile(optimizer=Adam(learning_rate=0.001), loss="mae")

    cv1_mc = ModelCheckpoint(f"data/models/{cicada_v1.name}", save_best_only=True)
    cv1_log = CSVLogger(f"data/models/{cicada_v1.name}-training.log", append=True)
    
    cv2_mc = ModelCheckpoint(f"data/models/{cicada_v2.name}", save_best_only=True)
    cv2_log = CSVLogger(f"data/models/{cicada_v2.name}-training.log", append=True)

    cicada_v1.fit(
        X_train.reshape((-1, 252, 1)),
        y_train,
        validation_data=(X_val.reshape((-1, 252, 1)), y_val),
        epochs=total_epochs_needed,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        callbacks=[cv1_mc, cv1_log],
    )

    cicada_v2.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=total_epochs_needed,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        callbacks=[cv2_mc, cv2_log],
    )

    preds_v1 = cicada_v1.predict(X_test.reshape((-1, 252, 1)))
    preds_v2 = cicada_v2.predict(X_test)

    np.save("data/predictions/x_test_cicada_v1_preds.npy", preds_v1.flatten())
    np.save("data/predictions/x_test_cicada_v2_preds.npy", preds_v2.flatten())

    input_data = get_input_data()

    for key in input_data:
        preds_v1 = cicada_v1.predict(input_data[key].reshape((-1, 252, 1)))
        preds_v2 = cicada_v2.predict(input_data[key])
        np.save(f"data/predictions/{key}_cicada_v1_preds.npy", preds_v1.flatten())
        np.save(f"data/predictions/{key}_cicada_v2_preds.npy", preds_v2.flatten())
