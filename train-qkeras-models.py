import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Activation,
    AveragePooling2D,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    Reshape,
    UpSampling2D,
    Conv2DTranspose
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from qkeras import QActivation, QConv2D, QDense, QDenseBatchnorm, quantized_bits


class TeacherAutoencoder:
    def __init__(self, input_shape: tuple):
        self.input_shape = input_shape

    def get_model(self):
        inputs = Input(shape=self.input_shape, name="teacher_inputs_")
        x = Reshape((18, 14, 1), name="teacher_reshape")(inputs)
        x = Conv2D(20, (3, 3), strides=1, padding="same", name="teacher_conv2d_1")(x)
        x = Activation("relu", name="teacher_relu_1")(x)
        x = AveragePooling2D((2, 2), name="teacher_pool_1")(x)
        x = Conv2D(30, (3, 3), strides=1, padding="same", name="teacher_conv2d_2")(x)
        x = Activation("relu", name="teacher_relu_2")(x)
        x = Flatten(name="teacher_flatten")(x)
        x = Dense(80, activation="relu", name="teacher_latent")(x)
        x = Dense(9 * 7 * 30, name="teacher_dense")(x)
        x = Reshape((9, 7, 30), name="teacher_reshape2")(x)
        x = Activation("relu", name="teacher_relu_3")(x)
        x = Conv2D(30, (3, 3), strides=1, padding="same", name="teacher_conv2d_3")(x)
        x = Activation("relu", name="teacher_relu_4")(x)
        x = UpSampling2D((2, 2), name="teacher_upsampling")(x)
        x = Conv2D(20, (3, 3), strides=1, padding="same", name="teacher_conv2d_4")(x)
        x = Activation("relu", name="teacher_relu_5")(x)
        outputs = Conv2D(
            1,
            (3, 3),
            activation="relu",
            strides=1,
            padding="same",
            name="teacher_outputs",
        )(x)
        return Model(inputs, outputs, name="teacher")


class TeacherAutoencoderRevised:
    def __init__(self, input_shape: tuple):
        self.input_shape = input_shape

    def get_model(self):
        inputs = Input(shape=self.input_shape, name="teacher_inputs_")
        x = Reshape((18, 14, 1), name="teacher_reshape")(inputs)
        x = Conv2D(20, (3, 3), strides=1, padding="same", name="teacher_conv2d_1")(x)
        x = Activation("relu", name="teacher_relu_1")(x)
        x = AveragePooling2D((2, 2), name="teacher_pool_1")(x)
        x = Conv2D(30, (3, 3), strides=1, padding="same", name="teacher_conv2d_2")(x)
        x = Activation("relu", name="teacher_relu_2")(x)
        x = Flatten(name="teacher_flatten")(x)
        x = Dense(80, activation="relu", name="teacher_latent")(x)
        x = Dense(9 * 7 * 30, name="teacher_dense")(x)
        x = Reshape((9, 7, 30), name="teacher_reshape2")(x)
        x = Activation("relu", name="teacher_relu_3")(x)
        x = Conv2D(30, (3, 3), strides=1, padding="same", name="teacher_conv2d_3")(x)
        x = Activation("relu", name="teacher_relu_4")(x)
        x = Conv2DTranspose(30, (3, 3), strides=2, padding="same", name="teacher_conv_transpose")(x)
        x = Conv2D(20, (3, 3), strides=1, padding="same", name="teacher_conv2d_4")(x)
        x = Activation("relu", name="teacher_relu_5")(x)
        outputs = Conv2D(
            1,
            (3, 3),
            activation="relu",
            strides=1,
            padding="same",
            name="teacher_outputs",
        )(x)
        return Model(inputs, outputs, name="teacher-transpose")


class CicadaV1:
    def __init__(self, input_shape: tuple):
        self.input_shape = input_shape

    def get_model(self):
        inputs = Input(shape=self.input_shape, name="inputs_")
        x = QDenseBatchnorm(
            16,
            kernel_quantizer=quantized_bits(8, 1, 1, alpha=1.0),
            bias_quantizer=quantized_bits(8, 3, 1, alpha=1.0),
            name="dense1",
        )(inputs)
        x = QActivation("quantized_relu(10, 6)", name="relu1")(x)
        x = Dropout(1 / 8)(x)
        x = QDense(
            1,
            kernel_quantizer=quantized_bits(12, 3, 1, alpha=1.0),
            use_bias=False,
            name="dense2",
        )(x)
        outputs = QActivation("quantized_relu(16, 8)", name="outputs")(x)
        return Model(inputs, outputs, name="cicada-v1")


class CicadaV2:
    def __init__(self, input_shape: tuple):
        self.input_shape = input_shape

    def get_model(self):
        inputs = Input(shape=self.input_shape, name="inputs_")
        x = Reshape((18, 14, 1), name="reshape")(inputs)
        x = QConv2D(
            4,
            (2, 2),
            strides=2,
            padding="valid",
            use_bias=False,
            kernel_quantizer=quantized_bits(12, 3, 1, alpha=1.0),
            name="conv",
        )(x)
        x = QActivation("quantized_relu(10, 6)", name="relu0")(x)
        x = Flatten(name="flatten")(x)
        x = Dropout(1 / 9)(x)
        x = QDenseBatchnorm(
            16,
            kernel_quantizer=quantized_bits(8, 1, 1, alpha=1.0),
            bias_quantizer=quantized_bits(8, 3, 1, alpha=1.0),
            name="dense1",
        )(x)
        x = QActivation("quantized_relu(10, 6)", name="relu1")(x)
        x = Dropout(1 / 8)(x)
        x = QDense(
            1,
            kernel_quantizer=quantized_bits(12, 3, 1, alpha=1.0),
            use_bias=False,
            name="dense2",
        )(x)
        outputs = QActivation("quantized_relu(16, 8)", name="outputs")(x)
        return Model(inputs, outputs, name="cicada-v2")
    

if __name__ == "__main__":
    from utils import get_data
    import math

    data_path = "saved_inputs_targets"
    X_train, y_train, is_outlier_train, X_val, y_val, is_outlier_val, X_test, y_test, is_outlier_test = get_data(data_path)

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

    cv1_mc = ModelCheckpoint(f"data/{cicada_v1.name}", save_best_only=True)
    cv1_log = CSVLogger(f"data/{cicada_v1.name}-training.log", append=True)
    
    cv2_mc = ModelCheckpoint(f"data/{cicada_v2.name}", save_best_only=True)
    cv2_log = CSVLogger(f"data/{cicada_v2.name}-training.log", append=True)

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

    np.save("data/cicada_v1_preds.npy", preds_v1.flatten())
    np.save("data/cicada_v2_preds.npy", preds_v2.flatten())
