import os

import carla
import tensorflow as tf

from const import USE_LAST_WEIGHT


class DeepReinforcementModel:

    def __init__(self):
        self.model = self._build_model()
        self.last_prediction = None
        self.save_to_nb_epoch = 0
        self.max_save_by_epoch = 10

    def predict(self, input_ai):
        direction = input_ai["gps"]
        direction = tf.convert_to_tensor([[direction]], dtype=tf.float32)
        output = self.model(direction)
        action = tf.argmax(
            output[0]
        ).numpy()  # 5 actions: straight, left, right, brake, reverse
        self.last_prediction = action
        return self._forward_to_control(action)

    def train(self, output_to_compute_error):
        corrected_output = self._find_correct_output(output_to_compute_error)
        if self.last_prediction is not None:
            target = tf.convert_to_tensor([corrected_output], dtype=tf.float32)
            with tf.GradientTape() as tape:
                direction = tf.convert_to_tensor(
                    [[output_to_compute_error["gps"]]], dtype=tf.float32
                )
                output = self.model(direction)
                loss = tf.keras.losses.MeanSquaredError()(target, output[0])
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.model.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables)
            )
        self.save_to_nb_epoch += 1
        if self.save_to_nb_epoch >= self.max_save_by_epoch:
            self._save_weights(self.model)
            self.save_to_nb_epoch = 0

    @staticmethod
    def _forward_to_control(output_ai):
        if output_ai == 1:  # turn left
            control = carla.VehicleControl()
            control.steer = -1.0
            control.throttle = 0.5
            return control
        elif output_ai == 2:  # turn right
            control = carla.VehicleControl()
            control.steer = 1.0
            control.throttle = 0.5
            return control
        elif output_ai == 0:  # go straight
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.5
            return control
        elif output_ai == 3:  # brake
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            return control
        elif output_ai == 4:  # reverse
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.5
            control.reverse = True
            return control
        return carla.VehicleControl()  # default to no control

    @staticmethod
    def _find_correct_output(output_to_compute_error):
        if output_to_compute_error["is_blocked"]:
            return 4  # reverse
        if output_to_compute_error["gps"] == 0:
            return 0  # go straight
        elif output_to_compute_error["gps"] == -1:
            return 1  # turn left
        elif output_to_compute_error["gps"] == 1:
            return 2  # turn right
        else:
            return 0  # default to straight

    @staticmethod
    def _build_model(filename="deep_reinforcement_model_.weights.h5"):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(1,)),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(
                    5, activation="softmax"
                ),  # 5 actions: straight, left, right, brake, reverse
            ]
        )
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        if os.path.exists(filename) and USE_LAST_WEIGHT:
            model.load_weights(filename)
        return model

    @staticmethod
    def _save_weights(model, filename="deep_reinforcement_model_.weights.h5"):
        model.save_weights(filename)
