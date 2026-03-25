import os
import random

import carla
import tensorflow as tf
import numpy as np

from const import USE_LAST_WEIGHT, PRINT_DEBUG_OUTPUT_MODEL


class DeepReinforcementModel:

    def __init__(self, epsilon=0.1, epsilon_decay=0.995, min_epsilon=0.01):
        self.model = self._build_model()
        self.last_prediction = None
        self.save_to_nb_epoch = 0
        self.max_save_by_epoch = 10
        self.epsilon = epsilon  # exploration rate
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.replay_buffer = []
        self.buffer_size = 1000

    def predict(self, input_ai, training=False):

        if training and random.random() < self.epsilon:
            output = np.random.rand(4)
        else:
            direction_tensor = tf.convert_to_tensor(
                [
                    [
                        input_ai["gps"],
                        input_ai["center_left"],
                        input_ai["center_right"],
                        input_ai["distance_vehicle_in_front"],
                        input_ai["distance_fire_light"],
                    ]
                ],
                dtype=tf.float32,
            )
            output = self.model(direction_tensor).numpy()[0]

        if PRINT_DEBUG_OUTPUT_MODEL:
            print(f"Prediction: {output}")
        self.last_prediction = output
        return self._forward_to_control(output)

    def train(self, output_to_compute_error):
        corrected_output = self._find_correct_output(output_to_compute_error)
        if PRINT_DEBUG_OUTPUT_MODEL:
            print(f"Corrected output: {corrected_output}")

        if self.last_prediction is not None:
            self.replay_buffer.append(
                {
                    "input": [
                        output_to_compute_error["gps"],
                        output_to_compute_error["center_left"],
                        output_to_compute_error["center_right"],
                        output_to_compute_error["distance_vehicle_in_front"],
                        output_to_compute_error["distance_fire_light"],
                    ],
                    "target": corrected_output,
                }
            )

            if len(self.replay_buffer) > self.buffer_size:
                self.replay_buffer.pop(0)

            if len(self.replay_buffer) >= 32:
                self._train_on_batch()

        self.save_to_nb_epoch += 1
        if self.save_to_nb_epoch >= self.max_save_by_epoch:
            self._save_weights(self.model)
            self.save_to_nb_epoch = 0
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def _train_on_batch(self, batch_size=32):
        batch = random.sample(self.replay_buffer, batch_size)
        inputs = tf.convert_to_tensor([b["input"] for b in batch], dtype=tf.float32)
        targets = tf.convert_to_tensor([b["target"] for b in batch], dtype=tf.float32)
        self.model.fit(inputs, targets, epochs=1, verbose=0)

    @staticmethod
    def _forward_to_control(output_ai):
        control = carla.VehicleControl()
        control.steer = float(np.clip(output_ai[0], -1.0, 1.0))
        control.throttle = float(np.clip(output_ai[1], 0.0, 1.0))
        control.brake = round(np.clip(output_ai[2], 0.0, 1.0))
        control.reverse = True if round(output_ai[3]) == 1 else False
        return control

    @staticmethod
    def _find_correct_output(output_to_compute_error):
        control = output_to_compute_error["control"]
        is_blocked = output_to_compute_error["is_blocked"]
        if is_blocked:
            return [0.0, 0.0, 1.0, 0.0]
        elif control.reverse:
            return [0.0, 0.0, 0.0, 1.0]
        else:
            return [control.steer, control.throttle, control.brake, 0.0]

    @staticmethod
    def _build_model(filename="deep_reinforcement_model_.weights.h5"):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(5,)),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(4, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse"
        )
        if os.path.exists(filename) and USE_LAST_WEIGHT:
            model.load_weights(filename)
        return model

    @staticmethod
    def _save_weights(model, filename="deep_reinforcement_model_.weights.h5"):
        model.save_weights(filename)
