
import numpy as np

IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600
FPS = 10


def carla_to_bgr(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    return array[:, :, :3]
