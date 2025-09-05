from pathlib import Path

BASE_DIR = Path(__file__).parent

IMAGE_DIR_TRAIN = BASE_DIR / 'stage_2_train_images'
IMAGE_DIR_TEST = BASE_DIR / 'stage_2_test_images'

class_names = ["Lung Opacity", "Normal"]