# config.py
import os

class Config:
    # Dataset ve output yolları
    DATASET_PATH = './SOCOFing/Real'
    OUTPUT_DIR = './output'
    MODEL_PATH = './models'

    # Model parametreleri
    IMG_HEIGHT = 96
    IMG_WIDTH = 96
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001

    # Öğrenci ve outsider sayısı
    NUM_STUDENTS = 30
    NUM_OUTSIDERS = 20
    TOTAL_SAMPLES = 50

    # Tanıma eşik değeri
    SIMILARITY_THRESHOLD = 0.85

    def __init__(self):
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.MODEL_PATH, exist_ok=True)
