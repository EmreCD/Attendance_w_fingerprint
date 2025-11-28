import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
from data_prep import Config, FingerprintDataProcessor

class SiameseNetwork:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.feature_extractor = None

    def create_base_network(self):
        input_shape = (self.config.IMG_HEIGHT, self.config.IMG_WIDTH, 1)
        model = Sequential([
            layers.Conv2D(64,(3,3),activation='relu',input_shape=input_shape,padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64,(3,3),activation='relu',padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2,2)), layers.Dropout(0.25),
            layers.Conv2D(128,(3,3),activation='relu',padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128,(3,3),activation='relu',padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2,2)), layers.Dropout(0.25),
            layers.Conv2D(256,(3,3),activation='relu',padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256,(3,3),activation='relu',padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2,2)), layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='sigmoid')
        ])
        return model

    def build_siamese_model(self):
        input_a = layers.Input(shape=(self.config.IMG_HEIGHT,self.config.IMG_WIDTH,1))
        input_b = layers.Input(shape=(self.config.IMG_HEIGHT,self.config.IMG_WIDTH,1))
        base_network = self.create_base_network()
        feature_a = base_network(input_a)
        feature_b = base_network(input_b)
        l1_distance = layers.Lambda(lambda tensors: tf.abs(tensors[0]-tensors[1]))([feature_a,feature_b])
        prediction = layers.Dense(1, activation='sigmoid')(l1_distance)
        self.model = Model(inputs=[input_a,input_b], outputs=prediction)
        self.feature_extractor = base_network
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return self.model

    def create_training_pairs(self, data):
        pairs, labels = [], []
        person_groups = {}
        for item in data:
            pid = item['student_id']
            if pid not in person_groups:
                person_groups[pid] = []
            person_groups[pid].append(item['image'])
        for pid, images in person_groups.items():
            for i in range(len(images)-1):
                pairs.append([images[i],images[i+1]])
                labels.append(1)
        all_images = [item['image'] for item in data]
        all_pids = [item['student_id'] for item in data]
        num_negative = len(pairs)
        while len(labels) < 2*num_negative:
            idx1, idx2 = random.sample(range(len(all_images)),2)
            if all_pids[idx1]!=all_pids[idx2]:
                pairs.append([all_images[idx1],all_images[idx2]])
                labels.append(0)
        return np.array(pairs), np.array(labels)

    def train(self, data):
        pairs, labels = self.create_training_pairs(data)
        split = int(len(pairs)*0.8)
        train_pairs, val_pairs = pairs[:split], pairs[split:]
        train_labels, val_labels = labels[:split], labels[split:]
        train_pairs = train_pairs.reshape(-1,2,self.config.IMG_HEIGHT,self.config.IMG_WIDTH,1)
        val_pairs = val_pairs.reshape(-1,2,self.config.IMG_HEIGHT,self.config.IMG_WIDTH,1)
        history = self.model.fit([train_pairs[:,0],train_pairs[:,1]],train_labels,
                                 validation_data=([val_pairs[:,0],val_pairs[:,1]],val_labels),
                                 batch_size=32, epochs=50, verbose=1)
        self.model.save(os.path.join(self.config.OUTPUT_DIR,'fingerprint_model.h5'))
        print("✓ Model kaydedildi")
        return history

if __name__ == "__main__":
    config = Config()
    processor = FingerprintDataProcessor(config)
    processor.load_dataset()
    all_data = processor.student_data + processor.outsider_data
    model = SiameseNetwork(config)
    model.build_siamese_model()
    model.train(all_data)
