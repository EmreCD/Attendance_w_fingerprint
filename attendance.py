import numpy as np
import pandas as pd
import random
from tensorflow.keras.models import load_model
from train_model import SiameseNetwork
from prepare_data import Config, FingerprintDataProcessor

class AttendanceSystem:
    def __init__(self, config, model, data_processor):
        self.config = config
        self.model = model
        self.feature_extractor = model.feature_extractor
        self.data_processor = data_processor
        self.student_features = self._extract_student_features()

    def _extract_student_features(self):
        features = {}
        for student in self.data_processor.student_data:
            img = student['image'].reshape(1,self.config.IMG_HEIGHT,self.config.IMG_WIDTH,1)
            feature = self.feature_extractor.predict(img, verbose=0)
            features[student['student_id']] = {'feature': feature,'name': student['name'],'surname': student['surname']}
        return features

    def cosine_similarity(self, vec1, vec2):
        return np.dot(vec1.flatten(), vec2.flatten()) / (np.linalg.norm(vec1)*np.linalg.norm(vec2))

    def identify_fingerprint(self, test_image):
        test_feature = self.feature_extractor.predict(test_image.reshape(1,self.config.IMG_HEIGHT,self.config.IMG_WIDTH,1), verbose=0)
        best_match, best_similarity = None, 0
        for sid, student in self.student_features.items():
            sim = self.cosine_similarity(test_feature, student['feature'])
            if sim > best_similarity:
                best_similarity = sim
                best_match = sid
        if best_similarity >= self.config.SIMILARITY_THRESHOLD:
            student = self.student_features[best_match]
            return {'status':'Öğrenci','student_id':best_match,'name':student['name'],'surname':student['surname'],'similarity':float(best_similarity),'match':True}
        return {'status':'Dış Kişi','student_id':'N/A','name':'Bilinmiyor','surname':'Bilinmiyor','similarity':float(best_similarity),'match':False}

    def run_attendance(self, num_tests=30):
        all_samples = self.data_processor.student_data + self.data_processor.outsider_data
        test_samples = random.sample(all_samples, num_tests)
        results = []
        correct_predictions = 0
        for i, sample in enumerate(test_samples,1):
            pred = self.identify_fingerprint(sample['image'])
            actual_is_student = sample['is_student']
            predicted_is_student = pred['match']
            is_correct = actual_is_student==predicted_is_student
            if is_correct: correct_predictions+=1
            results.append({'test_no':i,
                            'actual_name':f"{sample['name']} {sample['surname']}",
                            'actual_status':'Öğrenci' if actual_is_student else 'Dış Kişi',
                            'predicted_name':f"{pred['name']} {pred['surname']}",
                            'predicted_status':pred['status'],
                            'similarity':pred['similarity'],
                            'correct':is_correct})
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(self.config.OUTPUT_DIR,'yoklama_sonuclari.csv'),index=False,encoding='utf-8-sig')
        accuracy = (correct_predictions/num_tests)*100
        print(f"Doğruluk: {accuracy:.2f}%")
        print(df.head(10))
        return df, accuracy

if __name__=="__main__":
    config = Config()
    processor = FingerprintDataProcessor(config)
    processor.load_dataset()
    model = SiameseNetwork(config)
    model.model = load_model(os.path.join(config.OUTPUT_DIR,'fingerprint_model.h5'))
    model.feature_extractor = model.model.layers[2]  # base network
    attendance = AttendanceSystem(config, model, processor)
    attendance.run_attendance(num_tests=30)
