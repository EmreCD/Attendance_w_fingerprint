import os
import cv2
import random
import pandas as pd

class Config:
    DATASET_PATH = './SOCOFing/Real'
    OUTPUT_DIR = './output'
    NUM_STUDENTS = 30
    NUM_OUTSIDERS = 20
    IMG_HEIGHT = 96
    IMG_WIDTH = 96

    def __init__(self):
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

class FingerprintDataProcessor:
    def __init__(self, config):
        self.config = config
        self.student_data = []
        self.outsider_data = []

    def load_and_preprocess_image(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        img = cv2.resize(img, (self.config.IMG_WIDTH, self.config.IMG_HEIGHT))
        img = cv2.equalizeHist(img)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = img.astype('float32') / 255.0
        return img

    def load_dataset(self):
        all_fingerprints = []
        for filename in os.listdir(self.config.DATASET_PATH):
            if filename.endswith('.BMP'):
                img = self.load_and_preprocess_image(os.path.join(self.config.DATASET_PATH, filename))
                if img is not None:
                    person_id = filename.split('__')[0]
                    finger_name = filename.split('__')[1].replace('.BMP','')
                    all_fingerprints.append({
                        'image': img,
                        'person_id': person_id,
                        'filename': filename,
                        'finger_name': finger_name
                    })

        random.shuffle(all_fingerprints)
        person_ids = sorted(list(set([f['person_id'] for f in all_fingerprints])))
        student_ids = person_ids[:self.config.NUM_STUDENTS]
        outsider_ids = person_ids[self.config.NUM_STUDENTS:self.config.NUM_STUDENTS+self.config.NUM_OUTSIDERS]

        for sid in student_ids:
            self.student_data.extend([f for f in all_fingerprints if f['person_id']==sid])
        for oid in outsider_ids:
            self.outsider_data.extend([f for f in all_fingerprints if f['person_id']==oid])

        turkish_names = [
            ("Ahmet","Yılmaz"), ("Mehmet","Kaya"), ("Ayşe","Demir"), ("Fatma","Çelik"),
            ("Ali","Şahin"), ("Zeynep","Öztürk"), ("Mustafa","Aydın"), ("Elif","Özdemir"),
            ("Hüseyin","Arslan"), ("Emine","Doğan"), ("Can","Kılıç"), ("Merve","Aslan"),
            ("Emre","Çetin"), ("Selin","Kara"), ("Burak","Koç"), ("Deniz","Kurt"),
            ("Cem","Özkan"), ("Ebru","Şimşek"), ("Onur","Yıldız"), ("Gizem","Yıldırım"),
            ("Tolga","Avcı"), ("Pınar","Çakır"), ("Serkan","Aksoy"), ("Burcu","Türk"),
            ("Murat","Güneş"), ("Seda","Erdoğan"), ("Kerem","Özer"), ("Esra","Polat"),
            ("Barış","Kaplan"), ("İrem","Yavuz")
        ]

        for i, sid in enumerate(student_ids):
            for f in [f for f in self.student_data if f['person_id']==sid]:
                f['name'] = turkish_names[i][0]
                f['surname'] = turkish_names[i][1]
                f['student_id'] = f"2024{i+1:03d}"
                f['is_student'] = True

        for f in self.outsider_data:
            f['name'] = f"Dış Kişi {f['person_id']}"
            f['surname'] = "---"
            f['student_id'] = "N/A"
            f['is_student'] = False

        print(f"✓ {len(student_ids)} öğrenci ve {len(outsider_ids)} dış kişi yüklendi")
        return True

    def create_student_database(self):
        db = []
        for f in self.student_data + self.outsider_data:
            db.append({
                'student_id': f['student_id'],
                'name': f['name'],
                'surname': f['surname'],
                'filename': f['filename'],
                'finger_name': f['finger_name']
            })
        df = pd.DataFrame(db)
        df.to_csv(os.path.join(self.config.OUTPUT_DIR,'ogrenci_veritabani.csv'), index=False, encoding='utf-8-sig')
        print(f"✓ Öğrenci veritabanı kaydedildi")
        return df

if __name__ == "__main__":
    config = Config()
    processor = FingerprintDataProcessor(config)
    processor.load_dataset()
    df = processor.create_student_database()
    print(df.head())
