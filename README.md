# Obesity Classification Model

## 🎯 เป้าหมายของโครงงาน

โปรเจกต์นี้มีวัตถุประสงค์เพื่อสร้างโมเดล Machine Learning ที่สามารถ**ทำนายระดับน้ำหนักของบุคคล**จากข้อมูลพฤติกรรมและลักษณะทางกายภาพ โดยแบ่งออกเป็น 4 ระดับ:

1. **Underweight** - น้ำหนักต่ำกว่าเกณฑ์
2. **Normal** - น้ำหนักปกติ
3. **Overweight** - น้ำหนักเกิน
4. **Obesity** - โรคอ้วน

## 📊 Dataset

ข้อมูลประกอบด้วย **1,610 รายการ** พร้อม 14 features ที่เกี่ยวข้องกับ:
- ข้อมูลทางกายภาพ (เพศ, อายุ, ส่วนสูง, น้ำหนัก)
- พฤติกรรมการกิน (การกินผัก, ผลไม้, มื้ออาหาร, แคลอรี่)
- พฤติกรรมการออกกำลังกาย
- พฤติกรรมการใช้เทคโนโลยี
- การเดินทาง

## 🔬 วิธีการทำงาน

### 1. Data Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- การจัดการข้อมูลที่ไม่สมดุล (Imbalanced Data) ด้วย **SMOTE** (Synthetic Minority Over-sampling Technique)

### 2. Model Training
ใช้ 3 โมเดลหลัก:
- **Random Forest Classifier**
- **Gradient Boosting Classifier**
- **Logistic Regression**

### 3. Ensemble Method
รวมผลการทำนายจาก 3 โมเดลด้วยวิธี **Voting Classifier** เพื่อเพิ่มความแม่นยำ

### 4. Hyperparameter Tuning
ใช้ **GridSearchCV** พร้อม **StratifiedKFold Cross-Validation** (5-fold) เพื่อหาพารามิเตอร์ที่ดีที่สุด

## 📈 ผลลัพธ์

### Performance Summary:
- **Obesity**: ทำนายได้ดีที่สุด (108 ถูก, 10 พลาด)
- **Normal**: ทำนายได้ดี (117 ถูก, 15 พลาด)
- **Overweight**: พบความสับสนกับ Obesity (46 ถูก, 11 พลาด)
- **Underweight**: ข้อมูลน้อย ทำให้ทำนายได้ยาก (12 ถูก, 3 พลาด)

### ปัญหาที่พบ:
1. **Class Imbalance** - กลุ่ม Underweight มีข้อมูลน้อยที่สุด
2. **Boundary Confusion** - โมเดลแยกระหว่าง Overweight กับ Obesity ได้ยาก
3. **Feature Overlap** - บางครั้งสับสนระหว่าง Normal กับ Obesity

## 🚀 แนวทางการพัฒนาต่อ

ดูรายละเอียดใน [to_improved.md](to_improved.md) ซึ่งประกอบด้วย:

1. **Advanced SMOTE Techniques** - BorderlineSMOTE, ADASYN
2. **Class Weight Balancing** - จัดการข้อมูลไม่สมดุล
3. **Feature Engineering** - สร้าง feature ใหม่
4. **Advanced Ensemble** - Stacking, XGBoost, CatBoost
5. **Hyperparameter Optimization** - RandomizedSearchCV
6. **Cost-Sensitive Learning** - ปรับ weight ตามความสำคัญของ class
7. **Neural Network Approach** - ลอง Deep Learning
8. และอื่นๆ อีก 3 วิธี

## 📁 โครงสร้างไฟล์

```
obesity-classification/
│
├── obesity.ipynb              # Jupyter Notebook หลัก
├── obesity_dataset.csv        # ชุดข้อมูล
├── requirements.txt           # Python dependencies
├── to_improved.md            # แนวทางการพัฒนาโมเดล
└── README.md                 # ไฟล์นี้
```

## 🛠️ การติดตั้งและใช้งาน

### 1. Clone Repository
```bash
git clone https://github.com/68076012/obesity-classification.git
cd obesity-classification
```

### 2. สร้าง Virtual Environment
```bash
python -m venv obesity_env
source obesity_env/bin/activate  # Linux/Mac
# หรือ
obesity_env\Scripts\activate     # Windows
```

### 3. ติดตั้ง Dependencies
```bash
pip install -r requirements.txt
```

### 4. เปิด Jupyter Notebook
```bash
jupyter notebook obesity.ipynb
```

## 📦 Dependencies

- pandas
- numpy
- scikit-learn
- imbalanced-learn
- seaborn
- matplotlib

## 👥 ผู้จัดทำ

- **Student ID**: 68076012
- **Email**: 68076012@kmitl.ac.th

## 📝 License

This project is for educational purposes.

## 🙏 Acknowledgments

- Dataset source: [ระบุแหล่งที่มาของข้อมูล]
- Built with scikit-learn and imbalanced-learn

---

**หมายเหตุ**: โปรเจกต์นี้เป็นส่วนหนึ่งของการศึกษาด้าน Machine Learning และ Data Science
