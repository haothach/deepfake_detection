# Deepfake Detection with Machine Learning

## Dataset

Source & provenance. Videos are derived from the Kaggle dataset “Deep Fake Detection (DFD) Entire Original Dataset” by Sanikat Tiwarekar. See the original source here: [Kaggle Dataset](https://www.kaggle.com/datasets/sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset).

This repo uses a curated subset and subject-disjoint split prepared from that source.

Distribution (Google Drive). The working copies for this project are hosted on Google Drive and must be downloaded before running the notebooks:

**Drive folder (download required):** [Google Drive](https://drive.google.com/drive/folders/1QhxwAu1dmYE-YLpQPJGamxm4A58s-P-T?usp=drive_link)

### Dataset Samples

**Raw frames (from video):**

![Raw Frames](https://res.cloudinary.com/dwt2cschx/image/upload/v1755594527/Screenshot_2025-08-19_160827_if6h0x.png)

**Cropped faces (detected with Haar Cascade):**

![Cropped Faces](https://res.cloudinary.com/dwt2cschx/image/upload/v1755594594/Screenshot_2025-08-19_160942_np59jc.png)

### Train/Test Split (subject-disjoint)

| Split | Real | Fake | Total | Identity IDs |
| ----: | ---: | ---: | ----: | ------------ |
| Train |  289 |  545 |   834 | 1–22         |
|  Test |   74 |  144 |   218 | 23–28        |


## Pipeline

1. **Prepare Data** (`prepare_data.ipynb`)  
   - Extract frames from videos  
   - Detect & crop faces using **Haar Cascade (eyes, face)**  
   - Save processed images  

2. **Transform & Build Model** (`transform_build_model.ipynb`)  
   - Feature extraction with **LBP**, **HOG**, and **HOG+LBP**  
   - Train ML models (**Linear Regression**, **SVC**, **Random Forest**)  
   - Evaluate performance  

## Models Used

- **Linear Regression**  
- **Support Vector Classifier (SVC)**  
- **Random Forest**  

---

## Acknowledgments

- Dataset: [Deep Fake Detection (DFD) Entire Original Dataset](https://www.kaggle.com/datasets/sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset) curated by **Sanikat Tiwarekar**  
- Haar Cascade classifiers from [OpenCV](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html)  
- Feature extraction methods (LBP, HOG) from [scikit-image](https://scikit-image.org/docs/stable/)  
- Machine learning models (Linear Regression, SVC, Random Forest) implemented using [scikit-learn](https://scikit-learn.org/stable/)  
