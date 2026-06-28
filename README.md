# Suitability Analysis of College Majors Using Supervised Learning


![Field](https://img.shields.io/badge/Field-Education%20Data%20Mining-blue)
![Methodology](https://img.shields.io/badge/Method-Supervised%20Learning-orange)

This research project, conducted at **National Chi Nan University**, applies supervised machine learning to help undergraduate students in the College of Management select the academic major that best fits their performance and potential.

## 📌 Motivation & Objectives
With the rise of interdisciplinary learning, students often face "major selection anxiety." This study aims to reduce the mismatch between student capabilities and their chosen majors by analyzing historical academic data.

* **The Problem**: Many students struggle to identify which major (Information Management, Finance, Business Administration, etc.) aligns with their academic strengths.
* **The Goal**: Build a predictive model that uses early-semester grades to suggest the most suitable academic path.

---

## 📌 Why KNN for Major Suitability?
In this study, we identified **KNN** as a key classifier because academic performance patterns tend to "cluster." Students with similar grades in core foundation courses (like Accounting or Programming) often excel in the same specialized majors.

### How KNN Works in This Project:
1. **Feature Space**: Each student is represented as a data point in a multi-dimensional space, where each dimension is a score from a specific core course.
2. **Similarity Measurement**: The model calculates the **Euclidean Distance** between a new student's grades and the historical data of seniors who have already completed their majors.
3. **Voting (K-Value)**: By looking at the *K* most similar "neighbors" (seniors), the model assigns the student to the major that is most frequent among those neighbors.

---

## 🛠️ Implementation Details

### 1. Data Processing for KNN
* **Data Collection**  We collected 100 data from current students through distributing surveys.
* **Feature Selection**: We focused on "Indicator Courses" that distinguish majors, such as *Intro to Programming* for IM majors and *Financial Management* for Finance majors.
* **Normalization (Crucial for KNN)**: Since KNN relies on distance, we normalized all grades to a consistent scale (0 to 1) to prevent courses with higher credit weights from dominating the distance calculation.

### 2. Hyperparameter Tuning
* **Optimal K-Value**: We performed cross-validation to find the best `K`. A small `K` might be sensitive to noise (outliers), while a large `K` might include students with different academic profiles.
* **Distance Metrics**: Evaluated different ways to measure "academic similarity," ensuring that the most relevant subjects had the proper impact on the prediction.

---

## 📊 Performance Analysis: KNN Results
* **Instance-Based Learning**: Unlike models that create general rules, KNN effectively captured the "unique academic footprints" of students in different tracks.
* **Interpretability**: KNN allowed us to explain to students: *"You are recommended for this major because your grade profile is 85% similar to successful students in this track."*
* **Comparative Advantage**: While we tested SVM and Random Forest, KNN provided a highly intuitive baseline for **personalized academic counseling**.


---

## 📊 Key Research Findings
* **High Predictive Accuracy**: The models successfully categorized students based on their aptitude, with **Random Forest** and **SVM** showing the most robust performance across cross-validation folds.
* **Indicator Analysis**: Found that performance in **"Intro to Programming"** was a primary indicator for the Information Management track, while **"Economics"** and **"Accounting"** were key for Finance and Business tracks.
* **Balanced Evaluation**: Used **Macro F1-scores** to ensure the model performed well even for majors with fewer students (addressing data imbalance).

---

## 🚀 Future Applications
1.  **Automated Advisory System**: Integrate the model into the university's course selection system to provide real-time suitability reports for freshmen.
2.  **Dynamic Tracking**: Update predictions every semester to help students pivot their academic focus based on their latest growth.
3.  **Career Path Mapping**: Extend the model to include internship and employment data to link academic suitability with career success.

---
