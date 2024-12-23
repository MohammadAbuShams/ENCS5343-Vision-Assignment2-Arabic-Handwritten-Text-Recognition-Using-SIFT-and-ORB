# Arabic Handwritten Text Recognition Using SIFT and ORB

This project focuses on recognizing Arabic handwritten text using two feature extraction techniques: **SIFT (Scale-Invariant Feature Transform)** and **ORB (Oriented FAST and Rotated BRIEF)**. The methods were integrated into a **Visual Bag of Words (VBoW)** system and evaluated using the **XGBoost** classifier.

---

## Table of Contents
- [Introduction and Objective](#introduction-and-objective)
- [Background](#background)
- [Methodology](#methodology)
- [Results, Analysis, Visualization, and Comparison](#results-analysis-visualization-and-comparison)
- [Conclusion and Future Work](#conclusion-and-future-work)
- [Setup and Usage](#setup-and-usage)
- [Technologies Used](#technologies-used)
- [Contributors](#contributors)

---

## Introduction and Objective

Handwritten text recognition is challenging in computer vision, especially for Arabic due to its cursive nature and contextual variations. This project evaluates the performance of **SIFT** and **ORB** in recognizing Arabic handwritten text.

### Objectives
1. **Feature Extraction**: Use SIFT and ORB to find and describe features in Arabic handwriting images.
2. **Visual Dictionary**: Create a visual dictionary with k-means clustering for feature quantization.
3. **Classification**: Evaluate using an XGBoost classifier for accuracy, speed, and robustness under transformations.

---

## Background

Handwritten text recognition has been a long-standing challenge, especially for scripts like Arabic. Its complexity arises from the script's cursive nature and the variability in character representation. 

### Feature Extraction Methods:
- **SIFT**: Identifies robust and invariant key points that remain consistent under transformations like scaling and rotation.
- **ORB**: Provides a faster alternative to SIFT, focusing on binary descriptors for efficient computation.

### Visual Bag of Words (VBoW):
The VBoW model transforms visual features into a structured representation by clustering descriptors into "visual words," making them usable for classification models.

---

## Methodology

1. **Dataset Preparation and Preprocessing**:
   - **Dataset**: `isolated_words_per_user` containing 8,144 images of Arabic handwritten text from 82 users.
   - **Preprocessing**: Convert to grayscale and resize to 256x256 pixels for uniformity.

2. **Feature Extraction**:
   - **SIFT**: Extracts high-quality and invariant features.
   - **ORB**: Extracts more features but sacrifices precision for speed.

3. **Descriptor Matching**:
   - **SIFT**: Matched using BFMatcher with L2 norm.
   - **ORB**: Matched using BFMatcher with Hamming distance.

4. **Visual Bag of Words (VBoW)**:
   - **Clustering**: Descriptors clustered using k-means (clusters tested: 50, 100, 150, 300, 500).
   - **Histogram Generation**: Creates histograms of visual word occurrences for each image.

5. **Classification**:
   - **XGBoost**: Gradient boosting classifier trained on VBoW histograms.

6. **Robustness Testing**:
   - Transformations applied: Scaling, rotation, illumination, and noise.

---

## Results, Analysis, Visualization, and Comparison

### Performance Metrics

| Metrics                  | SIFT            | ORB             |
|--------------------------|-----------------|-----------------|
| **Time Efficiency**      | 510.8 seconds  | 580.68 seconds  |
| **Average Key Points**   | 83.32          | 313.84          |
| **Accuracy (Average)**   | 41.81%         | 31.19%          |
| **Best Classifier Accuracy** | 25.12%     | 17.43%          |

### Observations:
- SIFT is faster in execution and offers higher precision.
- ORB detects more key points but has reduced precision and increased processing time.
- BoVW histograms for SIFT are more focused, leading to better classification.

### Visualization:
- Histograms of visual word distributions show clearer separations for SIFT compared to ORB.

---

## Conclusion and Future Work

### Conclusion
This project demonstrated that:
- **SIFT** outperforms **ORB** in accuracy and robustness for Arabic handwritten text recognition.
- While ORB finds more key points, its quality is lower, impacting classification accuracy.

### Future Work
- Explore deep learning models like CNNs to enhance feature extraction.
- Expand the dataset to include more diverse handwriting styles.
- Optimize ORB for precision while retaining its speed advantages.

---

## Contributors

- [Mohammad AbuShams](https://github.com/MohammadAbuShams)
- [Joud Hijaz](https://github.com/JoudHijaz)
