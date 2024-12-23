# Mohammad AbuShams 1200549.
# Joud Hijaz 1200342.

import cv2
import numpy as np
import os
import glob
import time
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from tkinter import Tk, filedialog, Toplevel, Label, Canvas, Button
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.cluster import KMeans, MiniBatchKMeans
import matplotlib.pyplot as plt
from PIL import Image, ImageTk


# Load images from the dataset folder.
def load_images(dataset_path):
    images = []
    labels = []
    filenames = []
    if not os.path.exists(dataset_path):
        raise ValueError("The dataset folder does not exist!")

    image_paths = glob.glob(os.path.join(dataset_path, "**", "*.*"), recursive=True)
    print(f"Dataset Name: {os.path.basename(dataset_path)}")
    print(f"Number of Files Found: {len(image_paths)}\n")
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(cv2.resize(img, (256, 256)))
            labels.append(os.path.basename(os.path.dirname(path)))
            filenames.append(os.path.basename(path))
    return images, labels, filenames



# Extract features using SIFT or ORB.
def extract_features(img, method='sift'):
    if method == 'sift':
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(img, None)
    elif method == 'orb':
        orb = cv2.ORB_create(nfeatures=1000)
        kp, des = orb.detectAndCompute(img, None)
    else:
        raise ValueError("Method must be 'sift' or 'orb'")
    return kp, des


# For BoVW, gather all descriptors from all images for k-means training.
def gather_all_descriptors(images, method='sift'):
    desc_list = []
    all_desc = []
    for img in images:
        kp, des = extract_features(img, method)
        desc_list.append(des)
        if des is not None:
            all_desc.append(des)
    if len(all_desc) == 0:
        raise ValueError(f"No descriptors found for method={method}.")
    return desc_list, np.vstack(all_desc)


# Perform k-means clustering to build a visual vocabulary.
def build_visual_vocabulary(desc_matrix, k=50):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(desc_matrix)
    return kmeans


# Compute BoVW histograms for all images.
def compute_bovw_histograms(desc_list, kmeans):
    k = kmeans.n_clusters
    histograms = []
    for des in desc_list:
        if des is None or len(des) == 0:
            hist = np.zeros(k, dtype=np.float32)
        else:
            try:
                labels = kmeans.predict(des)
                hist, _ = np.histogram(labels, bins=np.arange(k + 1), density=True)
            except Exception as e:
                print(f"Error during k-means prediction: {e}")
                hist = np.zeros(k, dtype=np.float32)
        histograms.append(hist)
    return np.array(histograms, dtype=np.float32)


# Create BoVW dataset for training and evaluation.
def create_bovw_dataset(images, labels, method='sift', n_clusters=50):
    start_time = time.time()
    desc_list, big_desc_matrix = gather_all_descriptors(images, method=method)
    kmeans = build_visual_vocabulary(big_desc_matrix, k=n_clusters)
    hist_features = compute_bovw_histograms(desc_list, kmeans)
    execution_time = time.time() - start_time

    avg_keypoints = np.mean([0 if d is None else len(d) for d in desc_list])
    target_labels = np.array(labels)
    return hist_features, target_labels, execution_time, avg_keypoints, kmeans


# Calculate robustness and accuracy metrics.
def calculate_robustness_and_accuracy(images, method):
    variations = ["scaling", "rotation", "illumination", "noise"]
    robustness_results = {}
    accuracy_results = {}

    for variation in variations:
        matches_per_variation = []
        accuracies_per_variation = []

        for img in images:
            if variation == "scaling":
                modified_img = cv2.resize(img, None, fx=0.5, fy=0.5)
            elif variation == "rotation":
                rows, cols = img.shape
                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
                modified_img = cv2.warpAffine(img, M, (cols, rows))
            elif variation == "illumination":
                modified_img = cv2.convertScaleAbs(img, alpha=1.5, beta=50)
            elif variation == "noise":
                noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
                modified_img = cv2.add(img, noise)
            else:
                continue

            kp1, des1 = extract_features(img, method=method)
            kp2, des2 = extract_features(modified_img, method=method)

            if des1 is not None and des2 is not None:
                if method == "sift":
                    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
                else:  # ORB
                    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

                matches = matcher.match(des1, des2)
                matches_per_variation.append(len(matches))
                accuracy = (len(matches) / max(len(kp1), len(kp2))) * 100
                accuracies_per_variation.append(accuracy)
            else:
                matches_per_variation.append(0)
                accuracies_per_variation.append(0)

        robustness_results[variation] = np.mean(matches_per_variation)
        accuracy_results[variation] = np.mean(accuracies_per_variation)

    return robustness_results, accuracy_results


# Save predictions to a CSV file.
def save_predictions_to_csv(filenames, predictions, method, filename="testing_images.csv"):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if os.path.getsize(filename) == 0:
            writer.writerow(["Filename", "Method", "Predicted User"])
        for img_name, pred in zip(filenames, predictions):
            writer.writerow([img_name, method, pred])
    print(f"Results saved to {filename}")


# Display SIFT/ORB keypoints on an image.
def display_keypoints(img, keypoints, window_title, max_keypoints=50):
    if keypoints is None or len(keypoints) == 0:
        print(f"No keypoints to display for {window_title}.")
        return
    keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)[:max_keypoints]
    img_with_keypoints = cv2.drawKeypoints(
        img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    img_rgb = cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)

    keypoints_window = Toplevel(root)
    keypoints_window.title(window_title)

    canvas = Canvas(keypoints_window, width=256, height=256)
    canvas.pack()
    canvas.create_image(0, 0, anchor="nw", image=img_tk)
    canvas.image = img_tk
    print(f"Displaying {len(keypoints)} keypoints for {window_title}.")


# Plot BoVW Histogram for an Image.
def plot_bovw_histogram(hist, method='sift', image_identifier='Image'):
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(hist)), hist, color='skyblue')
    plt.title(f"BoVW Histogram for {method.upper()} - {image_identifier}")
    plt.xlabel("Visual Word Index")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


# Plot Average BoVW Histograms per Class.
def plot_average_bovw_histogram(hist_features, labels, label_encoder, method='sift'):
    unique_classes = np.unique(labels)
    avg_histograms = []

    for cls in unique_classes:
        cls_indices = np.where(labels == cls)[0]
        avg_hist = hist_features[cls_indices].mean(axis=0)
        avg_histograms.append(avg_hist)

    avg_histograms = np.array(avg_histograms)

    plt.figure(figsize=(20, 10))
    for idx, cls in enumerate(unique_classes):
        class_name = label_encoder.inverse_transform([cls])[0]
        plt.plot(avg_histograms[idx], label=class_name)

    plt.title(f"Average BoVW Histograms per Class for {method.upper()}")
    plt.xlabel("Visual Word Index")
    plt.ylabel("Average Frequency")

    plt.legend(
        bbox_to_anchor=(0.5, -0.1),
        loc='upper center',
        ncol=8,
        fontsize='small'
    )

    plt.tight_layout(pad=2.0)
    plt.show()


# Upload an image and predict the user using BoVW + XGBoost.
def upload_image():
    filename = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
    if not filename:
        return

    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Failed to load image.")
        return

    resized_img = cv2.resize(img, (256, 256))

    prediction_window = Toplevel(root)
    prediction_window.title("Image and Predictions")

    img_for_display = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2RGB)
    from PIL import Image, ImageTk
    img_pil = Image.fromarray(img_for_display)
    img_tk = ImageTk.PhotoImage(img_pil)

    canvas = Canvas(prediction_window, width=256, height=256)
    canvas.pack()
    canvas.create_image(0, 0, anchor="nw", image=img_tk)
    canvas.image = img_tk

    # SIFT predictions.
    sift_keypoints, sift_desc = extract_features(resized_img, method="sift")
    display_keypoints(resized_img, sift_keypoints, "SIFT Keypoints")

    if sift_desc is not None and sift_desc.size > 0:
        labels_sift = kmeans_sift.predict(sift_desc)
        sift_hist, _ = np.histogram(labels_sift, bins=np.arange(kmeans_sift.n_clusters + 1), density=True)
        sift_hist = sift_hist.reshape(1, -1)
        try:
            hist_scaled = scaler_sift.transform(sift_hist)
            sift_pred_label = sift_classifier.predict(hist_scaled)
            sift_predicted_user = label_encoder.inverse_transform(sift_pred_label)[0]
        except ValueError as ve:
            print(f"Error during scaling SIFT histogram: {ve}")
            sift_predicted_user = "Scaling Error"
    else:
        sift_predicted_user = "No SIFT descriptors detected"

    # ORB predictions.
    orb_keypoints, orb_desc = extract_features(resized_img, method="orb")
    display_keypoints(resized_img, orb_keypoints, "ORB Keypoints")

    if orb_desc is not None and orb_desc.size > 0:
        labels_orb = kmeans_orb.predict(orb_desc)
        orb_hist, _ = np.histogram(labels_orb, bins=np.arange(kmeans_orb.n_clusters + 1), density=True)
        orb_hist = orb_hist.reshape(1, -1)
        try:
            orb_hist_scaled = scaler_orb.transform(orb_hist)
            orb_pred_label = orb_classifier.predict(orb_hist_scaled)
            orb_predicted_user = label_encoder.inverse_transform(orb_pred_label)[0]
        except ValueError as ve:
            print(f"Error during scaling ORB histogram: {ve}")
            orb_predicted_user = "Scaling Error"
    else:
        orb_predicted_user = "No ORB descriptors detected"

    result_label = Label(
        prediction_window,
        text=(
            f"Results:\n"
            f"Predicted User (SIFT): {sift_predicted_user}\n"
            f"Predicted User (ORB): {orb_predicted_user}\n"
        ),
        font=("Helvetica", 12),
        justify="left",
    )
    result_label.pack(pady=10)


# Tune XGBoost parameters using RandomizedSearchCV.
def tune_xgb_parameters_fast(X_train, y_train, param_distributions=None, random_state=42):
    if param_distributions is None:
        param_distributions = {
             'n_estimators': [50, 100],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8],
        'colsample_bytree': [0.8, 1.0]
        }
    xgb = XGBClassifier(eval_metric='mlogloss', random_state=random_state)
    random_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_distributions,
        scoring='accuracy',
        n_iter=16,
        cv=3,
        n_jobs=-1,
        refit=True,
        random_state=random_state
    )
    random_search.fit(X_train, y_train)
    best_estimator = random_search.best_estimator_
    best_params = random_search.best_params_
    best_score = random_search.best_score_
    return best_estimator, best_params, best_score


# Tune the cluster count after feature extraction.
def tune_cluster_count_after_extraction(images, labels, method='sift',
                                        cluster_candidates=[50, 100, 150, 300, 500],
                                        random_state=42):
    from sklearn.cluster import MiniBatchKMeans

    desc_list, big_desc_matrix = gather_all_descriptors(images, method=method)

    max_descriptors = 10000
    if len(big_desc_matrix) > max_descriptors:
        np.random.seed(random_state)
        indices = np.random.choice(len(big_desc_matrix), max_descriptors, replace=False)
        desc_matrix_sub = big_desc_matrix[indices]
    else:
        desc_matrix_sub = big_desc_matrix

    best_k = None
    best_acc = 0.0

    print(f"\nTuning cluster count for {method}... Candidates = {cluster_candidates}")
    print(f"Using up to {len(desc_matrix_sub)} descriptors for KMeans.")
    for k in cluster_candidates:
        start_time = time.time()

        kmeans = MiniBatchKMeans(
            n_clusters=k,
            random_state=random_state,
            batch_size=2048,
            n_init=3,
            max_iter=100
        )
        kmeans.fit(desc_matrix_sub)

        hist_features = compute_bovw_histograms(desc_list, kmeans)

        X_train_tmp, X_test_tmp, y_train_tmp, y_test_tmp = train_test_split(
            hist_features, labels, test_size=0.3, random_state=random_state
        )

        scaler_tmp = StandardScaler()
        X_train_tmp = scaler_tmp.fit_transform(X_train_tmp)
        X_test_tmp = scaler_tmp.transform(X_test_tmp)

        xgb = XGBClassifier(eval_metric='mlogloss', random_state=random_state)
        xgb.fit(X_train_tmp, y_train_tmp)

        preds = xgb.predict(X_test_tmp)
        acc = accuracy_score(y_test_tmp, preds)
        elapsed = time.time() - start_time
        print(f"  k={k}, accuracy={acc * 100:.2f}%, time={elapsed:.2f}s")

        if acc > best_acc:
            best_acc = acc
            best_k = k

    print(f"Best cluster count for {method} = {best_k} (accuracy : {best_acc * 100:.2f}%)")
    return best_k


# MAIN.
if __name__ == "__main__":
    dataset_path = "isolated_words_per_user"

    print("Loading Dataset...")
    images, labels, filenames = load_images(dataset_path)

    if len(images) < 10:
        raise ValueError("Not enough images loaded!")

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    print("Extracting features using SIFT...")
    sift_features, sift_labels, sift_time, sift_avg_kp, kmeans_sift_initial = create_bovw_dataset(
        images, encoded_labels, method='sift', n_clusters=50
    )

    print("Extracting features using ORB...")
    orb_features, orb_labels, orb_time, orb_avg_kp, kmeans_orb_initial = create_bovw_dataset(
        images, encoded_labels, method='orb', n_clusters=50
    )

    print("\n--- Metrics for Feature Extraction ---")
    print(f"SIFT Time Efficiency: {sift_time:.2f} seconds")
    print(f"ORB Time Efficiency: {orb_time:.2f} seconds")
    print(f"SIFT Average Keypoints: {sift_avg_kp:.2f}")
    print(f"ORB Average Keypoints: {orb_avg_kp:.2f}")

    # Robustness & Accuracy.
    print("\n--- Robustness and Accuracy Metrics ---")
    sift_robustness, sift_accuracy_metrics = calculate_robustness_and_accuracy(images, "sift")
    orb_robustness, orb_accuracy_metrics = calculate_robustness_and_accuracy(images, "orb")

    print("\nSIFT Robustness and Accuracy:")
    for variation in sift_robustness.keys():
        print(f"  {variation.capitalize()}:")
        print(f"    Matches: {sift_robustness[variation]:.2f}")
        print(f"    Accuracy: {sift_accuracy_metrics[variation]:.2f}%")
    sift_avg_accuracy = np.mean(list(sift_accuracy_metrics.values()))
    print(f"\nAverage Accuracy for SIFT Variations: {sift_avg_accuracy:.2f}%")

    print("\nORB Robustness and Accuracy:")
    for variation in orb_robustness.keys():
        print(f"  {variation.capitalize()}:")
        print(f"    Matches: {orb_robustness[variation]:.2f}")
        print(f"    Accuracy: {orb_accuracy_metrics[variation]:.2f}%")
    orb_avg_accuracy = np.mean(list(orb_accuracy_metrics.values()))
    print(f"Average Accuracy for ORB Variations: {orb_avg_accuracy:.2f}%")

    # Plot BoVW Histogram for the first 5 SIFT images.
    print("\nPlotting BoVW Histograms for first 5 SIFT images...")
    for i in range(min(5, len(sift_features))):
        plot_bovw_histogram(
            hist=sift_features[i],
            method='sift',
            image_identifier=filenames[i]
        )

    # Plot BoVW Histogram for the first 5 ORB images.
    print("\nPlotting BoVW Histograms for first 5 ORB images...")
    for i in range(min(5, len(orb_features))):
        plot_bovw_histogram(
            hist=orb_features[i],
            method='orb',
            image_identifier=filenames[i]
        )

    # Plot Average BoVW Histograms per Class for SIFT.
    print("\nPlotting Average BoVW Histograms per Class for SIFT...")
    plot_average_bovw_histogram(
        hist_features=sift_features,
        labels=encoded_labels,
        label_encoder=label_encoder,
        method='sift'
    )

    # Plot Average BoVW Histograms per Class for ORB.
    print("\nPlotting Average BoVW Histograms per Class for ORB...")
    plot_average_bovw_histogram(
        hist_features=orb_features,
        labels=encoded_labels,
        label_encoder=label_encoder,
        method='orb'
    )


    print("\nSplitting data into training and testing sets...")
    # Split data and filenames for SIFT & ORB.
    X_train_sift_init, X_test_sift_init, y_train_sift_init, y_test_sift_init, filenames_train_sift_init, filenames_test_sift_init = train_test_split(
        sift_features, encoded_labels, filenames, test_size=0.3, random_state=42
    )
    X_train_orb_init, X_test_orb_init, y_train_orb_init, y_test_orb_init, filenames_train_orb_init, filenames_test_orb_init = train_test_split(
        orb_features, encoded_labels, filenames, test_size=0.3, random_state=42
    )

    # Fit and transform the scaler.
    scaler_sift_init = StandardScaler()
    X_train_sift_init = scaler_sift_init.fit_transform(X_train_sift_init)
    X_test_sift_init = scaler_sift_init.transform(X_test_sift_init)

    scaler_orb_init = StandardScaler()
    X_train_orb_init = scaler_orb_init.fit_transform(X_train_orb_init)
    X_test_orb_init = scaler_orb_init.transform(X_test_orb_init)

    # TUNE cluster count with MiniBatchKMeans (batch_size=2048).
    best_k_sift = tune_cluster_count_after_extraction(
        images, encoded_labels, method='sift',
        cluster_candidates=[50, 100, 150, 300, 500],
        random_state=42
    )
    best_k_orb = tune_cluster_count_after_extraction(
        images, encoded_labels, method='orb',
        cluster_candidates=[50, 100, 150, 300, 500],
        random_state=42
    )


    print("\n--- Re-building final BoVW with best k for SIFT & ORB ---")
    desc_list_sift, big_desc_matrix_sift = gather_all_descriptors(images, 'sift')
    kmeans_sift = MiniBatchKMeans(n_clusters=best_k_sift, random_state=42, batch_size=2048, n_init=5)
    kmeans_sift.fit(big_desc_matrix_sift)
    sift_features = compute_bovw_histograms(desc_list_sift, kmeans_sift)

    desc_list_orb, big_desc_matrix_orb = gather_all_descriptors(images, 'orb')
    kmeans_orb = MiniBatchKMeans(n_clusters=best_k_orb, random_state=42, batch_size=2048, n_init=5)
    kmeans_orb.fit(big_desc_matrix_orb)
    orb_features = compute_bovw_histograms(desc_list_orb, kmeans_orb)


    print("\nRe-splitting data with the best cluster counts for SIFT & ORB...")
    X_train_sift, X_test_sift, y_train_sift, y_test_sift, filenames_train_sift, filenames_test_sift = train_test_split(
        sift_features, encoded_labels, filenames, test_size=0.3, random_state=42
    )
    scaler_sift = StandardScaler()
    X_train_sift = scaler_sift.fit_transform(X_train_sift)
    X_test_sift = scaler_sift.transform(X_test_sift)

    X_train_orb, X_test_orb, y_train_orb, y_test_orb, filenames_train_orb, filenames_test_orb = train_test_split(
        orb_features, encoded_labels, filenames, test_size=0.3, random_state=42
    )
    scaler_orb = StandardScaler()
    X_train_orb = scaler_orb.fit_transform(X_train_orb)
    X_test_orb = scaler_orb.transform(X_test_orb)

    # XGBoost Tuning: SIFT.
    print("\nPerforming parameter tuning for SIFT XGBoost...")
    sift_param_distributions = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8],
        'colsample_bytree': [0.8, 1.0]
    }
    sift_best_model, sift_best_params, sift_best_score = tune_xgb_parameters_fast(
        X_train_sift, y_train_sift, param_distributions=sift_param_distributions
    )
    print(f"Best Params for SIFT: {sift_best_params}")

    # XGBoost Tuning: ORB.
    print("\nPerforming parameter tuning for ORB XGBoost...")
    orb_param_distributions = {
         'n_estimators': [50, 100],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8],
        'colsample_bytree': [0.8, 1.0]
    }
    orb_best_model, orb_best_params, orb_best_score = tune_xgb_parameters_fast(
        X_train_orb, y_train_orb, param_distributions=orb_param_distributions
    )
    print(f"Best Params for ORB: {orb_best_params}")


    # Final training & evaluation.
    print("\nUsing the best model from parameter tuning for final SIFT training/evaluation...")
    sift_classifier = sift_best_model

    print("Using the best model from parameter tuning for final ORB training/evaluation...")
    orb_classifier = orb_best_model

    print("\nEvaluating Final Classifiers with Tuned Parameters...")
    sift_predictions = sift_classifier.predict(X_test_sift)
    orb_predictions = orb_classifier.predict(X_test_orb)

    sift_accuracy = accuracy_score(y_test_sift, sift_predictions)
    orb_accuracy = accuracy_score(y_test_orb, orb_predictions)

    print(f"SIFT Classifier Accuracy (Tuned XGBoost): {sift_accuracy * 100:.2f}%")
    print(f"ORB Classifier Accuracy (Tuned XGBoost): {orb_accuracy * 100:.2f}%")

    average_accuracy = (sift_accuracy + orb_accuracy) / 2
    print(f"Average Accuracy (Tuned XGBoost): {average_accuracy * 100:.2f}%")

    print("Saving SIFT predictions to CSV...")
    sift_predicted_users = label_encoder.inverse_transform(sift_predictions)
    save_predictions_to_csv(filenames_test_sift, sift_predicted_users, "SIFT")

    print("Saving ORB predictions to CSV...")
    orb_predicted_users = label_encoder.inverse_transform(orb_predictions)
    save_predictions_to_csv(filenames_test_orb, orb_predicted_users, "ORB")

    print("\n=== Done! ===")


# Initialize the Tkinter root window.
root = Tk()
root.title("FAST Image Classification with SIFT & ORB (XGBoost + BoVW)")

upload_button = Button(
    root,
    text="Upload and Predict Image",
    command=upload_image,
    font=("Helvetica", 14),
    padx=10,
    pady=5
)
upload_button.pack(pady=20)

root.mainloop()
