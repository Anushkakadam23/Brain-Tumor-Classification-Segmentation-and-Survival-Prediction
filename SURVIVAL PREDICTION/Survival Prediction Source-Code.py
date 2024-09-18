import os
import numpy as np
import nibabel as nib
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.stats import randint, uniform
from imblearn.over_sampling import SMOTE
import itertools
import seaborn as sns

# Constants
VOLUME_SLICES = 100
VOLUME_START_AT = 22

# Paths
train_img_dir = r'C:/survival prediction/BraTS2020_TrainingData/input_data/images'
train_t1_dir = r'C:/survival prediction/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
train_seg_dir = r'C:/survival prediction/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
validation_img_dir = r'C:/survival prediction/BraTS2020_ValidationData/input_valdata/images'
validation_mask_dir = r'C:/survival prediction/BraTS2020_ValidationData/input_valdata/masks'
validation_t1_dir = r'C:/survival prediction/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData'

# Load CSV files
train_csv_path = r'C:/survival prediction/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/survival_info.csv'
val_csv_path = r'C:/survival prediction/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData/survival_evaluation.csv'

train_df = pd.read_csv(train_csv_path)
val_df = pd.read_csv(val_csv_path)

# Encode 'Extent_of_Resection' for training and validation data
if 'Extent_of_Resection' in train_df.columns:
    label_encoder = LabelEncoder()
    train_df['Extent_of_Resection_Encoded'] = label_encoder.fit_transform(train_df['Extent_of_Resection'].fillna('NA'))

if 'ResectionStatus' in val_df.columns:
    label_encoder_val = LabelEncoder()
    val_df['ResectionStatus_Encoded'] = label_encoder_val.fit_transform(val_df['ResectionStatus'].fillna('NA'))

# Filter out rows that do not have corresponding images, masks, and seg files
train_df = train_df[train_df['Brats20ID'].apply(lambda x: os.path.exists(os.path.join(train_seg_dir, f'{x}/{x}_seg.nii')))]
val_df = val_df[val_df['BraTS20ID'].apply(lambda x: os.path.exists(os.path.join(validation_mask_dir, f'mask_{x[-3:]}.npy')))]

# Function to convert survival days
def convert_survival_days(survival_str):
    if 'ALIVE' in survival_str:
        days = int(re.search(r'\d+', survival_str).group()) + 19
    else:
        days = int(survival_str)
    return days

# Apply the function to the Survival_days column
train_df['Survival_days'] = train_df['Survival_days'].apply(convert_survival_days)

# Ensure the Survival_days column is numeric and handle non-numeric values
train_df['Survival_days'] = pd.to_numeric(train_df['Survival_days'], errors='coerce')
train_df = train_df.dropna(subset=['Survival_days'])  # Drop rows with NaN in Survival_days

# Visualize distribution of survival days (Boxplot)
plt.figure(figsize=(8, 6))
plt.boxplot(train_df['Survival_days'], vert=False, patch_artist=True)
plt.title('Boxplot of Survival Days in Training Data')
plt.xlabel('Survival Days')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()

# Helper functions
def get_mask_sizes_for_volume(image_volume, volume_slices, volume_start_at):
    totals = dict([(1, 0), (2, 0), (3, 0)])
    max_slices = image_volume.shape[2]

    for i in range(volume_start_at, min(volume_start_at + volume_slices, max_slices)):
        arr = image_volume[:, :, i].flatten()
        arr[arr == 4] = 3

        unique, counts = np.unique(arr, return_counts=True)
        unique = unique.astype(int)
        values_dict = dict(zip(unique, counts))
        for k in range(1, 4):
            totals[k] += values_dict.get(k, 0)
    return totals

def get_brain_size_for_volume(image_volume):
    total = 0
    for i in range(image_volume.shape[2]):
        arr = image_volume[:, :, i].flatten()
        image_count = np.count_nonzero(arr)
        total += image_count
    return total

# Extract features for training data
train_features = []
train_labels = []

for index, row in train_df.iterrows():
    img_id = row['Brats20ID']
    seg_path = os.path.join(train_seg_dir, f'{img_id}/{img_id}_seg.nii')
    t1_path = os.path.join(train_t1_dir, f'{img_id}/{img_id}_t1.nii')
    
    seg_data = nib.load(seg_path).get_fdata()
    t1_data = nib.load(t1_path).get_fdata()
    
    masks = get_mask_sizes_for_volume(seg_data, VOLUME_SLICES, VOLUME_START_AT)
    brain_vol = get_brain_size_for_volume(t1_data)

    necrotic_core_vol = masks[1] / brain_vol if brain_vol != 0 else 0
    edema_vol = masks[2] / brain_vol if brain_vol != 0 else 0
    enhancing_vol = masks[3] / brain_vol if brain_vol != 0 else 0

    extent_of_resection_encoded = row['Extent_of_Resection_Encoded']

    train_features.append([row['Age'], necrotic_core_vol, edema_vol, enhancing_vol, extent_of_resection_encoded])
    
    if row['Survival_days'] < 250:
        train_labels.append([1, 0, 0])
    elif 250 <= row['Survival_days'] < 450:
        train_labels.append([0, 1, 0])
    else:
        train_labels.append([0, 0, 1])

train_features = np.array(train_features)
train_labels = np.array(train_labels)

# Extract features for validation data
val_features = []
val_img_paths = []  # Store paths to validation images for visualization

for index, row in val_df.iterrows():
    img_id = row['BraTS20ID']
    mask_path = os.path.join(validation_mask_dir, f'mask_{img_id[-3:]}.npy')
    t1_path = os.path.join(validation_t1_dir, f'{img_id}/{img_id}_t1.nii')
    img_path = os.path.join(validation_img_dir, img_id)  # Path to the original image directory

    mask_data = np.load(mask_path)
    t1_data = nib.load(t1_path).get_fdata()
    
    masks = get_mask_sizes_for_volume(mask_data, VOLUME_SLICES, VOLUME_START_AT)
    brain_vol = get_brain_size_for_volume(t1_data)

    necrotic_core_vol = masks[1] / brain_vol if brain_vol != 0 else 0
    edema_vol = masks[2] / brain_vol if brain_vol != 0 else 0
    enhancing_vol = masks[3] / brain_vol if brain_vol != 0 else 0

    resection_status_encoded = row['ResectionStatus_Encoded']

    val_features.append([row['Age'], necrotic_core_vol, edema_vol, enhancing_vol, resection_status_encoded])
    val_img_paths.append(mask_path)  # Save the path for visualization

val_features = np.array(val_features)

# Prepare data for modeling
X = train_features
y = train_labels.argmax(axis=1)

# Hold out a validation set from the training data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Standardize the features
scaler = StandardScaler()
X_train_smote = scaler.fit_transform(X_train_smote)
X_val = scaler.transform(X_val)
val_features_scaled = scaler.transform(val_features)

# Plot correlation matrix between all features
corr_matrix = pd.DataFrame(X_train_smote, columns=['Age', 'Necrotic_Core_Vol', 'Edema_Vol', 'Enhancing_Vol', 'Extent_of_Resection']).corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Features')
plt.show()

# Remaining part of the code for model training and evaluation (RandomForest, GradientBoosting, SVM, VotingClassifier, DummyClassifier, etc.)
# Include all the machine learning modeling and evaluation code here as in your original script.

# Define a dictionary to map the categorical labels
label_map = {0: "Short", 1: "Medium", 2: "Long"}

# Random Forest Classifier
rf = RandomForestClassifier(random_state=42)

# Define parameter grid for RandomizedSearchCV
param_dist_rf = {
    'n_estimators': randint(50, 100),  # Adjusting number of trees
    'max_depth': randint(3, 10),  # Limiting depth of trees
    'min_samples_split': randint(5, 20),  # Increasing min samples for split
    'min_samples_leaf': randint(5, 20)  # Increasing min samples for leaf
}

# Randomized search with cross-validation
random_search_rf = RandomizedSearchCV(estimator=rf, param_distributions=param_dist_rf, n_iter=100, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
random_search_rf.fit(X_train_smote, y_train_smote)

# Best parameters and cross-validation score
print(f"Best parameters for Random Forest: {random_search_rf.best_params_}")
print(f"Best cross-validation score for Random Forest: {random_search_rf.best_score_}")

# Predict on hold-out validation set using the best estimator from RandomizedSearchCV
rf_best = random_search_rf.best_estimator_
rf_best.fit(X_train_smote, y_train_smote)
train_predictions_rf = rf_best.predict(X_val)
train_predictions_labels_rf = [label_map[p] for p in train_predictions_rf]

# Classification report for Random Forest on training set
print("Classification Report for Random Forest (Training Set):")
print(classification_report(y_train_smote, rf_best.predict(X_train_smote), target_names=["Short", "Medium", "Long"]))

# Classification report for hold-out validation set for Random Forest
print("Classification Report for Random Forest (Hold-out Validation Set):")
print(classification_report(y_val, train_predictions_rf, target_names=["Short", "Medium", "Long"]))

# Confusion Matrix for hold-out validation set for Random Forest
conf_matrix_rf = confusion_matrix(y_val, train_predictions_rf)
print("Confusion Matrix for Hold-out Validation Set (Random Forest):")
print(conf_matrix_rf)

# Calculate percentage accuracy for the hold-out validation set for Random Forest
accuracy_rf = accuracy_score(y_val, train_predictions_rf)
print(f"Accuracy for Hold-out Validation Set (Random Forest): {accuracy_rf * 100:.2f}%")

# Confusion Matrix for Random Forest
conf_matrix_rf = confusion_matrix(y_val, train_predictions_rf)
print("Confusion Matrix for Hold-out Validation Set (Random Forest):")
print(conf_matrix_rf)

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix_rf, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Random Forest")
plt.colorbar()
tick_marks = np.arange(3)
plt.xticks(tick_marks, ["Short", "Medium", "Long"], rotation=45)
plt.yticks(tick_marks, ["Short", "Medium", "Long"])

# Add text annotations
thresh = conf_matrix_rf.max() / 2.
for i, j in itertools.product(range(conf_matrix_rf.shape[0]), range(conf_matrix_rf.shape[1])):
    plt.text(j, i, format(conf_matrix_rf[i, j], 'd'),
             horizontalalignment="center",
             color="white" if conf_matrix_rf[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()

################################################################


# Gradient Boosting Classifier
gb = GradientBoostingClassifier(random_state=42)

# Define parameter grid for RandomizedSearchCV
param_dist_gb = {
    'n_estimators': randint(50, 100),
    'max_depth': randint(3, 10),
    'min_samples_split': randint(5, 20),
    'min_samples_leaf': randint(5, 20),
    'learning_rate': uniform(0.01, 0.1)
}

# Randomized search with cross-validation
random_search_gb = RandomizedSearchCV(estimator=gb, param_distributions=param_dist_gb, n_iter=100, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
random_search_gb.fit(X_train_smote, y_train_smote)

# Best parameters and cross-validation score for Gradient Boosting
print(f"Best parameters for Gradient Boosting: {random_search_gb.best_params_}")
print(f"Best cross-validation score for Gradient Boosting: {random_search_gb.best_score_}")

# Predict on hold-out validation set using the best estimator from RandomizedSearchCV for Gradient Boosting
gb_best = random_search_gb.best_estimator_
gb_best.fit(X_train_smote, y_train_smote)
train_predictions_gb = gb_best.predict(X_val)
train_predictions_labels_gb = [label_map[p] for p in train_predictions_gb]

# Classification report for Gradient Boosting on training set
print("Classification Report for Gradient Boosting (Training Set):")
print(classification_report(y_train_smote, gb_best.predict(X_train_smote), target_names=["Short", "Medium", "Long"]))

# Classification report for hold-out validation set for Gradient Boosting
print("Classification Report for Gradient Boosting (Hold-out Validation Set):")
print(classification_report(y_val, train_predictions_gb, target_names=["Short", "Medium", "Long"]))

# Confusion Matrix for hold-out validation set for Gradient Boosting
conf_matrix_gb = confusion_matrix(y_val, train_predictions_gb)
print("Confusion Matrix for Hold-out Validation Set (Gradient Boosting):")
print(conf_matrix_gb)

# Calculate percentage accuracy for the hold-out validation set for Gradient Boosting
accuracy_gb = accuracy_score(y_val, train_predictions_gb)
print(f"Accuracy for Hold-out Validation Set (Gradient Boosting): {accuracy_gb * 100:.2f}%")

# Confusion Matrix for Gradient Boosting
conf_matrix_gb = confusion_matrix(y_val, train_predictions_gb)
print("Confusion Matrix for Hold-out Validation Set (Gradient Boosting):")
print(conf_matrix_gb)

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix_gb, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Gradient Boosting")
plt.colorbar()
tick_marks = np.arange(3)
plt.xticks(tick_marks, ["Short", "Medium", "Long"], rotation=45)
plt.yticks(tick_marks, ["Short", "Medium", "Long"])

# Add text annotations
thresh = conf_matrix_gb.max() / 2.
for i, j in itertools.product(range(conf_matrix_gb.shape[0]), range(conf_matrix_gb.shape[1])):
    plt.text(j, i, format(conf_matrix_gb[i, j], 'd'),
             horizontalalignment="center",
             color="white" if conf_matrix_gb[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()


#################################################################################

# SVM Classifier
svm = SVC(probability=True, random_state=42)

# Define parameter grid for RandomizedSearchCV
param_dist_svm = {
    'C': uniform(0.1, 10),
    'gamma': ['scale', 'auto'],
    'kernel': ['linear', 'rbf']
}

# Randomized search with cross-validation
random_search_svm = RandomizedSearchCV(estimator=svm, param_distributions=param_dist_svm, n_iter=100, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
random_search_svm.fit(X_train_smote, y_train_smote)

# Best parameters and cross-validation score for SVM
print(f"Best parameters for SVM: {random_search_svm.best_params_}")
print(f"Best cross-validation score for SVM: {random_search_svm.best_score_}")

# Predict on hold-out validation set using the best estimator from RandomizedSearchCV for SVM
svm_best = random_search_svm.best_estimator_
svm_best.fit(X_train_smote, y_train_smote)
train_predictions_svm = svm_best.predict(X_val)
train_predictions_labels_svm = [label_map[p] for p in train_predictions_svm]


# Classification report for SVM on training set
print("Classification Report for SVM (Training Set):")
print(classification_report(y_train_smote, svm_best.predict(X_train_smote), target_names=["Short", "Medium", "Long"]))

# Classification report for hold-out validation set for SVM
print("Classification Report for SVM (Hold-out Validation Set):")
print(classification_report(y_val, train_predictions_svm, target_names=["Short", "Medium", "Long"]))

# Confusion Matrix for hold-out validation set for SVM
conf_matrix_svm = confusion_matrix(y_val, train_predictions_svm)
print("Confusion Matrix for Hold-out Validation Set (SVM):")
print(conf_matrix_svm)

# Calculate percentage accuracy for the hold-out validation set for SVM
accuracy_svm = accuracy_score(y_val, train_predictions_svm)
print(f"Accuracy for Hold-out Validation Set (SVM): {accuracy_svm * 100:.2f}%")

# Confusion Matrix for SVM
conf_matrix_svm = confusion_matrix(y_val, train_predictions_svm)
print("Confusion Matrix for Hold-out Validation Set (SVM):")
print(conf_matrix_svm)

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix_svm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix - SVM")
plt.colorbar()
tick_marks = np.arange(3)
plt.xticks(tick_marks, ["Short", "Medium", "Long"], rotation=45)
plt.yticks(tick_marks, ["Short", "Medium", "Long"])

# Add text annotations
thresh = conf_matrix_svm.max() / 2.
for i, j in itertools.product(range(conf_matrix_svm.shape[0]), range(conf_matrix_svm.shape[1])):
    plt.text(j, i, format(conf_matrix_svm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if conf_matrix_svm[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()




##############################################################################
# Ensemble - Voting Classifier
voting_clf = VotingClassifier(estimators=[
    ('rf', rf_best),
    ('gb', gb_best),
    ('svm', svm_best)
], voting='soft')

voting_clf.fit(X_train_smote, y_train_smote)

# Predict on hold-out validation set using VotingClassifier
train_predictions_voting = voting_clf.predict(X_val)
train_predictions_labels_voting = [label_map[p] for p in train_predictions_voting]

# Predict on provided validation set using VotingClassifier
val_predictions_voting = voting_clf.predict(val_features_scaled)
val_predictions_labels_voting = [label_map[p] for p in val_predictions_voting]

# Visualization of some of the validation masks and their predictions
for i in range(min(10, len(val_img_paths))):  # Show up to 5 examples
    mask_path = val_img_paths[i]
    mask_data = np.load(mask_path)
    plt.imshow(mask_data[:, :, mask_data.shape[2] // 2], cmap='gray')
    plt.title(f'Predicted: {val_predictions_labels_voting[i]}')
    plt.show()


# Classification report for Voting Classifier on training set
print("Classification Report for Voting Classifier (Training Set):")
print(classification_report(y_train_smote, voting_clf.predict(X_train_smote), target_names=["Short", "Medium", "Long"]))

# Classification report for hold-out validation set for Voting Classifier
print("Classification Report for Voting Classifier (Hold-out Validation Set):")
print(classification_report(y_val, train_predictions_voting, target_names=["Short", "Medium", "Long"]))

# Confusion Matrix for hold-out validation set for Voting Classifier
conf_matrix_voting = confusion_matrix(y_val, train_predictions_voting)
print("Confusion Matrix for Hold-out Validation Set (Voting Classifier):")
print(conf_matrix_voting)

# Calculate percentage accuracy for the hold-out validation set for Voting Classifier
accuracy_voting = accuracy_score(y_val, train_predictions_voting)
print(f"Accuracy for Hold-out Validation Set (Voting Classifier): {accuracy_voting * 100:.2f}%")

# Confusion Matrix for Voting Classifier
conf_matrix_voting = confusion_matrix(y_val, train_predictions_voting)
print("Confusion Matrix for Hold-out Validation Set (Voting Classifier):")
print(conf_matrix_voting)

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix_voting, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Voting Classifier")
plt.colorbar()
tick_marks = np.arange(3)
plt.xticks(tick_marks, ["Short", "Medium", "Long"], rotation=45)
plt.yticks(tick_marks, ["Short", "Medium", "Long"])

# Add text annotations
thresh = conf_matrix_voting.max() / 2.
for i, j in itertools.product(range(conf_matrix_voting.shape[0]), range(conf_matrix_voting.shape[1])):
    plt.text(j, i, format(conf_matrix_voting[i, j], 'd'),
             horizontalalignment="center",
             color="white" if conf_matrix_voting[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()

#################################################3
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Baseline model for comparison
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train_smote, y_train_smote)

# Predict on the hold-out validation set
baseline_predictions = baseline.predict(X_val)

# Generate the classification report
print("Classification Report for Dummy Classifier (Hold-out Validation Set):")
print(classification_report(y_val, baseline_predictions, target_names=["Short", "Medium", "Long"]))

# Generate the confusion matrix
conf_matrix_baseline = confusion_matrix(y_val, baseline_predictions)
print("Confusion Matrix for Hold-out Validation Set (Dummy Classifier):")
print(conf_matrix_baseline)

# Optional: Plot the confusion matrix for better visualization
import matplotlib.pyplot as plt
import itertools

plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix_baseline, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Dummy Classifier")
plt.colorbar()
tick_marks = np.arange(3)
plt.xticks(tick_marks, ["Short", "Medium", "Long"], rotation=45)
plt.yticks(tick_marks, ["Short", "Medium", "Long"])

# Add text annotations
thresh = conf_matrix_baseline.max() / 2.
for i, j in itertools.product(range(conf_matrix_baseline.shape[0]), range(conf_matrix_baseline.shape[1])):
    plt.text(j, i, format(conf_matrix_baseline[i, j], 'd'),
             horizontalalignment="center",
             color="white" if conf_matrix_baseline[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()
