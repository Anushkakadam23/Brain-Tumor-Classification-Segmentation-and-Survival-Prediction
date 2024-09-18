import numpy as np
import cv2
import os
import glob
import random  
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import imutils

# Define directories
train_dir = "C:/survival prediction/transfer learning/Training"
test_dir = "C:/survival prediction/transfer learning/Testing"

def crop_img(img):
    """
    Finds the extreme points on the image and crops the rectangular out of them
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    ADD_PIXELS = 0
    new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()

    return new_img

# Process images and save cleaned versions
def process_and_save_images(input_dir, output_dir, IMG_SIZE=256):
    for dir in os.listdir(input_dir):
        save_path = os.path.join(output_dir, dir)
        path = os.path.join(input_dir, dir)
        image_dir = os.listdir(path)
        for img in tqdm(image_dir):
            image = cv2.imread(os.path.join(path, img))
            new_img = crop_img(image)
            new_img = cv2.resize(new_img, (IMG_SIZE, IMG_SIZE))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            cv2.imwrite(os.path.join(save_path, img), new_img)

# Run processing on training and testing data
process_and_save_images(train_dir, 'cleaned/Training')
process_and_save_images(test_dir, 'cleaned/Testing')

# Visualize the Number of Images Per Class with Different Colors and Counts on Bars
def plot_image_distribution(directory, title_suffix):
    class_names = []
    image_counts = []
    
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            image_count = len(os.listdir(subdir_path))
            class_names.append(subdir)
            image_counts.append(image_count)
    
    # Assign a different color to each class
    colors = sns.color_palette("dark", len(class_names))
    
    plt.figure(figsize=(10, 5))
    bars = plt.bar(class_names, image_counts, color=colors)
    plt.title(f'Number of Images per Class ({title_suffix})')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45)
    
    # Add counts on top of each bar
    for bar, count in zip(bars, image_counts):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 5, count, ha='center', va='bottom', weight='bold')
    
    plt.show()

# Plot distribution of training and testing datasets

plot_image_distribution('cleaned/Training', 'Training Data')


plot_image_distribution('cleaned/Testing', 'Testing Data')

# Display Sample Images from Each Class with Colored Labels
def plot_sample_images(directory, classes, num_images=5):
    plt.figure(figsize=(15, 10))
    
    colors = sns.color_palette("dark", len(classes))
    
    for idx, class_name in enumerate(classes):
        class_dir = os.path.join(directory, class_name)
        images = os.listdir(class_dir)
        for i in range(num_images):
            img_path = os.path.join(class_dir, random.choice(images))
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for plotting
            plt.subplot(len(classes), num_images, idx * num_images + i + 1)
            plt.imshow(img)
            plt.axis('off')
            if i == 0:
                plt.title(class_name, fontsize=16, color=colors[idx], weight='bold')
    
    plt.show()

# Get class names
classes = os.listdir('cleaned/Training')

# Display sample images from each class
plot_sample_images('cleaned/Training', classes)

# Load data with ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest',
    horizontal_flip=True,
    vertical_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Create a data generator
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(256, 256),
    batch_size=32,  
    class_mode='categorical',  
)

validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Get a batch of one image
sample_image_batch, _ = next(train_generator)

# Extract the original image from the batch before any further augmentation
original_image = sample_image_batch[0]  # The image is already rescaled

# Apply augmentation manually to get a distinct augmented image
augmented_image = train_datagen.random_transform(original_image)

# Create a figure to display the original and augmented image side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Display the original image with 'inferno' colormap
ax1.set_title('Original Image', fontsize=16)
ax1.imshow(tf.image.rgb_to_grayscale(original_image), cmap='inferno')
ax1.axis('off')  # Hide axis for better visualization

# Display the augmented image with 'inferno' colormap
ax2.set_title('Augmented Image', fontsize=16)
ax2.imshow(tf.image.rgb_to_grayscale(augmented_image), cmap='inferno')
ax2.axis('off')  # Hide axis for better visualization

# Save the figure for your report
plt.savefig("original_and_augmented_image_report_inferno.png", bbox_inches='tight')

# Show the images
plt.show()


# Build the InceptionV3 model
IMAGE_SIZE = [256, 256]
inception = InceptionV3(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
for layer in inception.layers:
    layer.trainable = False

x = GlobalAveragePooling2D()(inception.output)
x = Dense(256, activation='relu')(x)
output = Dense(4, activation='softmax')(x)

model = Model(inputs=inception.input, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator,
    callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=0.001)],
    verbose=1,
    shuffle=True
)

# Evaluate model performance
train_results = model.evaluate(train_generator)
validation_results = model.evaluate(validation_generator)

print("Training Results:", dict(zip(model.metrics_names, train_results)))
print("Validation Results:", dict(zip(model.metrics_names, validation_results)))

# Plot accuracy and loss curves
plt.figure()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Generate confusion matrix
predictions = model.predict(validation_generator)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = validation_generator.classes
conf_matrix = confusion_matrix(true_labels, predicted_labels)
class_names = validation_generator.class_indices.keys()

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Generate classification report
report = classification_report(true_labels, predicted_labels, target_names=class_names)
print("Classification Report:\n", report)

# Generate model probabilities and associated predictions (for ensemble method)
inceptionv3_test_probabilities = model.predict(validation_generator)
inceptionv3_test_predictions = tf.argmax(inceptionv3_test_probabilities, axis=1)

# Save the model in keras format
model.save(r'C:/survival prediction/transfer learning/inception_v3_model.keras')


from tensorflow.keras.models import load_model

# Load the saved model
loaded_model = load_model(r'C:/survival prediction/transfer learning/inception_v3_model.keras')
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you already have the validation_generator defined
# Evaluate the model on the validation data
validation_results = loaded_model.evaluate(validation_generator)
print("Validation Results:", dict(zip(loaded_model.metrics_names, validation_results)))

# Generate predictions
predictions = loaded_model.predict(validation_generator)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = validation_generator.classes

# Generate classification report
report = classification_report(true_labels, predicted_labels, target_names=validation_generator.class_indices.keys())
print("Classification Report:\n", report)

# Plot confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)
class_names = validation_generator.class_indices.keys()

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
import matplotlib.pyplot as plt

# Assuming 'history' was your training history object
# Plot accuracy and loss curves
plt.figure()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Save the predictions and probabilities if needed
import numpy as np
np.save(r'C:/survival prediction/transfer learning/inception_v3_test_probabilities.npy', inceptionv3_test_probabilities)
np.save(r'C:/survival prediction/transfer learning/inception_v3_test_predictions.npy', inceptionv3_test_predictions)


###########################################################################

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import tensorflow as tf

# Define image size and load the MobileNetV2 model with pre-trained weights
IMAGE_SIZE = [256, 256]
mobilenet_v2 = MobileNetV2(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# Freeze the existing weights
for layer in mobilenet_v2.layers:
    layer.trainable = False

# Add your own classification head to the model
x = GlobalAveragePooling2D()(mobilenet_v2.output)
x = Dense(256, activation='relu')(x)
output = Dense(4, activation='softmax')(x)

# Create a new model with the MobileNetV2 base and your classification head
model = Model(inputs=mobilenet_v2.input, outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=0.001)],
    verbose=1,
    shuffle=True  # Keep shuffle=True to ensure randomness in training batches
)

# Evaluate model performance on training data
train_results = model.evaluate(train_generator)
print("Training Results:", dict(zip(model.metrics_names, train_results)))

# Evaluate model performance on validation data
validation_results = model.evaluate(validation_generator)
print("Validation Results:", dict(zip(model.metrics_names, validation_results)))

# Plot accuracy and loss curves
plt.figure()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Generate classification report
predictions = model.predict(validation_generator)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = validation_generator.classes

report = classification_report(true_labels, predicted_labels, target_names=validation_generator.class_indices.keys())
print("Classification Report:\n", report)

# Generate confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)
class_names = validation_generator.class_indices.keys()

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Save the model in keras format
model.save(r'C:/survival prediction/transfer learning/mobilenet_v2_model.keras')

# Generate model probabilities and associated predictions (for ensemble method or further analysis)
mobilenetv2_test_probabilities = model.predict(validation_generator)
mobilenetv2_test_predictions = tf.argmax(mobilenetv2_test_probabilities, axis=1)

# Save the predictions and probabilities if needed
import numpy as np
np.save(r'C:/survival prediction/transfer learning/mobilenetv2_test_probabilities.npy', mobilenetv2_test_probabilities)
np.save(r'C:/survival prediction/transfer learning/mobilenetv2_test_predictions.npy', mobilenetv2_test_predictions)

###########################################################################
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

IMAGE_SIZE = [256, 256]
resnet = tf.keras.applications.resnet.ResNet152(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# Freeze the existing weights
for layer in resnet.layers:
    layer.trainable = False

# Add your own classification head to the model
x = GlobalAveragePooling2D()(resnet.output)
x = Dense(256, activation='relu')(x)
output = Dense(4, activation='softmax')(x)

# Create a new model with the ResNet base and your classification head
model = Model(inputs=resnet.input, outputs=output)

# Compile the model (add optimizer, loss function, etc. based on your task)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=None,
    epochs=15,
    validation_data=validation_generator,
    validation_steps=None,
    verbose=1,
    callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=0.001)],
    shuffle=True
)

# Evaluate model performance on training data
train_results = model.evaluate(train_generator)
print("Training Results:", dict(zip(model.metrics_names, train_results)))

# Evaluate model performance on validation data
validation_results = model.evaluate(validation_generator)
print("Validation Results:", dict(zip(model.metrics_names, validation_results)))

# Plot accuracy and loss curves
plt.figure()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Generate classification report
predictions = model.predict(validation_generator)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = validation_generator.classes

report = classification_report(true_labels, predicted_labels, target_names=validation_generator.class_indices.keys())
print("Classification Report:\n", report)

# Generate confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)
class_names = validation_generator.class_indices.keys()

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Save the model in HDF5 format
model.save(r'C:/survival prediction/transfer learning/resnet_model.keras')


# Generate model probabilities and associated predictions (for ensemble method or further analysis)
resnet_test_probabilities = model.predict(validation_generator)
resnet_test_predictions = tf.argmax(resnet_test_probabilities, axis=1)

# Save the predictions and probabilities
np.save(r'C:/survival prediction/transfer learning/resnet_test_probabilities.npy', resnet_test_probabilities)
np.save(r'C:/survival prediction/transfer learning/resnet_test_predictions.npy', resnet_test_predictions)

#######################################################################################

from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import tensorflow as tf
import numpy as np

num_classes = 4

# Create a base model from VGG19
def create_Base_model_from_VGG19():  
    model_vgg19 = VGG19(
        weights='imagenet',  # Use ImageNet weights
        include_top=False,   # Exclude the fully connected layers
        input_shape=(256, 256, 3)  # Define input shape
    )
    # Freeze the base VGG19 layers
    for layer in model_vgg19.layers:
        layer.trainable = False
    return model_vgg19

# Add custom layers to the VGG19 base model
def add_custom_layers_vgg19():
    model_vgg19 = create_Base_model_from_VGG19()
    x = model_vgg19.output
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    predictions = Dense(num_classes, activation="softmax")(x)
    # Create the final model
    final_model = Model(inputs=model_vgg19.input, outputs=predictions)
    final_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return final_model

# Instantiate the final model
model = add_custom_layers_vgg19()

# Train the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=0.001)],
    verbose=1,
    shuffle=True
)

# Evaluate model performance on training data
train_results = model.evaluate(train_generator)
print("Training Results:", dict(zip(model.metrics_names, train_results)))

# Evaluate model performance on validation data
validation_results = model.evaluate(validation_generator)
print("Validation Results:", dict(zip(model.metrics_names, validation_results)))

# Plot accuracy and loss curves
plt.figure()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Generate classification report
predictions = model.predict(validation_generator)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = validation_generator.classes

report = classification_report(true_labels, predicted_labels, target_names=validation_generator.class_indices.keys())
print("Classification Report:\n", report)

# Generate confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)
class_names = validation_generator.class_indices.keys()

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Save the model in HDF5 format
model.save(r'C:/survival prediction/transfer learning/vgg19_model.keras')

# Generate model probabilities and associated predictions (for ensemble method or further analysis)
vgg19_test_probabilities = model.predict(validation_generator)
vgg19_test_predictions = tf.argmax(vgg19_test_probabilities, axis=1)

# Save the predictions and probabilities if needed
np.save(r'C:/survival prediction/transfer learning/vgg19_test_probabilities.npy', vgg19_test_probabilities)
np.save(r'C:/survival prediction/transfer learning/vgg19_test_predictions.npy', vgg19_test_predictions)

########################################################################
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming true_labels is already available from your validation generator
true_labels = validation_generator.classes

# Initialize DummyClassifier
dummy_clf = DummyClassifier(strategy="most_frequent")  # Strategy can be 'most_frequent', 'stratified', 'uniform', etc.
dummy_clf.fit(np.zeros_like(true_labels.reshape(-1, 1)), true_labels)  # Dummy data as input

# Predict using DummyClassifier
dummy_predictions = dummy_clf.predict(np.zeros_like(true_labels.reshape(-1, 1)))

# Generate classification report
print("Dummy Classifier Performance:")
print(classification_report(true_labels, dummy_predictions))

# Generate confusion matrix
conf_matrix = confusion_matrix(true_labels, dummy_predictions)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=validation_generator.class_indices.keys(), yticklabels=validation_generator.class_indices.keys())
plt.title('Confusion Matrix (Dummy Classifier)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Calculate accuracy
dummy_accuracy = accuracy_score(true_labels, dummy_predictions)
print(f'Dummy Classifier Accuracy: {dummy_accuracy:.4f}')

################################################


import tensorflow as tf

# Load the saved models
model_inceptionv3 = tf.keras.models.load_model(r'C:/survival prediction/transfer learning/inception_v3_model.keras')
model_mobilenetv2 = tf.keras.models.load_model(r'C:/survival prediction/transfer learning/mobilenet_v2_model.keras')
model_resnet = tf.keras.models.load_model(r'C:/survival prediction/transfer learning/resnet_model.keras')
model_vgg19 = tf.keras.models.load_model(r'C:/survival prediction/transfer learning/vgg19_model.keras')

print("Models loaded successfully.")


import numpy as np
import tensorflow as tf

# Load the saved model probabilities
inceptionv3_test_probabilities = np.load(r'C:/survival prediction/transfer learning/inception_v3_test_probabilities.npy')
mobilenetv2_test_probabilities = np.load(r'C:/survival prediction/transfer learning/mobilenetv2_test_probabilities.npy')
resnet_test_probabilities = np.load(r'C:/survival prediction/transfer learning/resnet_test_probabilities.npy')
vgg19_test_probabilities = np.load(r'C:/survival prediction/transfer learning/vgg19_test_probabilities.npy')

# Compute predictions
inceptionv3_test_predictions = np.argmax(inceptionv3_test_probabilities, axis=1)
mobilenetv2_test_predictions = np.argmax(mobilenetv2_test_probabilities, axis=1)
resnet_test_predictions = np.argmax(resnet_test_probabilities, axis=1)
vgg19_test_predictions = np.argmax(vgg19_test_probabilities, axis=1)

# Compute average probabilities
avg_probabilities = np.mean([
    inceptionv3_test_probabilities,
    mobilenetv2_test_probabilities,
    resnet_test_probabilities,
    vgg19_test_probabilities
], axis=0)

# Compute average ensemble predictions
avg_predictions = tf.argmax(avg_probabilities, axis=1)


#############################################

# Define weights for each model (adjust the weights based on model performance)
weights = [0.3, 0.3, 0.2, 0.2]

# Compute weighted average probabilities
weighted_avg_probabilities = np.sum([w * p for w, p in zip(weights, [
    inceptionv3_test_probabilities,
    mobilenetv2_test_probabilities,
    resnet_test_probabilities,
    vgg19_test_probabilities
])], axis=0)

# Compute weighted average ensemble predictions
weighted_avg_predictions = tf.argmax(weighted_avg_probabilities, axis=1)


#####################################

# Compute geometric mean probabilities
geometric_mean_probabilities = np.exp(
    np.mean(np.log([
        inceptionv3_test_probabilities,
        mobilenetv2_test_probabilities,
        resnet_test_probabilities,
        vgg19_test_probabilities
    ]), axis=0)
)

# Compute geometric mean ensemble predictions
geometric_mean_predictions = tf.argmax(geometric_mean_probabilities, axis=1)

######################################################
# Save the ensemble predictions if needed
np.save(r'C:/survival prediction/transfer learning/avg_predictions.npy', avg_predictions)
np.save(r'C:/survival prediction/transfer learning/weighted_avg_predictions.npy', weighted_avg_predictions)
np.save(r'C:/survival prediction/transfer learning/geometric_mean_predictions.npy', geometric_mean_predictions)
###########################################################

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming true_labels is already defined (e.g., from the validation generator)
true_labels = validation_generator.classes
# Extract class names from the validation generator
class_names = list(validation_generator.class_indices.keys())


# Simple Average Ensemble Evaluation
print("Classification Report (Simple Average):")
print(classification_report(true_labels, avg_predictions))

conf_matrix = confusion_matrix(true_labels, avg_predictions)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix (Simple Average)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Weighted Average Ensemble Evaluation
print("Classification Report (Weighted Average):")
print(classification_report(true_labels, weighted_avg_predictions))

conf_matrix = confusion_matrix(true_labels, weighted_avg_predictions)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix (Weighted Average)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Geometric Mean Ensemble Evaluation
print("Classification Report (Geometric Mean):")
print(classification_report(true_labels, geometric_mean_predictions))

conf_matrix = confusion_matrix(true_labels, geometric_mean_predictions)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix (Geometric Mean)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
#############################################################

from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, matthews_corrcoef, top_k_accuracy_score)

# Function to generate performance scores
def generate_performance_scores(y_true, y_pred, y_probabilities):
    model_accuracy = accuracy_score(y_true, y_pred)
    top_3_accuracy = top_k_accuracy_score(y_true, y_probabilities, k=3)
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    model_matthews_corrcoef = matthews_corrcoef(y_true, y_pred)

    print('=============================================')
    print(f'Performance Metrics:')
    print('=============================================')
    print(f'accuracy_score:\t\t{model_accuracy:.4f}')
    print(f'top_3_accuracy_score:\t{top_3_accuracy:.4f}')
    print(f'precision_score:\t{model_precision:.4f}')
    print(f'recall_score:\t\t{model_recall:.4f}')
    print(f'f1_score:\t\t{model_f1:.4f}')
    print(f'matthews_corrcoef:\t{model_matthews_corrcoef:.4f}')
    print('=============================================')

    performance_scores = {
        'accuracy_score': model_accuracy,
        'top_3_accuracy': top_3_accuracy,
        'precision_score': model_precision,
        'recall_score': model_recall,
        'f1_score': model_f1,
        'matthews_corrcoef': model_matthews_corrcoef
    }
    return performance_scores

import pandas as pd

inceptionv3_performance = generate_performance_scores(true_labels, inceptionv3_test_predictions, inceptionv3_test_probabilities)
mobilenetv2_performance = generate_performance_scores(true_labels, mobilenetv2_test_predictions, mobilenetv2_test_probabilities)
resnet_performance = generate_performance_scores(true_labels, resnet_test_predictions, resnet_test_probabilities)
vgg19_performance = generate_performance_scores(true_labels, vgg19_test_predictions, vgg19_test_probabilities)

avg_ensemble_performance = generate_performance_scores(true_labels, avg_predictions, avg_probabilities)
weighted_avg_ensemble_performance = generate_performance_scores(true_labels, weighted_avg_predictions, weighted_avg_probabilities)
geometric_mean_ensemble_performance = generate_performance_scores(true_labels, geometric_mean_predictions, geometric_mean_probabilities)

# Record metrics with DataFrame
performance_df = pd.DataFrame({
    'model_inception_v3': inceptionv3_performance,
    'model_mobilenet_v2': mobilenetv2_performance,
    'model_resnet152': resnet_performance,
    'model_vgg19': vgg19_performance,
    'average_ensemble': avg_ensemble_performance,
    'weighted_average_ensemble': weighted_avg_ensemble_performance,
    'geometric_mean_ensemble': geometric_mean_ensemble_performance
}).T

# Set Pandas options to display all columns and rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_rows', None)     # Show all rows
pd.set_option('display.width', None)        # Auto-detect the width of the display
pd.set_option('display.max_colwidth', None) # Show full content of each column
# View Performance DataFrame
performance_df
##########################################3

import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Function to compute inference time
def compute_inference_time(model, ds, sample_count, inference_runs=5):
    total_inference_times = []
    inference_rates = []

    for _ in range(inference_runs):
        start = time.perf_counter()
        model.predict(ds, verbose=0)
        end = time.perf_counter()

        # Compute total inference time
        total_inference_time = end - start

        # Compute inference rate
        inference_rate = total_inference_time / sample_count

        total_inference_times.append(total_inference_time)
        inference_rates.append(inference_rate)

    # Calculate average total inference time with uncertainty
    avg_inference_time = sum(total_inference_times) / len(total_inference_times)
    avg_inference_time_uncertainty = (max(total_inference_times) - min(total_inference_times)) / 2

    # Calculate average inference rate with uncertainty
    avg_inference_rate = sum(inference_rates) / len(inference_rates)
    avg_inference_rate_uncertainty = (max(inference_rates) - min(inference_rates)) / 2

    print('====================================================')
    print(f'Model:\t\t{model.name}')
    print(f'Inference Time:\t{round(avg_inference_time, 6)}s ± {round(avg_inference_time_uncertainty, 6)}s')
    print(f'Inference Rate:\t{round(avg_inference_rate, 6)}s/sample ± {round(avg_inference_rate_uncertainty, 6)}s/sample')
    print('====================================================')

    return avg_inference_time, avg_inference_rate

# Corrected paths using raw string literals
model_inceptionv3 = tf.keras.models.load_model(r'C:/survival prediction/transfer learning/inception_v3_model.keras')
model_mobilenetv2 = tf.keras.models.load_model(r'C:/survival prediction/transfer learning/mobilenet_v2_model.keras')
model_resnet = tf.keras.models.load_model(r'C:/survival prediction/transfer learning/resnet_model.keras')
model_vgg19 = tf.keras.models.load_model(r'C:/survival prediction/transfer learning/vgg19_model.keras')

# Debug: Confirm model loading
print("Models loaded successfully.")

# Ensure validation generator is defined and has data
print(f"Validation samples: {len(validation_generator)}")

# Calculate inference times
inceptionv3_inference = compute_inference_time(model_inceptionv3, validation_generator, len(validation_generator))
mobilenetv2_inference = compute_inference_time(model_mobilenetv2, validation_generator, len(validation_generator))
resnet_inference = compute_inference_time(model_resnet, validation_generator, len(validation_generator))
vgg19_inference = compute_inference_time(model_vgg19, validation_generator, len(validation_generator))

# Calculate ensemble inference time
ensemble_inference_time = (
    inceptionv3_inference[0] + mobilenetv2_inference[0] + resnet_inference[0] + vgg19_inference[0],
    inceptionv3_inference[1] + mobilenetv2_inference[1] + resnet_inference[1] + vgg19_inference[1]
)

print('====================================================')
print(f'Model:\t\tEnsemble')
print(f'Inference Time:\t{ensemble_inference_time[0]:.6f}s')
print(f'Inference Rate:\t{ensemble_inference_time[1]:.6f}s/sample')
print('====================================================')

# Define trade-off function
def dist(x1, x2, y1, y2):
    return np.sqrt(np.square(x2 - x1) + np.square(y2 - y1))

# List model names and scores
model_names = ['InceptionV3', 'MobileNetV2', 'ResNet152', 'VGG19', 'Average Ensemble', 'Weighted Average Ensemble', 'Geometric Mean Ensemble']

model_scores = [
    inceptionv3_performance['matthews_corrcoef'],
    mobilenetv2_performance['matthews_corrcoef'],
    resnet_performance['matthews_corrcoef'],
    vgg19_performance['matthews_corrcoef'],
    avg_ensemble_performance['matthews_corrcoef'],
    weighted_avg_ensemble_performance['matthews_corrcoef'],
    geometric_mean_ensemble_performance['matthews_corrcoef']
]

model_rates = [
    inceptionv3_inference[1],
    mobilenetv2_inference[1],
    resnet_inference[1],
    vgg19_inference[1],
    ensemble_inference_time[1],
    ensemble_inference_time[1],
    ensemble_inference_time[1]
]

# Compute trade-offs
ideal_inference_rate = 0.0001  # Desired inference time (hypothetical)
ideal_mcc = 1.0  # Max MCC

trade_offs = [dist(ideal_inference_rate, rate, ideal_mcc, score) for rate, score in zip(model_rates, model_scores)]

# View trade-off scores
print('Trade-Off Scores: Inference Rate vs. MCC')
for name, rate, score, trade in zip(model_names, model_rates, model_scores, trade_offs):
    print('---------------------------------------------------------')
    print(f'Model: {name}\nInference Rate: {rate:.5f} | MCC: {score:.4f} | Trade-Off: {trade:.4f}')

# View model with best trade-off score
best_model_index = np.argmin(trade_offs)
best_model_name = model_names[best_model_index]
best_model_trade_off = trade_offs[best_model_index]

print('=========================================================')
print(f'Best Model Based on Trade-Off:\t{best_model_name}\nTrade-Off Score:\t{best_model_trade_off:.4f}')
print('=========================================================')

###############################################
#load the model

# Generate predictions
inceptionv3_predictions = model_inceptionv3.predict(validation_generator)
inceptionv3_predicted_labels = np.argmax(inceptionv3_predictions, axis=1)

# Generate classification report
inceptionv3_report = classification_report(true_labels, inceptionv3_predicted_labels, target_names=class_names)
print("InceptionV3 Classification Report:\n", inceptionv3_report)

# Generate confusion matrix
inceptionv3_conf_matrix = confusion_matrix(true_labels, inceptionv3_predicted_labels)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(inceptionv3_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('InceptionV3 Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Generate predictions
mobilenetv2_predictions = model_mobilenetv2.predict(validation_generator)
mobilenetv2_predicted_labels = np.argmax(mobilenetv2_predictions, axis=1)

# Generate classification report
mobilenetv2_report = classification_report(true_labels, mobilenetv2_predicted_labels, target_names=class_names)
print("MobileNetV2 Classification Report:\n", mobilenetv2_report)

# Generate confusion matrix
mobilenetv2_conf_matrix = confusion_matrix(true_labels, mobilenetv2_predicted_labels)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(mobilenetv2_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('MobileNetV2 Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# Generate predictions
resnet_predictions = model_resnet.predict(validation_generator)
resnet_predicted_labels = np.argmax(resnet_predictions, axis=1)

# Generate classification report
resnet_report = classification_report(true_labels, resnet_predicted_labels, target_names=class_names)
print("ResNet Classification Report:\n", resnet_report)

# Generate confusion matrix
resnet_conf_matrix = confusion_matrix(true_labels, resnet_predicted_labels)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(resnet_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('ResNet Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# Generate predictions
vgg19_predictions = model_vgg19.predict(validation_generator)
vgg19_predicted_labels = np.argmax(vgg19_predictions, axis=1)

# Generate classification report
vgg19_report = classification_report(true_labels, vgg19_predicted_labels, target_names=class_names)
print("VGG19 Classification Report:\n", vgg19_report)

# Generate confusion matrix
vgg19_conf_matrix = confusion_matrix(true_labels, vgg19_predicted_labels)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(vgg19_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('VGG19 Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

