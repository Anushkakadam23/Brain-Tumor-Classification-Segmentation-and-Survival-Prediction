
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import segmentation_models_3D as sm
from keras.models import Model, load_model
from keras.layers import Input, Conv3D, MaxPooling3D, concatenate, Conv3DTranspose, Dropout
from keras.optimizers import Adam
from keras.metrics import MeanIoU
import pandas as pd
import glob
import os
import numpy as np

# Load images and masks
def load_img(img_dir, img_list):
    images = []
    for i, image_name in enumerate(img_list):
        if image_name.split('.')[1] == 'npy':
            image = np.load(os.path.join(img_dir, image_name)).astype(np.float32)
            images.append(image)
    images = np.array(images)
    return images

# Image loader generator
def imageLoader(img_dir, img_list, mask_dir, mask_list, batch_size):
    L = len(img_list)
    while True:
        batch_start = 0
        batch_end = batch_size
        while batch_start < L:
            limit = min(batch_end, L)
            X = load_img(img_dir, img_list[batch_start:limit])
            Y = load_img(mask_dir, mask_list[batch_start:limit])
            yield (X, Y)
            batch_start += batch_size
            batch_end += batch_size

# Directories for training and validation data
train_img_dir = r'C:/survival prediction/BraTS2020_TrainingData/train_val/train/images/'
train_mask_dir = r'C:/survival prediction/BraTS2020_TrainingData/train_val/train/masks/'
val_img_dir = r'C:/survival prediction/BraTS2020_TrainingData/train_val/val/images/'
val_mask_dir = r'C:/survival prediction/BraTS2020_TrainingData/train_val/val/masks/'

train_img_list = os.listdir(train_img_dir)
train_mask_list = os.listdir(train_mask_dir)
val_img_list = os.listdir(val_img_dir)
val_mask_list = os.listdir(val_mask_dir)

print("Training Images:", train_img_list)
print("Training Masks:", train_mask_list)

print("Validation Images:", val_img_list)
print("Validation Masks:", val_mask_list)

batch_size = 2

train_img_datagen = imageLoader(train_img_dir, train_img_list, train_mask_dir, train_mask_list, batch_size)
val_img_datagen = imageLoader(val_img_dir, val_img_list, val_mask_dir, val_mask_list, batch_size)

# Calculate class weights
columns = ['0', '1', '2', '3']
df = pd.DataFrame(columns=columns)

train_mask_list = sorted(glob.glob(r'C:/survival prediction/BraTS2020_TrainingData/train_val/train/masks/*.npy'))

for img in range(len(train_mask_list)):
    temp_image = np.load(train_mask_list[img])
    temp_image = np.argmax(temp_image, axis=3)
    val, counts = np.unique(temp_image, return_counts=True)
    counts_dict = {str(key): 0 for key in range(len(columns))}
    counts_dict.update(dict(zip(val.astype(str), counts)))
    df = pd.concat([df, pd.DataFrame([counts_dict])], ignore_index=True)

label_0 = df['0'].sum()
label_1 = df['1'].sum()
label_2 = df['2'].sum()
label_3 = df['3'].sum()
total_labels = label_0 + label_1 + label_2 + label_3
n_classes = 4

wt0 = round((total_labels / (n_classes * label_0)), 2) if label_0 != 0 else 0
wt1 = round((total_labels / (n_classes * label_1)), 2) if label_1 != 0 else 0
wt2 = round((total_labels / (n_classes * label_2)), 2) if label_2 != 0 else 0
wt3 = round((total_labels / (n_classes * label_3)), 2) if label_3 != 0 else 0

print(f"Weight for class 0: {wt0}")
print(f"Weight for class 1: {wt1}")
print(f"Weight for class 2: {wt2}")
print(f"Weight for class 3: {wt3}")

class_weights = [wt0, wt1, wt2, wt3]



kernel_initializer =  'he_uniform' #Try others if you want


################################################################
def simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS, num_classes):
#Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS))
    #s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
    s = inputs

    #Contraction path
    c1 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c1)
    p1 = MaxPooling3D((2, 2, 2))(c1)
    
    c2 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c2)
    p2 = MaxPooling3D((2, 2, 2))(c2)
     
    c3 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c3)
    p3 = MaxPooling3D((2, 2, 2))(c3)
     
    c4 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c4)
    p4 = MaxPooling3D(pool_size=(2, 2, 2))(c4)
     
    c5 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c5)
    
    #Expansive path 
    u6 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c6)
     
    u7 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c7)
     
    u8 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c8)
     
    u9 = Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c9)
     
    outputs = Conv3D(num_classes, (1, 1, 1), activation='softmax')(c9)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    #compile model outside of this function to make it flexible. 
    model.summary()
    
    return model


# Verify if the model is working properly
model = simple_unet_model(128, 128, 128, 3, 4)
print(model.input_shape)
print(model.output_shape)

# Adjust class weights
dice_loss = sm.losses.DiceLoss(class_weights=np.array(class_weights))
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

metrics = ['accuracy', sm.metrics.IOUScore(threshold=0.5)]
LR = 0.0001
optim = Adam(LR)

# Model initialization and training
model = simple_unet_model(128, 128, 128, 3, 4)
model.compile(optimizer=optim, loss=total_loss, metrics=metrics)

steps_per_epoch = len(train_img_list) // batch_size
val_steps_per_epoch = len(val_img_list) // batch_size

history = model.fit(train_img_datagen,
                    steps_per_epoch=steps_per_epoch,
                    epochs=100,
                    verbose=1,
                    validation_data=val_img_datagen,
                    validation_steps=val_steps_per_epoch)

# Save the model
save_dir = r'C:/survival prediction/BraTS2020_TrainingData/models'
os.makedirs(save_dir, exist_ok=True)
model.save(os.path.join(save_dir, 'brats_3d.hdf5'))
model.save(os.path.join(save_dir, 'brats_3d.keras'))

#plot the training and validation IoU and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

iou_score = history.history['iou_score']
val_iou_score = history.history['val_iou_score']

plt.plot(epochs, acc, 'y', label='Training IOU Score')
plt.plot(epochs, val_acc, 'r', label='Validation IOU Score')
plt.title('Training and validation IOU Score')
plt.xlabel('Epochs')
plt.ylabel('IOU Score')
plt.legend()
plt.show()


from keras.models import load_model

#Now, let us add the iou_score function we used during our initial training
my_model = load_model(r'C:\survival prediction\BraTS2020_TrainingData\models\brats_3d.hdf5', 
                      custom_objects={'dice_loss_plus_1focal_loss': total_loss,
                                      'iou_score':sm.metrics.IOUScore(threshold=0.5)})


#Now all set to continue the training process. 
history2=my_model.fit(train_img_datagen,
          steps_per_epoch=steps_per_epoch,
          epochs=1,
          verbose=1,
          validation_data=val_img_datagen,
          validation_steps=val_steps_per_epoch,
          )
#################################################

#For predictions you do not need to compile the model, so ...
my_model = load_model(r'C:\survival prediction\BraTS2020_TrainingData\models\brats_3d.hdf5', 
                      compile=False)


# Evaluate model on validation data
batch_size = 8
test_img_datagen = imageLoader(val_img_dir, val_img_list, val_mask_dir, val_mask_list, batch_size)
test_image_batch, test_mask_batch = next(test_img_datagen)

test_mask_batch_argmax = np.argmax(test_mask_batch, axis=4)
test_pred_batch = my_model.predict(test_image_batch)
test_pred_batch_argmax = np.argmax(test_pred_batch, axis=4)

IOU_keras = MeanIoU(num_classes=4)
IOU_keras.update_state(test_pred_batch_argmax, test_mask_batch_argmax)
print("Mean IoU =", IOU_keras.result().numpy())

# Visualize predictions
def visualize_predictions(test_img, test_mask, test_pred, n_slice=55):
    plt.figure(figsize=(12, 8))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(test_img[:, :, n_slice, 1], cmap='gray')
    plt.subplot(232)
    plt.title('Testing Label')
    plt.imshow(test_mask[:, :, n_slice])
    plt.subplot(233)
    plt.title('Prediction on test image')
    plt.imshow(test_pred[:, :, n_slice])
    plt.show()

visualize_predictions(test_image_batch[0], test_mask_batch_argmax[0], test_pred_batch_argmax[0])

# Calculate class-wise accuracy
def calculate_class_accuracies(y_true, y_pred, num_classes):
    class_accuracies = []
    for i in range(num_classes):
        class_i_mask = y_true == i
        class_i_accuracy = np.sum(y_pred[class_i_mask] == i) / np.sum(class_i_mask)
        class_accuracies.append(class_i_accuracy)
    return class_accuracies

class_accuracies = calculate_class_accuracies(test_mask_batch_argmax.flatten(), test_pred_batch_argmax.flatten(), 4)
for i, acc in enumerate(class_accuracies):
    print(f"Accuracy for class {i}: {acc * 100:.2f}%")

# Save predictions for validation data
validation_img_dir = r'C:/survival prediction/BraTS2020_ValidationData/input_valdata/images'
validation_mask_dir = r'C:/survival prediction/BraTS2020_ValidationData/input_valdata/masks'
os.makedirs(validation_mask_dir, exist_ok=True)
img_files = sorted([f for f in os.listdir(validation_img_dir) if f.endswith('.npy')])

for img_file in img_files:
    test_img = np.load(os.path.join(validation_img_dir, img_file))
    test_img_input = np.expand_dims(test_img, axis=0)
    test_prediction = model.predict(test_img_input)
    test_prediction_argmax = np.argmax(test_prediction, axis=4)[0, :, :, :]
    mask_file = os.path.join(validation_mask_dir, img_file.replace('image_', 'mask_'))
    np.save(mask_file, test_prediction_argmax)
    print(f"Saved predicted mask for {img_file} to {mask_file}")

print("Prediction and saving of masks completed.")
