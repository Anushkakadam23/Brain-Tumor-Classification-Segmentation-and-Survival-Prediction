import os
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_3D as sm
from keras.models import Model, load_model
from keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate, Dropout, BatchNormalization, Activation, Lambda, LeakyReLU, Multiply, Add
from keras.optimizers import Adam
from keras.metrics import MeanIoU
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TerminateOnNaN
from tensorflow.keras import backend as K
from scipy.ndimage import rotate, zoom
from tensorflow.keras import mixed_precision

# Reset mixed precision to default (float32)
mixed_precision.set_global_policy('float32')

# Clear any previous sessions to reset the environment
K.clear_session()

# Function for 3D data augmentation
def augment_data_3d(X, Y):
    # Random rotation
    if np.random.rand() > 0.5:
        angle = np.random.uniform(-10, 10)
        X = rotate(X, angle, axes=(1, 2), reshape=False, mode='nearest')
        Y = rotate(Y, angle, axes=(1, 2), reshape=False, mode='nearest')
    
    # Random flipping
    if np.random.rand() > 0.5:
        X = np.flip(X, axis=1)
        Y = np.flip(Y, axis=1)
    
    if np.random.rand() > 0.5:
        X = np.flip(X, axis=2)
        Y = np.flip(Y, axis=2)
    
    return X, Y

# Function to load images
def load_img(img_dir, img_list):
    images = []
    for image_name in img_list:
        file_path = os.path.join(img_dir, image_name)
        if os.path.exists(file_path):
            image = np.load(file_path).astype(np.float32)
            images.append(image)
        else:
            print(f"File {file_path} does not exist.")
    return np.array(images)

# Data generator with augmentation
def imageLoader(img_dir, img_list, mask_dir, mask_list, batch_size, augment=False):
    L = len(img_list)
    while True:
        batch_start = 0
        batch_end = batch_size
        while batch_start < L:
            limit = min(batch_end, L)
            X = load_img(img_dir, img_list[batch_start:limit])
            Y = load_img(mask_dir, mask_list[batch_start:limit])
            if augment:
                X, Y = augment_data_3d(X, Y)
            yield (X, Y)
            batch_start += batch_size
            batch_end += batch_size

# Define the custom loss functions
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def tversky(y_true, y_pred, smooth=1, alpha=0.7):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)

def focal_tversky_loss(y_true, y_pred, gamma=0.75):
    tv = tversky(y_true, y_pred)
    return K.pow((1 - tv), gamma)

# Helper function for repeating elements with specified output shape
def repeat_elem(tensor, rep):
    return Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=4),
                  arguments={'repnum': rep},
                  output_shape=lambda s: (s[0], s[1], s[2], s[3], s[4] * rep))(tensor)

# Attention block definition
def attention_block(x, gating, inter_shape):
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(gating)

    phi_g = Conv3D(filters=inter_shape, kernel_size=1, strides=1, padding='same')(gating)
    theta_x = Conv3D(filters=inter_shape, kernel_size=3, strides=(
        shape_x[1] // shape_g[1],
        shape_x[2] // shape_g[2],
        shape_x[3] // shape_g[3]
    ), padding='same')(x)
    xg_sum = Add()([phi_g, theta_x])
    xg_sum = Activation('relu')(xg_sum)

    psi = Conv3D(filters=1, kernel_size=1, padding='same')(xg_sum)
    sigmoid_psi = Activation('sigmoid')(psi)

    upsampled_sigmoid_psi = UpSampling3D(size=(
        shape_x[1] // K.int_shape(sigmoid_psi)[1],
        shape_x[2] // K.int_shape(sigmoid_psi)[2],
        shape_x[3] // K.int_shape(sigmoid_psi)[3]
    ))(sigmoid_psi)

    upsampled_sigmoid_psi = repeat_elem(upsampled_sigmoid_psi, shape_x[4])
    attention_coeffs = Multiply()([upsampled_sigmoid_psi, x])
    output = Conv3D(filters=shape_x[4], kernel_size=1, strides=1, padding='same')(attention_coeffs)
    output = BatchNormalization()(output)
    return output

# Gating signal definition
def gating_signal(input, output_size, batch_norm=False):
    x = Conv3D(output_size, (1, 1, 1), padding='same')(input)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

# Attention U-Net model definition
def attention_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS, num_classes, batch_norm=True):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS))

    c1 = Conv3D(16, (3, 3, 3), activation=LeakyReLU(alpha=0.1), kernel_initializer='he_uniform', padding='same')(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv3D(16, (3, 3, 3), activation=LeakyReLU(alpha=0.1), kernel_initializer='he_uniform', padding='same')(c1)
    p1 = MaxPooling3D((2, 2, 2))(c1)

    c2 = Conv3D(32, (3, 3, 3), activation=LeakyReLU(alpha=0.1), kernel_initializer='he_uniform', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv3D(32, (3, 3, 3), activation=LeakyReLU(alpha=0.1), kernel_initializer='he_uniform', padding='same')(c2)
    p2 = MaxPooling3D((2, 2, 2))(c2)

    c3 = Conv3D(64, (3, 3, 3), activation=LeakyReLU(alpha=0.1), kernel_initializer='he_uniform', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv3D(64, (3, 3, 3), activation=LeakyReLU(alpha=0.1), kernel_initializer='he_uniform', padding='same')(c3)
    p3 = MaxPooling3D((2, 2, 2))(c3)

    c4 = Conv3D(128, (3, 3, 3), activation=LeakyReLU(alpha=0.1), kernel_initializer='he_uniform', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv3D(128, (3, 3, 3), activation=LeakyReLU(alpha=0.1), kernel_initializer='he_uniform', padding='same')(c4)
    p4 = MaxPooling3D((2, 2, 2))(c4)

    c5 = Conv3D(256, (3, 3, 3), activation=LeakyReLU(alpha=0.1), kernel_initializer='he_uniform', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv3D(256, (3, 3, 3), activation=LeakyReLU(alpha=0.1), kernel_initializer='he_uniform', padding='same')(c5)

    gating_6 = gating_signal(c5, 128, batch_norm)
    att_6 = attention_block(c4, gating_6, 128)
    u6 = UpSampling3D((2, 2, 2))(c5)
    u6 = concatenate([u6, att_6])
    c6 = Conv3D(128, (3, 3, 3), activation=LeakyReLU(alpha=0.1), kernel_initializer='he_uniform', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv3D(128, (3, 3, 3), activation=LeakyReLU(alpha=0.1), kernel_initializer='he_uniform', padding='same')(c6)

    gating_7 = gating_signal(c6, 64, batch_norm)
    att_7 = attention_block(c3, gating_7, 64)
    u7 = UpSampling3D((2, 2, 2))(c6)
    u7 = concatenate([u7, att_7])
    c7 = Conv3D(64, (3, 3, 3), activation=LeakyReLU(alpha=0.1), kernel_initializer='he_uniform', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv3D(64, (3, 3, 3), activation=LeakyReLU(alpha=0.1), kernel_initializer='he_uniform', padding='same')(c7)

    gating_8 = gating_signal(c7, 32, batch_norm)
    att_8 = attention_block(c2, gating_8, 32)
    u8 = UpSampling3D((2, 2, 2))(c7)
    u8 = concatenate([u8, att_8])
    c8 = Conv3D(32, (3, 3, 3), activation=LeakyReLU(alpha=0.1), kernel_initializer='he_uniform', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv3D(32, (3, 3, 3), activation=LeakyReLU(alpha=0.1), kernel_initializer='he_uniform', padding='same')(c8)

    gating_9 = gating_signal(c8, 16, batch_norm)
    att_9 = attention_block(c1, gating_9, 16)
    u9 = UpSampling3D((2, 2, 2))(c8)
    u9 = concatenate([u9, att_9])
    c9 = Conv3D(16, (3, 3, 3), activation=LeakyReLU(alpha=0.1), kernel_initializer='he_uniform', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv3D(16, (3, 3, 3), activation=LeakyReLU(alpha=0.1), kernel_initializer='he_uniform', padding='same')(c9)

    outputs = Conv3D(num_classes, (1, 1, 1))(c9)
    outputs = BatchNormalization()(outputs)
    outputs = Activation('softmax')(outputs)

    model = Model(inputs=[inputs], outputs=[outputs], name="Attention_UNet")
    model.summary()

    return model

# Training setup (directories)
train_img_dir = r'C:/survival prediction/BraTS2020_TrainingData/train_val/train/images/'
train_mask_dir = r'C:/survival prediction/BraTS2020_TrainingData/train_val/train/masks/'
val_img_dir = r'C:/survival prediction/BraTS2020_TrainingData/train_val/val/images/'
val_mask_dir = r'C:/survival prediction/BraTS2020_TrainingData/train_val/val/masks/'

train_img_list = os.listdir(train_img_dir)
train_mask_list = os.listdir(train_mask_dir)
val_img_list = os.listdir(val_img_dir)
val_mask_list = os.listdir(val_mask_dir)

batch_size = 2
train_img_datagen = imageLoader(train_img_dir, train_img_list, train_mask_dir, train_mask_list, batch_size, augment=True)
val_img_datagen = imageLoader(val_img_dir, val_img_list, val_mask_dir, val_mask_list, batch_size)

total_loss = focal_tversky_loss  # Use focal tversky loss as the loss function
metrics = ['accuracy', sm.metrics.IOUScore(threshold=0.5)]

LR = 0.0001
optim = Adam(LR)

# Model initialization and training
model = attention_unet_model(128, 128, 128, 3, 4)
model.compile(optimizer=optim, loss=total_loss, metrics=metrics)

steps_per_epoch = len(train_img_list) // batch_size
val_steps_per_epoch = len(val_img_list) // batch_size

# Define callbacks
checkpoint_path = r'C:/survival prediction/BraTS2020_TrainingData/models/best_model.keras'
log_path = r'C:/survival prediction/BraTS2020_TrainingData/training_log.csv'

callbacks = [
    ModelCheckpoint(checkpoint_path, monitor='val_loss', mode='min', verbose=1, save_best_only=True),
    CSVLogger(log_path, separator=',', append=True),
    TerminateOnNaN()
]

# Train the model for exactly 100 epochs
history = model.fit(train_img_datagen,
                    steps_per_epoch=steps_per_epoch,
                    epochs=100,
                    verbose=1,
                    validation_data=val_img_datagen,
                    validation_steps=val_steps_per_epoch,
                    callbacks=callbacks
                    )

# Save the model
save_dir = r'C:/survival prediction/BraTS2020_TrainingData/models'
os.makedirs(save_dir, exist_ok=True)
model.save(os.path.join(save_dir, 'brats_3d_attention_unet.keras'))
model.save(os.path.join(save_dir, 'brats_3d_attention_unet.hdf5'))

# Plot training and validation metrics
def plot_metrics(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    iou = history.history['iou_score']
    val_iou = history.history['val_iou_score']
    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(12, 8))
    plt.subplot(231)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(232)
    plt.plot(epochs, acc, 'y', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(233)
    plt.plot(epochs, iou, 'y', label='Training IOU')
    plt.plot(epochs, val_iou, 'r', label='Validation IOU')
    plt.title('Training and validation IOU')
    plt.xlabel('Epochs')
    plt.ylabel('IOU')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_metrics(history)

# Load the saved model
model_path = os.path.join(save_dir, 'brats_3d_attention_unet.hdf5')
my_model = load_model(model_path, custom_objects={
    'focal_tversky_loss': focal_tversky_loss,
    'iou_score': sm.metrics.IOUScore(threshold=0.5),
    'LeakyReLU': LeakyReLU  # Ensure LeakyReLU is included in custom_objects
})

# Evaluate the model
batch_size = 8
test_img_datagen = imageLoader(val_img_dir, val_img_list, val_mask_dir, val_mask_list, batch_size)
test_image_batch, test_mask_batch = next(test_img_datagen)

test_mask_batch_argmax = np.argmax(test_mask_batch, axis=4)
test_pred_batch = model.predict(test_image_batch)
test_pred_batch_argmax = np.argmax(test_pred_batch, axis=4)

IOU_keras = MeanIoU(num_classes=4)
IOU_keras.update_state(test_pred_batch_argmax, test_mask_batch_argmax)
print("Mean IoU =", IOU_keras.result().numpy())


# Visualize single prediction for a sample
def visualize_single_prediction(test_img, test_mask, test_pred, sample_index=0, n_slice=55):
    plt.figure(figsize=(12, 8))
    
    # Testing Image
    plt.subplot(1, 3, 1)
    plt.title('Testing Image')
    plt.imshow(test_img[sample_index, :, :, n_slice], cmap='gray')

    # Testing Label
    plt.subplot(1, 3, 2)
    plt.title('Testing Label')
    plt.imshow(test_mask[sample_index, :, :, n_slice])

    # Prediction on Test Image
    plt.subplot(1, 3, 3)
    plt.title('Prediction on Test Image')
    plt.imshow(test_pred[sample_index, :, :, n_slice])

    plt.show()

# Call the function to visualize 4 different predictions
for i in range(4):
    visualize_single_prediction(test_image_batch, test_mask_batch_argmax, test_pred_batch_argmax, sample_index=i, n_slice=55)


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
