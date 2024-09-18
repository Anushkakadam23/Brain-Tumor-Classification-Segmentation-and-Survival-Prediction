# Brain-Tumor-Classification-Segmentation-and-Survival-Prediction

This study provides a comprehensive exploration into the application of advanced deep learning 
and machine learning techniques for brain tumor classification, segmentation, and survival 
prediction using MRI data. The research employed transfer learning models—InceptionV3, 
MobileNetV2, ResNet152V2, and VGG19—with pre-trained weights from the ImageNet 
dataset to address the classification task.

Among these, the InceptionV3 model exhibited the highest accuracy, achieving 92.83%, with 
ensemble methods, particularly the weighted average ensemble, slightly surpassing it at 
92.98%. This finding underscored the potential of model combination for enhanced accuracy. 
In contrast, ResNet152V2, despite its considerable depth and complexity, underperformed with 
an accuracy of 68.95%, highlighting the critical importance of selecting models that are wellsuited to the dataset's specific characteristics. MobileNetV2, noted for its efficiency, achieved 
a commendable balance between accuracy and speed, registering 90.54%, thereby rendering it 
suitable for real-time applications.

For segmentation, the study developed and optimized both a customized 3D U-Net and a 3D 
Attention U-Net for processing volumetric MRI data. The 3D U-Net achieved a mean 
Intersection over Union (IoU) score of 74.54% on validation data, while the 3D Attention UNet, enhanced with attention mechanisms to focus on critical regions, attained a slightly lower 
mean IoU of 69.19%. The attention mechanisms proved particularly advantageous in improving 
the segmentation of complex and smaller tumor regions, such as the enhancing tumor and 
edema, despite some fluctuations observed during training.

Survival prediction was approached through traditional machine learning models, including 
Random Forest, Gradient Boosting, Support Vector Machines (SVM), and a Voting Classifier 
ensemble. The Random Forest model achieved the highest training accuracy at 71%, though its 
validation accuracy declined to 56%, indicating potential overfitting. The Voting Classifier 
demonstrated a more balanced performance with a validation accuracy of 56%, suggesting that 
ensemble methods could mitigate some overfitting issues observed in individual models. 
However, the overall performance was constrained by the simplicity of the dataset, which 
included only age, tumor class weights, and survival days as features.



Due to the lack of a single comprehensive dataset that integrates both detection and survival 
prediction aspects, two separate datasets were utilized. This separation introduces challenges in 
terms of model integration and performance evaluation but allows for a focused approach to 
address the complexities of both tasks comprehensively. The integration of these datasets into 
the project is critical for achieving the dual objectives of accurate brain tumor detection and 
reliable segmentation and survival prediction, ultimately contributing to advancements in 
neuro-oncology research and clinical practice.


The first dataset, obtained from Kaggle1, comprises 7,023 human brain MRI images classified 
into four distinct categories: glioma, meningioma, no tumor (indicating the absence of tumors), 
and pituitary tumors. This dataset provided a rich source of labeled images for training and 
evaluating our brain tumor detection model. The diversity of the data in terms of tumor types 
is crucial for developing a robust classification system capable of accurately identifying various 
forms of brain tumors.


![random](https://github.com/user-attachments/assets/625ce40d-e2b7-48f3-b48c-8418639c2212)


The second dataset, also sourced from Kaggle2, is dedicated to the Brain Tumor Segmentation 
and survival prediction, includes multimodal MRI scans stored as NIfTI files (.nii.gz). These 
scans encompass native (T1), post-contrast T1-weighted (T1Gd), T2-weighted (T2), and T2 
Fluid Attenuated Inversion Recovery (T2-FLAIR) volumes. Annotations for this dataset include GD-enhancing tumor (ET), peritumoral edema (ED), and necrotic and non-enhancing 
tumor core (NCR/NET), facilitating the development of our survival prediction model. The rich 
annotation details and multimodal imaging data allow for comprehensive feature extraction 
necessary for accurate survival analysis.


![Figure 2024-07-28 172700](https://github.com/user-attachments/assets/ba52c72b-2a96-4345-aaed-0e5464010a19)

The segmentation masks classified the 
brain regions into four distinct categories:
• Class 0: 'NOT tumor' – healthy brain tissue,
• Class 1: 'CORE' – necrotic or non-enhancing tumor core,
• Class 2: 'EDEMA' – the area of swelling surrounding the tumor,
• Class 3: 'ENHANCING' – the active, contrast-enhancing tumor region.


1- https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

2- https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation

