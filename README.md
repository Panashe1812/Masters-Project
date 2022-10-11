# AI and Robotics Masters-Project 
Project Report available @ https://tinyurl.com/y3sbz27a
Notebooks for training vgg and resent models using PyTorch

ABSTRACT
This project evaluates various methods that are used to explain the predictions of deep learning algorithms during the design and implementation phases. Vgg 16. Vgg19, resnet18 and resnet34 models are trained on the Fairface dataset achieving accuracies of 59%, 61%,63 and 64 %  respectively. The trained model is used to calculate the prediction score. Explainable AI methods such as LIME shap, gradCAM and saliency maps are applied to the test images. Although the methods use represent the important features in terms of pixel, this study takes the approach that a group of such pixels within a locality can be labelled in human understandable explanations e.g. if pixel on the eye contributed the most to the prediction as represented by a high intensity region in the case of gradCAM or a green patch in the case of lime. The analysis of the confusion matrix shows a high false negative rates for Latino, southeast Asian and middle east classes which are subject of the analysis completed in this project. Based on this analysis this project concludes that with the exception of shap the methods evaluated in this  study are effective for mitigating bias in face recognition algorithms 

1.2	Project Aims and Objectives
The aim of the project is to investigate the effectiveness of explainable AI methods used to detect and monitor algorithmic bias. This is achieved by estimating the strengths and weaknesses of various approaches that used to detect and mitigate bias in machine learning algorithmic design and implementation. 

H0: Existing methods used in XAI are effective in detecting and mitigating bias in machine learning design and implementation.

H1: Existing methods used in XAI are effective in detecting and mitigating bias in machine learning design and implementation.

Definition of terms:

Explainable A.I. methods: LIME, SHAPley, gradCAM, gradCAM++ and Saliency Maps.

Effective Method: the method is effective if can localise a class given an image, and if the given visualisations can discriminate between different classes and if can provide insights that allows the model to be debugged.

Bias: when one class has a lower error rate than other classes in the dataset.

Detect bias: discover or identify the presence of bias in machine learning.

Mitigating bias:  To reduce the extent to which machine learning solutions are biased.

Machine learning design and implementation- the process of selecting the data, processing it and using to train the model as well the continuous monitoring for model and data drift.



1.3	Objectives 
    •	Train CNN based models on Fairface dataset and evaluate the model’s performance.
    •	Evaluate the model’s overall performance on test data and per class performance.
    •	Estimate the data quality and representativeness for the Machine Learning task.
    •	Estimate the risk for bias in datasets and algorithm.
    •	Apply various model-agnostic techniques to detect and mitigate bias.
      o	SHAP
      o	LIME 
      o	Grad CAM
      o	Saliency Maps
      o	Autoencoder based RNN
      


