## Build a Customized Recommender System on Amazon SageMaker

## Summary 

Amazon Transcribe is a fully managed transcription service, currently supporting 31 languages, that makes it easy to integrate speech-to-text capabilities into your application. If the language of your audio file is unknown or you intend on transcribing a batch of audio files that vary in languages, Transcribe currently has the ability to auto-detect the dominant language for any given audio file prior to running any transcription job. The feature, called Language ID, leverages a separate model that classifies the dominant language being spoken in the audio file and reports a confidence score for each of the possible transcription languages. While effective for most applications out-of-the-box, this model does not currently have the ability to be retrained using custom data. Training a custom language detection model allows you to hone the model to specific accents, background noise levels, speech distortions, and other characteristics unique to your use case. This project demonstrates how to train and deploy a custom spoken language classifier on SageMaker to automatically detect languages of audio files before being transcribed by Amazon Transcribe.

## Getting Started

[Create an Amazon SageMaker notebook instance](https://docs.aws.amazon.com/sagemaker/latest/dg/howitworks-create-ws.html) (a `ml.t2.medium` instance will suffice to run the notebooks for this project)

## Running Notebooks

There are four notebooks associated with this project:  
1. [data preparation.ipynb](data_preparation.ipynb)  
This notebook downloads the VoxForge dataset, prepares it for the classification task, and uploads the data to s3 for training.
2. [train_classifier.ipynb](train_classifier.ipynb)  
This notebook runs a SageMaker training job to train the language classifier using a custom docker image.
3. [evaluation.ipynb](evaluation.ipynb)  
This notebook deploys the trained model as a SageMaker model endpoint and runs numerous evaluation metrics on the test dataset.
4. [transcribe.ipynb](transcribe.ipynb)  
This notebook demonstrates an example transcription pipeline that uses your trained language classifier to automatically detect the language of audio files before being passed to Amazon Transcribe.