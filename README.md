## Build a Customized Recommender System on Amazon SageMaker

## Summary 

Many speech recognition services, such as Amazon Transcribe, require knowledge of the language being spoken in each audio file in order to run a transcription job.  If the language is unknown or if you intend on transcribing a batch of audio files that vary in language from file to file, pre-determining the languages can be time consuming.  In this post, I will demonstrate how to train and deploy a spoken language classifier on SageMaker, using an open-source dataset, VoxForge, and integrate it with Transcribe to achieve automatic multilingual speech recognition.

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