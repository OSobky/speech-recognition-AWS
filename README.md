<h1 align="center">
  <a href="https://github.com/OSobky/speech-recognition-AWS">
    <img src="docs/images/sm_background.png" alt="Logo" width="80" height="80">
    <img src="docs/images/psoc6.png" alt="Logo" width="90" height="90">
  </a>
</h1>

<div align="center">
  Speech Recognition AWS end-to-end solution on PSoC6
  <br />
  <br />
</div>

<details open="open">
<summary>Table of Contents</summary>


- [Introduction](#introduction)
  - [Motivation](#motivation)   
  - [Objective](#objective)   
- [Methodology](#methodology)
   - [Prerequisites](#prerequisites)
- [Results](#results)
- [Conclusion](#about)
- [Authors & contributors](#authors--contributors)


</details>

---

# Introduction

## Motivation


Machine Learning has found numerous real-world applications, but bringing the power of
Machine Learning to resource constrained and small footprint is still challenging. There are
many limiting factors to the deployment of ML models on microcontrollers such as libraries
used by the model, programming the microcontroller, data types, and etc. We will use Infineon PSoC6 board, shown
in Figure below, to recognise Keywords "yes" & "no" using a Machine Learning Model created on
AWS. The application listens to its surroundings with a microphone and indicates when it has
detected a word by lighting an LED or displaying data on a screen, depending on the
capabilities of the device.

<div align="center">
<img src="docs/images/psoc6.webp" title="Login Page" width="50%"> 
</div>





## Objective 
<br>

Our objective is to showcase the AWS Infrastructure for generating a model, and to demonstrate
the capabilities of the Infineon PSoC6 microcontroller. In this project, we will try to overcome
some of the challenges faced in the deployment of ML models to microcontrollers by using
microcontroller libraries. We will build an end-to-end ML solution which will showcase the
whole ML pipeline from collecting and preprocessing data to building a model using AWS
SageMaker and then deploying the created model to a microcontroller. The following Diagram shows the project workflow.


<div align="center">
<img src="docs/images/IDP Diagram.png" title="Login Page" width="70%"> 
</div>


<br>

# Methodology 

In this section we will discuss the milestones/phases needed for this project.

#### The different phases for the end-to-end example are described below: ####

1. [Data Acquisition & storage in AWS](#1-data-acquisition--storage-in-aws)
2. [Data pre-processing using AWS SageMaker Processing Jobs](#2-data-preprocessing-using-aws-sagemaker-processing-jobs)
3. [Model Training in AWS](#3-model-training-in-aws)
4. [ML Deployment Web using AWS SageMaker End-Points](#4-ml-deployment-web-using-aws-sagemaker-end-points)
5. [ML Deployment PSoC6 board](#5-ml-deployment-psoc6-board)
    
    
<br>



<br>

Before disucssing the details of each milestone, let's discuss the prerequisites for this project.

## Prerequisites

The following list is essintial for this project:
- AWS Account
- PSoC6 6 board
- [ModusToolBox (MTB)](https://www.infineon.com/cms/en/design-support/tools/sdk/modustoolbox-software/)


In the following sections, we will disscuss how each milestone done and the challanges faced in each one.


## 1. Data Acquisition & storage in AWS


First we will speak about the datasets used and then how to move the data to AWS. 

### Speech Commands Dataset

For this project, we uses the [Speech Commands dataset](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html), a dataset created by Google which contains around 65,000 one-second long audios of 30 short words (Yes, No, and etc.) said by thousands different people. However, for development we uses Mini-Speech Commands Dataset (~1k) to develop the whole pipeline then re-run it with the original dataset.

### Amazon Simple Storage Service (Amazon S3)

One of our main focus is showcasing AWS services specifically AWS SageMaker. There is a multiple way to stream data to SageMaker, in this project we will be using Amazon Simple Storage Service (Amazon S3) as our storage on AWS as it is the best for our use-case. 

<br>
the following are multiple ways to upload data to S3 bucket.

- #### Downloaded dataset locally then upload the required data using the UI ####
- Use the command line locally 
- Use the EC2 if the internet is slow
- Write a lambda function to download, extract required files, then upload them


In this project, we uses the first way in the previous list to upload data to S3 bucket. Now we have our data in S3 bucket, then it's time to use AWS SageMaker for preprocessing, training, and deployment. 

<br>

## 2. Data preprocessing using AWS SageMaker Processing Jobs

In this section, we dissucs the preprocessing techniques and how to use Processing Jobs for preprocessing in SageMaker. However, Beforehand we will dissucss what is SageMaker.

### SageMaker

Amazon SageMaker is a fully managed machine learning service. With SageMaker, data scientists and developers can quickly and easily build and train machine learning models, and then directly deploy them into a production-ready hosted environment. It provides an integrated Jupyter authoring notebook instance for easy access to your data sources for exploration and analysis, so you don't have to manage servers.

- AWS SageMaker consisting of multiple features. which can be used for:
    - Label
    - Build/Develop
    - Train
    - Deploy
    - ...

For more indepth disucssion on AWS SageMaker feature please click [here](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html#whatis-features).

- We will be using also Processing Jobs, Training Jobs, Endpoints provided by SageMaker

One of the feature we will use for development is [SageMaker Studio](https://docs.aws.amazon.com/sagemaker/latest/dg/studio.html). Which is An integrated machine learning environment where you can build, train, deploy, and analyze your models all in the same application.


The following diagram illustrates an overview of the workflow within SageMaker.

<div align="center">
<img src="docs/images/SageMaker-diagram.png" title="SageMaker-diagram" width="80%"> 
</div>

First we have the data inside of the S3 bucket, then we use SageMaker Studio for development. Moreover, we create a Processing Job to do the preprocessing as disscused before. Now, we have the processed data stored in the S3 bucket, we start training the model using Training Jobs which will save the created model as an artifact in the S3 bucket. In the next sections, we will dive deep into each step.


### Spectrograms


<div align="center">
<img src="docs/images/spectrogram.png" title="Login Page" width="70%"> 
</div>


The model doesn't take in raw audio sample data, instead it works with spectrograms which are two dimensional arrays that are made up of slices of frequency information, each taken from a different time window.

The recipe for creating the spectrogram data is that each frequency slice is created by running an FFT across a 30ms section of the audio sample data. The input samples are treated as being between -1 and +1 as real values (encoded as -32,768 and 32,767 in 16-bit signed integer samples).

This results in an FFT with 256 entries. Every sequence of six entries is averaged together, giving a total of 43 frequency buckets in the final slice. The results are stored as unsigned eight-bit values, where 0 represents a real number of zero, and 255 represents 127.5 as a real number.

Each adjacent frequency entry is stored in ascending memory order (frequency bucket 0 at data[0], bucket 1 at data[1], etc). The window for the frequency analysis is then moved forward by 20ms, and the process repeated, storing the results in the next memory row (for example bucket 0 in this moved window would be in data[43 + 0], etc). This process happens 49 times in total, producing a single channel image that is 43 pixels wide, and 49 rows high.

You can see the whole preporcessing procedure in [data-exploration.ipynb](data-exploration.ipynb)

<br>


### Processing Jobs


To analyze data and evaluate machine learning models on Amazon SageMaker, use Amazon SageMaker Processing. With Processing, you can use a simplified, managed experience on SageMaker to run your data processing workloads, such as feature engineering, data validation, model evaluation, and model interpretation. You can also use the Amazon SageMaker Processing APIs during the experimentation phase and after the code is deployed in production to evaluate performance.


<div align="center">
<img src="docs/images/processing-jobs.png" title="Login Page" width="70%"> 
</div>


The preceding diagram shows how Amazon SageMaker spins up a Processing job. Amazon SageMaker takes your script, copies your data from Amazon Simple Storage Service (Amazon S3), and then pulls a processing container. The processing container image can either be an Amazon SageMaker built-in image or a custom image that you provide. The underlying infrastructure for a Processing job is fully managed by Amazon SageMaker. Cluster resources are provisioned for the duration of your job, and cleaned up when a job completes. The output of the Processing job is stored in the Amazon S3 bucket you specified.


We use processing jobs for preprocessing in this project. You can see how to create and run a processing job in [preprocessing-job.ipynb](preprocessing-job.ipynb) file




<br>


## 3. Model Training in AWS
<br>

In this section, we will dissucs the model architcture and how to use Training Jobs for preprocessing in AWS SageMaker.
<br>

### Model architcture
<br>

This is a simple model comprising of a Convolutional 2D layer, a Fully Connected Layer or a MatMul Layer (output: logits) and a Softmax layer (output: probabilities) as shown below. Refer to the tiny_conv model architecture.



<div align="center">
<img src="docs/images/model_2.tflite.png" title="Login Page" width="20%"> 
</div>

This image was derived from visualizing the 'model_2.tflite' file in Netron

This doesn't produce a highly accurate model, but it's designed to be used as the first stage of a pipeline, running on a low-energy piece of hardware that can always be on, and then wake higher-power chips when a possible utterance has been found, so that more accurate analysis can be done. Additionally, the model takes in preprocessed speech input as a result of which we can leverage a simpler model for accurate results.

<br>

### Training Jobs


To train a model in SageMaker, you create a training job. The training job includes the following information:

- The URL of the Amazon Simple Storage Service (Amazon S3) bucket where you've stored the training data.
- The compute resources that you want SageMaker to use for model training. Compute resources are ML compute instances that are managed by SageMaker.
- The URL of the S3 bucket where you want to store the output of the job.
- The Amazon Elastic Container Registry path where the training code is stored.

The figure below shows the whole workflow for training and deployment using AWS SageMaker.


<br>

<div align="center">
<img src="docs/images/training-jobs.png" title="Login Page" width="70%"> 
</div>

<br>

There are multiple options for training algorithms:
- Built-in Algorithm
    - SageMaker provides dozens of built-in training algorithms and hundreds of pre-trained models. If one of these meets your needs, it's a great out-of-the-box solution for quick model training. 
- ##### Script Mode #####
    - You can submit custom Python code that uses TensorFlow, PyTorch, or Apache MXNet for model training.
- Docker container
    - Put your code together as a Docker image and specify the registry path of the image in a SageMaker
- AWS Marketplace
    - You can use algorithms published on AWS Marketplace by different entities
- Notebook instance 
    - Train in the notebook instance itself


<br>

We use the script mode with TensorFlow in this project. You can see how to create and run a training job in [training-job.ipynb](training-job.ipynb) file
 

### Model Evaluation and Testing

For training and testing tracking we used Tensorboard with SageMaker Studio. Please refer to the figures below for the training and test metrics. As the main goal of this project is not the model accuracy, we will not dive deep into the model evaluations.


<details>
<summary>Training/Testing accuracy  and Confusion Matrix</summary>
<br>


|                               Training accuracy                       |                         Testing accuracy                               |
| :-------------------------------------------------------------------: | :--------------------------------------------------------------------: |
| <img src="docs/images/training-accuracy.png" title="Home Page" width="100%"> | <img src="docs/images/Test-accuracy.png" title="Login Page" width="100%"> |
|                               Confusion Matrix                       |
| <img src="docs/images/cm.png" title="Confusion Matrix" width="100%"> |
</details>




<br>

## 4. ML Deployment Web using AWS SageMaker End-Points

As you can see in the the training and deployment figure above, the trained model now is in the S3 bucket (Model Artifact) and now we can use SageMaker to deploy the model. In this section we will discuss AWS SageMaker endpoints and the which one did we use. the following are the list of whole deployment methods provided by SageMaker:
- #### SageMaker real-time hosting services ####
- Serverless Inference
- SageMaker Asynchronous Inference
- SageMaker batch transform


We used the real-time hosting for inference. SageMaker SDK make it very easy to deploy the model. To deploy the model you only need to run the following command  `model.deploy()`. Please check [deploy-model.ipynb](depoly-model.ipynb) file to check how to create/delete an endpoint and how to inference from it.


<br>

## 5. ML Deployment PSoC6 board 

In this section, we will discuss how to deploy the created model to PSoC6 board. First let's have an overview of our goal. As stated before, we want to have a board that recognize `Yes` & `No` word. Moreover, the following list & diagram shows the workflow of the PSoC6 board when it hear a new word.

1. Data in: Getting data from the PSoC6 microphone and sending it to pre-processing
2. Pre-processing: Feature engineering
3. ML Inference: Using the deployed model on the board
4. Post-processing: Converting inference output to UART/LED

<div align="center">
<img src="docs/images/Micro-speech example.png" title="Login Page" width="70%"> 
</div>


Now, let's discuss how to deploy a model to the board. Deploying the model to the PSoc6 boards needs multiple steps, which are:

1. Create a TF Lite model from the Model artifact
    1. with Quantization
    2. Without Quantization
2. Generate a C Model 
3. Use pre-processing and post-processing C blocks
4. Deploy to PSoC6 using MTB




The preceeding diagram shows the whole pipeline done by the micro controller.

Before diving deep into how to deploy the model, we need to clarify the following. The main goal of this project is to showcase AWS and deploy the model to PSoC 6 board. Therefore, in step 4 we are using pre-processing and post-processing C blocks which were already implemented by Google example. However, we took a different approach for preprocessing (built-in processing function `tf.signal.stft`). As a result, the created model could not work with the C blocks which already been implemented. As a solution, we resued python script created by one of my colleagues which create the same model however with the same approach in the C blocks. The reused python scripts are in the [utils/keras_rewrite](utils/keras_rewrite) folder.

### Create TF Lite model 

Now lets disscuss the steps in more details. First step is to use the [TF Lite](https://www.tensorflow.org/lite) library to convert a TF model to TF Lite model. There are two ways to convert a model to TF Lite model, with [Quantization](https://en.wikipedia.org/wiki/Quantization) and without Qunatization. For deploying the model to a board it is better to work with Quantized TF Lite model since it smaller compared to the other way.


### Generate a C Model 

The second step to deploy the model is to convert the TF lite model to a C Model using xxd tool or use python script to convert it

### Use pre-processing and post-processing C blocks

You will find in the [C](C) folder the C blocks required for the deployment. which uses the C model and do a inference then ligth the led depends on the inference result. To include the C model in the embedding code, we need to change the [model.cpp](C/tensorflow/lite/micro/examples/micro_speech/micro_features.model.cpp) file 

> **[Disclaimer]**
> The C folder scripts was not created by me. I just reuse previously implemented C blocks by adding the created to model to the script

### Deploy to PSoC6 using MTB
Now we have all the required part, the C model, the preprocessing and postprocessing C blocks, we deployed the model to the PSoC6 board using [MTB](https://www.infineon.com/cms/en/design-support/tools/sdk/modustoolbox-software/)



# Results

The following video show the results of testing the deployed model on the PSoC6 board. Play the video with the sound on!

https://user-images.githubusercontent.com/36420198/176245891-519fdb49-fb55-4bc3-85fe-449075b0e5cd.mp4

As you can see, the board already recognize the two words `Yes` & `No`. When it hear `Yes` it have bright red light, when it hears `No` it have a dim red light, and when it hear a different word than `Yes` and `No` it turns off the light

## Challanges 

During this project, we faced multiple challanges. In this section we will disscuss the faced challlanges and how did we solve it.

### Model is too big 

First, we had a different model architecture than the one presenated. However, the problem was the converted TF lite model was too big to be deployed on the PSoC6 board. The non-quantized model wass 6.6 MB and the quantized one was 1.5 MB.  The first approach to try to solve this challange was cahnge the output dimension of the preporcessing block as it had (124, 129, 1) output features and changed it to  (49,40,1). However this is not decrease the model size alot. The second apporach was to change the model architecture as the first one had around 16 million parameters and the new one had 3 K paramaters. You can below the two architucures and differences.

<details>
<summary>Model architectures</summary>
<br>


|                               Old                       |                         New                              |
| :-------------------------------------------------------------------: | :--------------------------------------------------------------------: |
| <img src="docs/images/old model arch.png" title="Home Page" width="100%"> | <img src="docs/images/new model arch.png" title="Login Page" width="100%"> |

</details>

### Model Deployment on PSoC6

During the model deploymnet on PSoC6 using MTB, we faced an error however it did not say what was the error exactly, as the figure below shows.

<details>
<summary>Error</summary>
<img src="docs/images/mtb-error.png" title="Login Page" width="50%"> 

</details>

We had to check everything was working well to find the error. Moreover, the error that the pre-processing and post-processing blocks were producing different output feature size than the trained model. We had to change the model architecture slighty as you can see below to adapt to the C blocks

<details>
<summary> Model architecture change</summary>
<img src="docs/images/model change.png" title="Login Page" width="100%"> 

</details>


## Summary

In conclusion, we were able to showcase AWS which shows how easy is the ML development on AWS. Also, we were able to deploy the model on PSoC6 board. the project was done in five different milesotne which is the following:

1. [Data Acquisition & storage in AWS](#1-data-acquisition--storage-in-aws)
2. [Data pre-processing using AWS SageMaker Processing Jobs](#2-data-preprocessing-using-aws-sagemaker-processing-jobs)
3. [Model Training in AWS](#3-model-training-in-aws)
4. [ML Deployment Web using AWS SageMaker End-Points](#4-ml-deployment-web-using-aws-sagemaker-end-points)
5. [ML Deployment PSoC6 board](#5-ml-deployment-psoc6-board)






# Project assistance

I would like to thank Fizan Aslam & Philipp Van Kempen as my supervisors and supporting me through the whole project

Also I would like to thank [@atakeskinn](https://github.com/atakeskinn) for letting me reuse his work.


# Authors & contributors

The original setup of this repository is by [Omar Elsobky](https://github.com/OSobky).


