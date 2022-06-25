<h1 align="center">
  <a href="https://github.com/OSobky/speech-recognition-AWS">
    <!-- Please provide path to your logo here -->
    <img src="docs/images/sm_background.png" alt="Logo" width="80" height="80">
    <img src="docs/images/psoc6.png" alt="Logo" width="90" height="90">
  </a>
</h1>

<div align="center">
  Speech Recognition AWS end-to-end solution on PSoC6
  <br />
  <a href="#about"><strong>Explore the screenshots »</strong></a>
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
   - [Data Acquisition & storage in AWS](#data-acquisition--storage-in-aws)
   - [Model Training in AWS](#model-training-in-aws)
- [Results](#about)
- [Conclusion](#about)
- [Authors & contributors](#authors--contributors)
- [Acknowledgements](#acknowledgements)

</details>

---

# Introduction

## Motivation

<table><tr><td>

Machine Learning has found numerous real-world applications, but bringing the power of
Machine Learning to resource constrained and small footprint is still challenging. There are
many limiting factors to the deployment of ML models on microcontrollers such as libraries
used by the model, programming the microcontroller, data types, and etc. This IDP will be
completed in cooperation with XXXXXXX, we will use Infineon PSoC6 board, shown
in Figure below, to recognise Keywords "yes" & "no" using a Machine Learning Model created on
AWS. The application listens to its surroundings with a microphone and indicates when it has
detected a word by lighting an LED or displaying data on a screen, depending on the
capabilities of the device.

<div align="center">
<img src="docs/images/psoc6.webp" title="Login Page" width="50%"> 
</div>
<details>
<summary>Screenshots</summary>
<br>

> **[?]**
> Please provide your screenshots here.

|                               Home Page                               |                               Login Page                               |
| :-------------------------------------------------------------------: | :--------------------------------------------------------------------: |
| <img src="docs/images/screenshot.png" title="Home Page" width="50%"> | <img src="docs/images/screenshot.png" title="Login Page" width="50%"> |

</details>

</td></tr></table>



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

1. Data Acquisition & storage in AWS
    1. We will be using Speech Command dataset provided by Google
    2. The dataset will be stored in Amazon S3 Bucket.
2. Data pre-processing using AWS SageMaker Processing Jobs
3. ML Training using AWS SageMaker Training Jobs
4. ML Deployment Web using AWS SageMaker End-Points
5. ML Deployment PSoC6 board (Figure below)
    1. Data in: Getting data from the PSoC6 microphone and sending it to pre-processing
    2. Pre-processing: Feature engineering
    3. ML Inference: Using the deployed model on the board
    4. Post-processing: Converting inference output to UART/LED
    
<br>

<div align="center">
<img src="docs/images/Micro-speech example.png" title="Login Page" width="70%"> 
</div>

<br>

Before disucssing the details of each milestone, let's discuss the prerequisites for this project.

## Prerequisites

The following list is essintial for this project:
- AWS Account
- PSoC6 6 board
- Modus Tool Box (MTB)


In the following sections, we will disscuss how each milestone done and the challanges faced in each one.


##  Data Acquisition & storage in AWS


First we will speak about the datasets used and then how to move the data to AWS. 

### Speech Commands Dataset

For this project, we uses the Speech Commands dataset, a dataset created by Google which contains around 65,000 one-second long audios of 30 short words (Yes, No, and etc.) said by thousands different people. However, for development we uses Mini-Speech Commands Dataset (~1k) to develop the whole pipeline then re-run it with the original dataset.

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

## Data preprocessing using AWS SageMaker Processing Jobs

In this section, we dissucs the preprocessing techniques and how to use Processing Jobs for preprocessing in SageMaker. However, Beforehand we will dissucss what is SageMaker.

### SageMaker

- Fully managed ML service, consisting of multiple services. Used for:
    - Label
    - Build/Develop
    - Train
    - Deploy 

- Studio = Managed EC2 Instance (Virtual Machine) + Managed EBS Volume (Storage)

- we will be using also Processing Jobs, Training Jobs, Endpoints provided by SageMaker

we will use SageMaker Studio for the development. The following diagram illustrates the workflow within SageMaker

<div align="center">
<img src="docs/images/SageMaker-diagram.png" title="Login Page" width="70%"> 
</div>


### Spectrograms


<div align="center">
<img src="docs/images/spectrogram.png" title="Login Page" width="70%"> 
</div>


The model doesn't take in raw audio sample data, instead it works with spectrograms which are two dimensional arrays that are made up of slices of frequency information, each taken from a different time window.

The recipe for creating the spectrogram data is that each frequency slice is created by running an FFT across a 30ms section of the audio sample data. The input samples are treated as being between -1 and +1 as real values (encoded as -32,768 and 32,767 in 16-bit signed integer samples).

This results in an FFT with 256 entries. Every sequence of six entries is averaged together, giving a total of 43 frequency buckets in the final slice. The results are stored as unsigned eight-bit values, where 0 represents a real number of zero, and 255 represents 127.5 as a real number.

Each adjacent frequency entry is stored in ascending memory order (frequency bucket 0 at data[0], bucket 1 at data[1], etc). The window for the frequency analysis is then moved forward by 20ms, and the process repeated, storing the results in the next memory row (for example bucket 0 in this moved window would be in data[43 + 0], etc). This process happens 49 times in total, producing a single channel image that is 43 pixels wide, and 49 rows high.

<br>


### Processing Jobs

<div align="center">
<img src="docs/images/processing-jobs.png" title="Login Page" width="50%"> 
</div>


<br>


## Model Training in AWS

In this section, we will dissucs the model architcture and how to use Training Jobs for preprocessing in AWS SageMaker.
<br>

### Model architcture

This is a simple model comprising of a Convolutional 2D layer, a Fully Connected Layer or a MatMul Layer (output: logits) and a Softmax layer (output: probabilities) as shown below. Refer to the tiny_conv model architecture.



<div align="center">
<img src="docs/images/model_2.tflite.png" title="Login Page" width="50%"> 
</div>

This image was derived from visualizing the 'model_2.tflite' file in Netron

This doesn't produce a highly accurate model, but it's designed to be used as the first stage of a pipeline, running on a low-energy piece of hardware that can always be on, and then wake higher-power chips when a possible utterance has been found, so that more accurate analysis can be done. Additionally, the model takes in preprocessed speech input as a result of which we can leverage a simpler model for accurate results.

<br>

### Training Jobs

<div align="center">
<img src="docs/images/training-jobs.png" title="Login Page" width="50%"> 
</div>


<br>

## ML Deployment Web using AWS SageMaker End-Points
<br>

## ML Deployment PSoC6 board 


## Usage

> **[?]**
> How does one go about using it?
> Provide various use cases and code examples here.


## Project assistance

If you want to say **thank you** or/and support active development of Speech Recognition AWS end-to-end solution on PSoC6:

- Add a [GitHub Star](https://github.com/OSobky/speech-recognition-AWS-readme) to the project.
- Tweet about the Speech Recognition AWS end-to-end solution on PSoC6.
- Write interesting articles about the project on [Dev.to](https://dev.to/), [Medium](https://medium.com/) or your personal blog.

Together, we can make Speech Recognition AWS end-to-end solution on PSoC6 **better**!


## Authors & contributors

The original setup of this repository is by [Omar Elsobky](https://github.com/OSobky).

For a full list of all authors and contributors, see [the contributors page](https://github.com/OSobky/speech-recognition-AWS-readme/contributors).


## Acknowledgements

> **[?]**
> If your work was funded by any organization or institution, acknowledge their support here.
> In addition, if your work relies on other software libraries, or was inspired by looking at other work, it is appropriate to acknowledge this intellectual debt too.
