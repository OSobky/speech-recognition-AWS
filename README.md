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
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
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
completed in cooperation with Infineon Technologies, we will use Infineon PSoC6 board, shown
in Figure below, to recognise Keywords "yes" & "no" using a Machine Learning Model created on
AWS. The application listens to its surroundings with a microphone and indicates when it has
detected a word by lighting an LED or displaying data on a screen, depending on the
capabilities of the device.

<div align="center">
<img src="docs/images/psoc6.webp" title="Login Page" width="30%"> 
</div>
<details>
<summary>Screenshots</summary>
<br>

> **[?]**
> Please provide your screenshots here.

|                               Home Page                               |                               Login Page                               |
| :-------------------------------------------------------------------: | :--------------------------------------------------------------------: |
| <img src="docs/images/screenshot.png" title="Home Page" width="100%"> | <img src="docs/images/screenshot.png" title="Login Page" width="100%"> |

</details>

</td></tr></table>



## Objective 
<br>

Our objective is to showcase the AWS Infrastructure for generating a model, and to demonstrate
the capabilities of the Infineon PSoC6 microcontroller. In this project, we will try to overcome
some of the challenges faced in the deployment of ML models to microcontrollers by using
microcontroller libraries. We will build an end-to-end ML solution which will showcase the
whole ML pipeline from collecting and preprocessing data to building a model using AWS
Infrastructure and then deploying the created model to a microcontroller.

<br>

# Methodology 

#### The different phases for the end-to-end example are described below: ####


1. Data Acquisition & storage in AWS
    1. We will be using Speech Command dataset provided by Google
    2. The dataset will be stored in Amazon S3 Bucket.
2. Data pre-processing using AWS SageMaker Processing Jobs
3. ML Training using AWS SageMaker Training Jobs
4. ML Deployment Web using AWS SageMaker Inference
5. ML Deployment PSoC6 board
    1. Data in: Getting data from the PSoC6 microphone and sending it to pre-processing
    2. Pre-processing: Feature engineering
    3. ML Inference: Using the deployed model on the board
    4. Post-processing: Converting inference output to UART/LED


## Getting Started

### Prerequisites

> **[?]**
> What are the project requirements/dependencies?

### Installation

> **[?]**
> Describe how to install and get started with the project.

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
