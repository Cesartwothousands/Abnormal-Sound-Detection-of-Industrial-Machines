# Records of Abnormal Sound Detection of Industrial Machines

Related work of my NUIST ungraduated project *Abnormal Sound Detection of Industrial Machines Based on Optimized Time-frequency Transform*. The final Optimized Time-frequency Transform method and CNN classification model need to handle considerable audio data and large matrices on cloud servers with large enough memory and strong enough GPU. **The programs here are what I use to keep a record of my experiments locally and to help the open source community as much as I can, but the principle is basically the same.**

- 0_Tools contains basic tools function for me to print some basic graphs, basic Mel_spectrum output, data processing(from mp4 audio to 2D graph) and function for processing files.
- 1_Signal_Processing contains different time-frequency transform I choose for the project, not only the implementation methods but also compared methods. 
- 2_Model contains the CNN classification model and result processing functions I use in the project. 
- 3_Papers_reading contents are collected from journals and blogs for researchers' study only. References and network addresses have been provided. I add my personal notes with it. Please pay attention to the protection of intellectual property for special use.

## Abstract

**Abstract**: With the rapid development of the Industrial Internet, there is a growing demand for automatic detection of mechanical anomalies. Sound signals can reflect the running status of  equipment, however, there are some problems such as high noise and complex feature extraction. Therefore, my work presents an optimized time-frequency transform method which combines continuous wavelet transform and Mel scale to extract time-frequency spectrum, and then accurately  classify using the convolutional neural network to detect the abnormal sound of industrial machinery.

**Key words**: **Abnormal Sound Detection**; **Time Frequency Analysis**; **Continuous Wavelet Transforms**;  **Mel Scale**; **Convolutional Neural Network**

## Background

<img src="pic1.png" alt="pic1" style="zoom:67%;" />

Pic1 shows the normal working state of the water pump in the real factory environment
, manual anomaly detection will have problems such as high labor costs, inability to get real-time feedback, and subjective errors.

## Data

Data from **MIMII Dataset**: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection. Pic2 shows  time-domain and frequency-domain diagrams of valves, water pumps, fans and slide rails in normal working state.

![pic2](pic2.png)
