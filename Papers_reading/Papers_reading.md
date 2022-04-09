# Papers reading

## 0. Review

1.  1 & 3： MIMII Dataset      will use it 
2.  5-6： 提出了一种基于无监督特征提取的方法
3.  

[Toc]

## 1. Dataset source

> [1] Harsh Purohit, Ryo Tanabe, Kenji Ichige, Takashi Endo, Yuki Nikaido, Kaori Suefusa, and Yohei Kawaguchi, “MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection,” arXiv preprint arXiv:1909.09347, 2019.


In this paper, we present a new dataset of industrial machine sounds that we call a sound dataset for malfunctioning industrial machine investigation and inspection (MIMII dataset). Normal sounds were recorded for different types of industrial machines (i.e., valves, pumps, fans, and slide rails), and to resemble a real-life scenario, various anomalous sounds were recorded(e.g., contamination, leakage, rotating unbalance, and rail damage). The purpose of releasing the MIMII dataset is to assist the machine-learning and signal-processing community with their development of automated facility maintenance.

### Key words

声学场景分类，异常检测，无监督异常声音检测

> 对自动机器检查的需求越来越大，这是因为需要更高质量的工厂设备维护。发现有故障的机器零件主要取决于现场工程师的经验，但由于检查请求的数量增加，目前缺乏现场专家。迫切需要一个有效和负担得起的解决方案来解决这个问题。
>
> 在过去的十年中，工业物联网和数据驱动技术已经彻底改变了制造业，并且已经采取了不同的方法来监控机械的状态。示例包括基于振动传感器的方法[1–4]、基于温度传感器的方法[5]和基于压力传感器的方法[6]。另一种方法是通过使用声学场景分类和事件检测技术来检测声音中的异常现象[7–13]。声学场景的分类和声学事件的检测已经取得了显著的进步，在这方面有许多有前途的最新研究[14–16]。很明显，大量开放基准数据集[17–20]的出现对于研究领域的发展至关重要。然而，据我们所知，在真实的工厂环境中，没有包含不同类型机器声音的公共数据集。
>
> In this paper, we introduce a new dataset of machine sounds under normal and anomalous operating conditions in real factory environments. We include the sound of four machine types—(i) valves, (ii) pumps, (iii) fans, and (iv) slide rails—and for each type of machine, we consider seven different product models. We assume that the main task is to find an anomalous condition of the machine during a 10-second sound segment in an unsupervised learning situation. In other words, only normal machine sounds can be used in the training phase, and we have to correctly distinguish between a normal machine sound and an abnormal machine sound in the test phase. The main contributions of this paper are as follows: (1) We created an open dataset for malfunctioning industrial machine investigation and inspection (MIMII), the first of its kind. We have released this dataset, and it is freely available for download at https://zenodo.org/record/3384388. This dataset contains 26,092 sound files for normal conditions of four different machine types. It also contains real-life anomalous sound files for each category of the machines. (2) Using our developed dataset, we have explored an autoencoder-based model for each type of machine with various noise conditions. These results can be taken as a benchmark to improve the accuracy of anomaly detection in the MIMII dataset.

### how to record

> The microphone array was kept at a distance of 50 cm from the machine (10 cm in the case of valves), and 10-second sound segments were recorded.

###  Content

![Dataset-content](F:\毕业论文\Industrial-Machine-Investigation-and-Inspection\Papers_reading\Dataset-content.png)

![image-20220212174914246](https://github.com/Cesartwothousands/Industrial-Machine-Investigation-and-Inspection/blob/main/Papers_reading/Dataset-content.png)

## 2. 机器学习基础---无监督学习之异常检测

> [2] https://www.cnblogs.com/ssyfj/p/12940077.html 

## 3. DCASE 2020 Task 2

> [3] https://dcase.community/challenge2020/task-unsupervised-detection-of-anomalous-sounds

### Description 

**异常声音检测 (ASD) 是识别目标机器发出的声音是正常还是异常的任务。**自动检测机械故障是第四次工业革命中的一项重要技术，包括基于人工智能 (AI) 的工厂自动化。通过观察机器的声音来及时检测机器异常可能对机器状态监测有用。

**该任务的主要挑战是在仅提供正常声音样本作为训练数据的情况下检测未知的异常声音**。在现实世界的工厂中，实际的异常声音很少发生并且高度多样化。因此，不可能故意制造和/或收集详尽的异常声音模式。这意味着我们必须检测在给定训练数据中未观察到的*未知异常声音。**这一点是工业设备的 ASD 与过去用于检测定义*的异常声音（如枪声或婴儿哭声）的监督 DCASE 任务之间的主要区别之一。

![Overview-of-ASD](F:\毕业论文\Industrial-Machine-Investigation-and-Inspection\Papers_reading\Overview-of-ASD.png)

![Overview-of-ASD](https://github.com/Cesartwothousands/Industrial-Machine-Investigation-and-Inspection/blob/main/Papers_reading/Overview-of-ASD.png)

## 4.  CN Patent ASD with DL

> [4] 西安邮电大学."一种基于深度学习的声音异常检测系统."CN110220593A.2019-09-10.

## 5. CN Patent 一种基于自监督特征提取的机械设备异常声音检测方法

> [5] 广东省科学院智能制造研究所."一种基于自监督特征提取的机械设备异常声音检测方法."CN113792597A.2021-12-14.

**十分不错，用的就是MIMII的数据集！！！**

### Bg

**随着工业生产自动化的迅速发展，工厂中机械设备正常运行对工业生产有着重要的作用，但工业设备的高度复杂性，导致人工难以检测出设备前期故障，从而造成重大经济损失，因此对机械设备的异常诊断研究有重要意义。目前基于振动信号的机械设备异常检测已被广泛应用。声音信号与振动信号类似，是反应设备运行状态的重要信息来源，并且声音信号具有采集方便、非接触测量、处理速度快等优点。通过对设备运行声音识别来达到监测设备运行状态的目的，并将检测结果用于各种下游任务，例如故障诊断和预测性维护。基于声音信号的机械设备异常检测有着广泛的应用前景，值得深入研究。**

目前对于机械设备异常声音检测大多基于信号处理的方法，首先对采集的声音信号进行预处理，然后对提取的特征进行检测。针对列车轴承轨边声学检测，提出了使用多普勒畸变对声音信号进行校正并采用核特征矩阵联合近似对角化的方法提取前校正后的声音信号特征，最后使用SVM进行故障诊断。提出了使用自适应多尺度多结构形态滤波、小波
阈值降噪方法和稀疏量分析相结合的方法实现对滚动轴承声学异常检测。提出了采用共振稀疏分解算法对原始声音信号进行降噪，提取信号瞬态冲击成分并使用小波包变换对信号进行分解，最终通过包络谱分析轴承故障。上述方法具有较高的准确率，但是在声音信号特征提取方面，需要对设备运行机理有充分的了解，依赖于信号处理技术和诊断经验。随着大数据时代的到来，基于数据驱动的方法也逐渐被应用在机械设备异常检测中。建立深层神经网络模型，可以直接从底层原始数据出发经过层层网络学习自适应地提取出声音信号特征，摆脱了对大量信号技术和诊断经验的依赖，具有更强的处理高维非线性数据的能力。提
出了直接使用发电机正常和故障声音训练BP神经网络达到检测发电机状态的目的。然而在现实中，工厂中机械设备故障率低、故障种类多且操作的环境相对复杂，很难获得的完整的异常声音样本，存在训练过程中设备异常声音样本不可用的难题。有人提出了使用one‑class  SVM进行异常声音检测。上述方法都具有良好地实用性，**但这些方法都是通过人工提取声音特征，造成声音信息的丢失。鉴于这种情况本发明使用自监督的方法进行声音特征提取，并最终进行异常声音检测。**

### Benefit

相较于人为设计的算法提取的特征，本发明是基于自监督特征提取的机械设备异常声音检测方法，可以直接从底层数据出发经过层层学习提取声音的深度特征，提高下一步自编码器网络的异常检测准确率，更适合做异常检测，准确率高，可应用于现实工厂机械设备异常检测任务。

![自监督特征提取](F:\毕业论文\Industrial-Machine-Investigation-and-Inspection\Papers_reading\自监督特征提取.png)

![自监督特征提取](https://github.com/Cesartwothousands/Industrial-Machine-Investigation-and-Inspection/blob/main/Papers_reading/自监督特征提取.png)

## 6. 基于自监督特征提取的机械异常声音检测

> [6] 薛英杰,陈颀,周松斌,刘忆森,韩威.基于自监督特征提取的机械异常声音检测研究[J/OL].激光与光电子学进展:1-16[2022-02-13].http://kns.cnki.net/kcms/detail/31.1690.TN.20210823.1139.026.html.

### Abstract

问题所在：基于声诊断的机械设备异常状态检测在工业自动化领域有重要意义。当前，无监督机械设备异常声音检测主要基于人工构造算法提取声音信号特征，再以此特征进一步进行异常检测，存在人为因素影响较大，人工提取方法通用性不强问题。

针对此问题，本文提出一种自监督特征提取新方法，并将此特征输入自编码器（AE）进行机械设备异常声音检测。本方法首先将声音样本转换为时频谱图，采用设备正常声音时频谱图作为训练样本，然后使用正常时频谱图和人为构造异常时频谱图构建自监督特征提取器（SSFE），以 SSFE 提取的正常样本特征训练 AE，实现无监督机械设备异常声识别。使用 MIMII 公开数据集进行实验，结果表明所提方法能够自适应提取滑轨、阀门、水泵和风扇等四种机械设备的声音特征，最终获得的 AUC 检测结果分别为 95.0%、92.7%、88.2%和 78.0%，相对于线性声谱图（Line Spectrum）、对数梅尔谱（Log-mel）、梅尔频率倒谱系数（MFCCs）等人工特征提取方法有显著提升。

### Bg

随着工业生产自动化的迅速发展，工厂中机械设备正常运行对工业生产有着重要的作用，因此对机械设备的异常诊断研究有重要意义。目前基于振动信号的机械设备异常检测已被广泛应用。声音信号与振动信号类似，是反应设备运行状态的重要信息来源，并且声音信号具有采集方便、非接触测量、处理速度快等优点[1]。机械设备异常声音是指设备处于异常状态运行时发出的声音，通过对机械设备运行声音识别来达到检测设备运行状态[2-4]，具有广泛的应用前景和重要的研究意义。龙磊[2]等人针对列车轴承轨边声学检测，提出了使用多普勒畸变对声音信号进行校正，并采用核特征矩阵联合近似对角化的方法提取校正后的声音信号特征，最后使用 SVM 进行故障诊断。李春雷[3]等人采集了发电机正常状态声音和其他三种异常状态声音，将声音信号在时域和频域进行分解后得到的能量值、均方差值和峭度值作为声音特征，最后通过有监督学习的方式训练 BP 神经网络达到检测发电机状态的目的。上述方法具有较高的准确率，但需要对异常情况有明确的定义并且训练过程中需要大量异常声音样本。

### 无监督的好处

然而现实工厂机械设备故障率低、故障种类多且操作环境相对复杂，难以收集到多类型、多工况的真实异常声音信号。无监督异常检测训练过程只需正常样本即可完成，因此无监督异常检测方法在工业声学检测中显得尤为重要。目前无监督异常检测主要有基于数据重构的方法[5-8]
、基于概率分布[9-11]的方法和基于分类器的方法[12,13]。基于数据重构的方法是将原始样本进行压缩，并根据压缩后的数据尽可能地重构原始样本，使用重构误差作为异常分数。例如自编码器，赵光权[5]等人使用轴承正常振动信号训练自编码器模型，模型能够以较小的误差重构测试正常样本，而异常样本重构误差则较大，以重构误差作为异常分数，达到检测轴承健康状态的目的。基于概率分布的方法是假设样本特征符合正态分布，如果样本数据点在整体分布上的概率密度值较小，即为异常。例如高斯混合模型，QU J [11] 等人提出了使用高斯混合模型对高光谱数据进行异常检测。基于分类器的方法是在特征空间中学习正常样本
周围的边界。例如单类支持向量机（OC-SVM），陈志全[12]等人提出 OC-SVM 异常环境声音检测算法，仅使用正常环境声音训练一个 OC-SVM 来判断被测声音是否为正常环境声音。上述方法已被证明具有良好的实用性。

同时由于工业场景较为复杂，工业声学信号存在高维度、非线性、多项混叠等问题，因此在进行声学异常检测时需要先对工业声学时频信号进行进一步的特征提取，才能进入异常检测器进行异常识别。提取声音信号特征是无监督机械声音异常检测的难点之一。目前常用的声学信息号特征提取方法包括线性声谱图（Line Spectrum）[14]、对数梅尔谱（Log-mel）[8]、梅尔频率倒谱系数（MFCCs）[15]、谐波分量（Hpss-h）和冲击分量（Hpss-p）[16]等。吴侃[14]首先提取离心泵运行声音的 Line Spectrum 特征，然后根据此特征进行设备异常检测。
2020DCASE[8]比赛任务 2 为无监督机器异常声音检测，基线系统首先提取机械设备正常运行声音的 Log-mel 特征，并用提取出的特征训练一个自编码器，并将重构误差作为判别标准。Truong H[15]等人提取了机械设备正常运行声音的 MFCCs 特征，同样也使用提取的特征训练自编码器，实现无监督机械设备异常声音检测。然而这些特征提取方法需要人为指定超参数，人为经验影响较大，且针对不同类型机械设备声音信号，特征提取效果往往通用性不强。而自监督特征提取是通过设置附属任务来学习对下游任务有用的特征，已广泛应用于图像领域[17-19]。SPYROS G[17]等人将图像进行不同角度的旋转，将识别图像旋转的角度作为附属任务来提取图像深度特征。上述方法已被证明在图像异常检测具有良好的实用性，但自监督学习在声学异常检测中暂未见报道。针对人工提取声音信号特征存在人为影响较大、通用性不强等问题，本文提出一种自监督特征提取新方法，由于提取的特征仍为高维特征，因此将此特征
输入自编码器（AE）进行机械设备异常声音检测。本文自监督特征提取方法首先将声音样本转换为时频谱图，并采用随机增强减弱、添加
粉噪声等方法生成异常时频谱图，然后使用正常时频谱图与人造异常时频谱图训练一个卷积二分类网络从而构建自监督特征提取器（SSFE）。最后使用 SSFE 提取的正常样本特征训练一个自编码器（AE），基于自编码重构误差，实现无监督机械设备异常声音检测。通过对 MIMII
公开数据集滑轨、阀门、水泵和风扇等 4 种机械设备进行异常声音检测证明了该方法的有效性。

## 7. Drill fault diagnosis

> [7] Tran T, Lundgren J. Drill fault diagnosis based on the scalogram and MEL spectrogram of sound signals using artificial intelligence[J]. IEEE Access, 2020, 8: 203655-203666.

### Abstract

问题：In industry, the ability to detect damage or abnormal functioning in machinery is very important. However, manual detection of machine fault sound is economically inefficient and labor-intensive. Hence,automatic **machine fault detection (MFD)** plays an important role in reducing operating and personnel costs compared to manual machine fault detection. 

This research aims to develop a drill fault detection system using state-of-the-art artifcial intelligence techniques. Many researchers have applied the traditional approach design for an MFD system, including handcrafted feature extraction of the raw sound signal,
feature selection, and conventional classication. However, drill sound fault detection based on conventional machine learning methods using the **raw sound signal** in the **time domain** faces a number of challenges. 

Challenges: 

#### good features

> For example, it can be difcult to extract and select good features to input in a classier, and the accuracy of fault detection may not be sufcient to meet industrial requirements. Hence, we propose a method that uses deep learning architecture to extract rich features from the image representation of sound signals combined with machine learning classiers to classify drill fault sounds of drilling machines. 

The proposed methods are trained and evaluated using the real sound dataset provided by the factory. The experiment results show
a good classication accuracy of 80.25 percent when using Mel spectrogram and scalogram images. The results promise signicant potential for using in the fault diagnosis support system based on the sounds of drilling machines.

### Other papers

#### SVM & ANN

[2]  

Kumar et al. [3] developed a system for automatic drilling operations using vibration signals. The authors used low pass Butterworth filter to preprocess vibration signals before extracting eight features from the time domain, eight features from the frequency domain, and five Morlet wavelet features.

#### Mel & SVM

Lee et al. [4] extracted Mel-frequency cepstrum coefficients (MFCCs) from the audio signals and also employed SVM for classification. The accuracy reached 94.1 percent on their dataset, which is collected from an NS-AM-type railway point machine at Sehwa Company in Daejeon, South Korea. The length of each sound on their dataset was around 5000 ms.

##### 问题

However, their method did not show a promising result when applied in our drill sound dataset because each sound recording is extremely short.

Kemalkar and Bairagi [5] extracted **MFCCs features** and made a comparison between these features and a library of features to decide on the **fault or non-fault state** of a bike engine. 

Zhang [6] used the principal component analysis (**PCA**) algorithm to extract and train the training samples. 

自组织映射(Self-organizing Maps,SOM)

Then, the author used self-organizing maps (**SOM**) to cluster the principal component by neural network clustering into four categories and the Bayesian discriminant method to identify the testing samples. The dataset for his experiment was collected by a self-developed drilling test rig using a signal acquisition hardware system (sensors, data acquisition cards, and industrial computers)

#### Covoluted Neural Network 

Ince et al. [7] proposed using a time-domain signal as the input of a small 1-D CNN to classify motors as either healthy or faulty. The authors used a balanced dataset of 260 healthy and 260 faulty cases for training a 1-D CNN model. 

Luo et al. [8] detected the fault stage of CNC machine tools based on their vibration signals. The authors used 10 000 samples; 9000 of these were used for training and 1000 were used for testing a deep auto-encoder (DAE) model. The DAE model, which is combined between the SAE layer and the BPNN layer, is used to classify impulse and non-impulse responses. A dynamics identification algorithm was then used to identify dynamic properties from impulse responses. 

Finally, similarities between the dynamic properties were used to detect the health of the CNC machine tool. Similarly, Long et al. [9] combined a sparse autoencoder (SAE) and an echo state network (ESN) to diagnose the transmission faults of delta 3-D printers. These authors collected attitude data from an attitude sensor and used SAE to extracted features from attitude data. Then these features were used as the input of an ESN for fault recognition. Long et al. [10] alsocombined a hybrid evolutionary algorithm featuring a competitive swarm optimizer and a local search to optimize parameter values and hyperparameter settings of echo state network for intelligent fault diagnosis. To test the performance of their proposed method, the authors conducted fault diagnosis experiments for a 3-D printer and a gearbox

Many recent studies gained remarkable results when using image representation of the sound signal to train state-of-the-art deep learning architectures such as convolutional neural networks (CNNs) on machine fault sound diagnosis [12]–[15].

###### CNN的问题

然而，绝大多数研究都是使用大型和平衡的数据集进行的。实际上，与钻机的正常工作声音相比，钻头断裂时记录的声音仅占整个数据集的一小部分。不平衡的现实世界数据集导致训练CNN架构的偏见;例如，CNN 模型可能对少数族裔类的预测很差，因为该类的数据较少。此外，由于我们现实世界数据集中的每个声音样本都太短，约为20.83 ms和41.67 ms，因此从原始声音信号中提取的特征不能携带太多重要的信息进行分类。与以前的研究相比，这导致钻声分类的难度更大。据我们所知，没有使用如此短的钻孔声音进行过研究。因此，使用端到端系统（仅使用传统的机器学习或先进的卷积机器学习架构）并不能满足行业领域声音分类的预期准确性。为了解决钻断声音检测的问题，需要一个由多层组成的架构（预处理、图像转换、特征提取、特征选择、分类）。对于每一层，可以使用包含深度学习架构和机器学习方法的不同算法。传统方法（包括多个步骤/层）的局限性在于，每个步骤都需要在不同的标准下单独优化。但是，在不同的标准下单独优化每个步骤，而不是丢失区分性信息，可以帮助提高每个步骤和整个系统的性能。

#### Overfitting

To overcome overfitting with the small dataset, we decided to extract features from the image representations of sounds (Mel spectrogram and scalogram images) and choose the conventional classification models for the classification task. The detail of our proposed methods is presented in the next section.

The number of features extracted from Mel spectrogram images is 25 088. To reduce the unnecessary features, neighborhood component analysis (NCA) [6] was utilized for selecting the most relevant information for classification. Minimizing the redundancy of extracted features from VGG19 can help the model train faster and more effectively. We used classifiers provided in the Matlab toolbox to classify drill sounds using the extracted features from images.

提取了25088特征，还用NCA做了有优化

#### Conclusion

在这项研究中，我们提出了一种用于钻音分类的新方法，该方法包括使用声音的图像表示（Mel频谱图和标尺图）并根据深度学习CNN从这些图像中提取特征。我们还利用NCA来减少CNN的提取特征的数量，然后再将其输入到机器学习分类器中。我们提出的方法在标度图和Mel三维频谱图图像上实现了80.25%。所获得的结果在行业早期故障钻探检测方面具有可喜性。在比较研究中，我们比较了两个时频分析（Mel三维图和标尺），当使用这些图像作为我们提出的方法的输入时。使用带有COI的凸块小波获得的显像图图像有助于提高我们程序的性能。此外，我们还尝试了传统方法（使用音高和13个MFCC特征作为KNN分类器的输入对钻孔声音进行分类）和数据集上最先进的深度卷积神经网络进行比较。实验结果证明了使用钻孔声音的图像表示对钻孔声音进行分类的鲁棒性，无论是使用CNN架构（如GoogLeNet）作为分类器还是使用我们提出的程序。

## 8. 顶煤放落过程煤矸声信号特征提取与分类方法

> [8] 袁源,汪嘉文,朱德昇,王家臣,王统海,杨克虎.顶煤放落过程煤矸声信号特征提取与分类方法[J].矿业科学学报,2021,6(06):711-720.DOI:10.19606/j.cnki.jmst.2021.06.010.

### 结果表明

在不同帧长下,基于时频域特征 的分类效果最稳定、准确率最高,随机森林、K 近邻、决策树、多层感知器模型分类准确率均达到 80% 以上,其中基于小波包分解与随机森林算法的分类器性能最好,分类准确率为 93. 06% 。 维 度较高的时频域特征向量之间存在相关性,降维可以提取少量的综合特征并降低系统的运算量, 利用主成分分析法将时频域特征向量降维至 20 后,分类准确率进一步提高至 94. 51% 。

文献 [10-11]研究了基于声信号的煤矸识别方法,该 方法抗干扰能力较强,但是目前采集与识别系统 尚未完善。基于声信号的煤矸识别方法,因其成本低、标 注便捷而广受重视。 文献[12]采用了独立分量分 析法对顶煤放落过程的声信号进行分析处理,根据 频谱差异初步判断了顶煤中的含矸率;文献[13] 考虑了环境噪声,采用含噪的超完备独立分量分析 方法解决放顶煤过程中声信号分离的问题;文献 [14]采用小波包变换的分析方法,发现频谱会随 煤矸混合比例不同而产生差异;文献[15] 经实验 分析得出煤矸声波时域信号中的峰峰值、方差和峭 度系数区分度明显;文献[16] 通过独立分量分析 将放煤声信号分离,并提取其梅尔倒谱系数特征作 为神经网络的输入,进行煤矸识别;文献[17]通过 希尔波特-黄变换的方法提取声振信号特征,基于双峰深度学习网络实现煤矸分类。

目前,放顶煤声信号特征提取和分类算法的研 究对比较少,且大部分测试没有现场数据的支持。 本文完成了工作面放顶煤声信号的采集,通过人工 标注建立放顶煤声信号样本库来研究声信号的机 理,综合分析样本帧长、特征差异、算法性能,对机 器学习分类模型进行评估。

### 时域特征及其计算公式

![时域特征及其计算公式](F:\毕业论文\Industrial-Machine-Investigation-and-Inspection\Papers_reading\时域特征及其计算公式.png)

### 离散傅里叶变换公式

![image-20220227120050090](F:\毕业论文\Industrial-Machine-Investigation-and-Inspection\Papers_reading\离散傅里叶变换公式.png)

基于蝶形因子的周期性和对称性,在 DFT 基础 上 改 进, 得 到 快 速 傅 里 叶 变 换 ( Fast Fourier Transform, FFT)方法,将复数乘法运算转换为复数 加法运算,大大降低了复杂程度与运算量,非常适用 于计算机运算。 进一步计算得到声信号的频域特征 及计算公式见表 2。 其中,XFC 为信号的重心频率; XMSF 为信号的均方频率;XVF 为信号的频率方差

### 频域特征及其计算公式

![image-20220227120207400](F:\毕业论文\Industrial-Machine-Investigation-and-Inspection\Papers_reading\频域特征及其计算公式.png)

### 时频域特征提取

![image-20220227120334884](F:\毕业论文\Industrial-Machine-Investigation-and-Inspection\Papers_reading\时频域特征提取.png)

### 机器学习分类算法的理论基础(支持向量机、K邻近、朴素贝叶斯、决策树、随机森林、神经网络)

袁源,汪嘉文,朱德昇,王家臣,王统海,杨克虎.顶煤放落过程煤矸声信号特征提取与分类方法[J].矿业科学学报,2021,6(06):711-720.DOI:10.19606/j.cnki.jmst.2021.06.010.               P4

### 样本帧长对分类准确率的影响 

不同帧长的样本会影响煤矸识别的准确率,本 文选取帧长为 50 ms、 100 ms、 150 ms、 200 ms、 250 ms、300 ms 的 6 种情况,帧移默认为 1 / 2 的帧 长,针对不同特征与不同分类器进行对比实验。

对于时域特征,SVM、MLP 的模型分类准确率 随着帧长的增加而降低,如图 7 所示;RF、KNN、DT 的 模 型 分 类 准 确 率 随 着 帧 长 增 加 而 增 加, 在 200 ms 之后趋于平缓,准确率基本保持不变; NBC 的模型分类准确率波动较大,在 100 ms 时取 得最大值。 RF、DT 模型表现较好,RF 模型在帧长 为 200 ms 时,最大值为 91. 60% 。·

### 降维

![image-20220227121457539](F:\毕业论文\Industrial-Machine-Investigation-and-Inspection\Papers_reading\PCA降维与准确性.png)

### 结 论 

本文研究了放顶煤声信号的特征提取方法,对 比了主流机器学习算法的分类表现,得出以下 结论: (1) 不同帧长的样本选取会对煤矸分类的准 确率 产 生 影 响。 选 择 帧 长 为 200 ms、 帧 移 为 100 ms 时,基于时域特征提取与时频域特征提取 的分类准确率最高,分别为 91. 60% 与 93. 35% 。 (2) 不同的特征选择适用于不同的算法模型。 本文的时频域特征提取方法适用于 RF、KNN 与 MLP 分类器,煤矸分类准确率分别为 93. 06% 、 90. 05% 、91. 02% 。 相比于时域特征和频域特征, 时频域特征能提高分类准确率。 (3) 对于时频域特征向量,降维有助于提升分 类准确率。 RF、KNN、DT、MLP 4 种模型的分类准 确率分别提升了 1. 45% 、0. 29% 、2. 40% 、0. 91% , 其中 RF 分类模型准确率最高,为 94. 51% ,对应的 时频域特征向量维度为 20。 (4) RF、KNN、DT、MLP 在获得较高分类准确 率的同时,模型大小分别为 6. 60 MB、6. 98 MB、 0. 05 MB 和 0. 01 MB。 综合考虑分类准确率与模 型大小,MLP 模型更具备低功耗平台的适用性。

## 9.基于呼吸声特征分析的肺部疾病诊断方法研究

> [9] 常峥. 基于呼吸声特征分析的肺部疾病诊断方法研究[D].重庆邮电大学,2021.DOI:10.27675/d.cnki.gcydx.2021.000285.

针对目前呼吸声信号识别率低的问题，提出了一种基于希尔伯特黄变换
(Hilbert-Huang  Transform,  HHT) 的 梅 尔 频 率 倒 谱 系 数 (Mel-Frequency  Cepstral 
Coefficients,  MFCC)与短时能量(Energy)融合的特征提取算法 HHT-MFCC+Energy。
首先，呼吸声信号通过 HHT 计算出希尔伯特(Hilbert)边际谱和边际谱能量；其次，
谱能量通过梅尔(Mel)滤波器得到特征向量，再对特征向量取对数和离散余弦变换
(Discrete  Cosine  Transform,  DCT)求出 HHT-MFCC 特征；最后，计算出呼吸声信号
的短时能量特征并与 HHT-MFCC 特征融合便可得到新特征融合算法。

 ### 结果

通过采集相关疾病的呼吸声信号并利用实验平台验证所提算法的有效性，利用
支持向量机(Support  Vector  Machine,  SVM)构建决策二叉树分类法中的偏二叉树分
类模型。实验结果表明，所提算法能有效识别健康、慢性阻塞性肺病以及肺炎呼吸
声信号，精确率(Precision)和召回率(Recall)均在0.920以上，准确率(Accuracy)为 0.942，
基本实现利用呼吸声信号诊断肺部疾病的目标。 

### 1.3.3  声音信号特征提取技术研究现状  

Page6

### 第 2 章

为 Hilbert-Huang 变换与呼吸声信号处理方法理论基础。介绍了希尔伯特
黄变换以及呼吸声信号的降噪方法和特征提取方法并对比了各方法的优缺点，确定
了本文在信号处理过程中预处理和特征提取环节的研究方法且阐述了诊断肺部疾病
大致流程。

## 10.卷积神经网络（CNN）在语音识别中的应用

> [10] https://www.cnblogs.com/jins-note/p/9897191.html

长短时记忆网络（LSTM，LongShort Term Memory）可以说是目前语音识别应用最广泛的一种结构，这种网络能够对语音的长时相关性进行建模，从而提高识别正确率。双向LSTM网络可以获得更好的性能，但同时也存在训练复杂度高、解码时延高的问题，尤其在工业界的实时识别系统中很难应用。

### 1 语音识别为什么要用CNN

通常情况下，语音识别都是基于时频分析后的语音谱完成的，而其中语音时频谱是具有结构特点的。要想提高语音识别率，就是需要克服语音信号所面临各种各样的多样性，包括说话人的多样性(说话人自身、以及说话人间)，环境的多样性等。一个卷积神经网络提供在时间和空间上的平移不变性卷积，将卷积神经网络的思想应用到语音识别的声学建模中，则可以利用卷积的不变性来克服语音信号本身的多样性。从这个角度来看，则可以认为是将整个语音信号分析得到的时频谱当作一张图像一样来处理，采用图像中广泛应用的深层卷积网络对其进行识别。

从实用性上考虑，CNN也比较容易实现大规模并行化运算。虽然在CNN卷积运算中涉及到很多小矩阵操作，运算很慢。不过对CNN的加速运算相对比较成熟，如Chellapilla等人提出一种技术可以把所有这些小矩阵转换成一个大矩阵的乘积。一些通用框架如Tensorflow，caffe等也提供CNN的并行化加速，为CNN在语音识别中的尝试提供了可能。

### 2 CLDNN

提到CNN在语音识别中的应用，就不得不提CLDNN（CONVOLUTIONAL, LONG SHORT-TERM MEMORY,FULLY CONNECTED DEEP NEURAL NETWORKS）[1]，在CLDNN中有两层CNN的应用，算是浅层CNN应用的代表。CNN 和 LSTM 在语音识别任务中可以获得比DNN更好的性能提升，对建模能力来说，CNN擅长减小频域变化，LSTM可以提供长时记忆，所以在时域上有着广泛应用，而DNN适合将特征映射到独立空间。而在CLDNN中，作者将CNN，LSTM和DNN串起来融合到一个网络中，获得比单独网络更好的性能。

CLDNN网络的通用结构是输入层是时域相关的特征，连接几层CNN来减小频域变化，CNN的输出灌入几层LSTM来减小时域变化，LSTM最后一层的输出输入到全连接DNN层，目的是将特征空间映射到更容易分类的输出层。之前也有将CNN LSTM和DNN融合在一起的尝试，不过一般是三个网络分别训练，最后再通过融合层融合在一起，而CLDNN是将三个网络同时训练。实验证明，如果LSTM输入更好的特征其性能将得到提高，受到启发，作者用CNN来减小频域上的变化使LSTM输入自适应性更强的特征，加入DNN增加隐层和输出层之间的深度获得更强的预测能力。

### LSTM

### 各企业发展

### 4.总结

由于CNN本身卷积在频域上的平移不变性，同时VGG、残差网络等深度CNN网络的提出，给CNN带了新的新的发展，使CNN成为近两年语音识别最火的方向之一。用法也从最初的2-3层浅层网络发展到10层以上的深层网络，从HMM-CNN框架到端到端CTC框架，各个公司也在deep CNN的应用上取得了令人瞩目的成绩。

总结一下，CNN发展的趋势大体为：

1 更加深和复杂的网络，CNN一般作为网络的前几层，可以理解为用CNN提取特征，后面接LSTM或DNN。同时结合多种机制，如attention model、ResNet 的技术等。

2 End to End的识别系统，采用端到端技术CTC ， LFR 等。

3 粗粒度的建模单元，趋势为从state到phone到character，建模单元越来越大。

但CNN也有局限性，[2,3]研究表明，卷积神经网络在训练集或者数据差异性较小的任务上帮助最大，对于其他大多数任务，相对词错误率的下降一般只在2%到3%的范围内。不管怎么说，CNN作为语音识别重要的分支之一，都有着极大的研究价值。

### 参考文献：

>  [ 1 ] Sainath,T.N, Vinyals, O., Senior, O.,Sak H:CONVOLUTIONAL, LONG SHORT-TERM MEMORY, FULLY CONNECTED DEEP NEURAL NETWORKS
>  [ 2 ] Sainath,T.N , Mohamed,A.r , Kingsbury ,B., Ramabhadran,B.:DEEP CONVOLUTIONAL NEURAL NETWORKS FOR LVCSR.In:Proc. International Conference on Acoustics, Speech and signal Processing(ICASSP),pp.8614-8618(2013)
>  [ 3 ] Deng, L.,Abdel-Hamid,O.,Yu,D.:A DEEP CONVOLUTIONAL NEURAL NETWORK USING HETEROGENEOUS POOLING FOR TRADING ACOUSTIC INVARIANCE WITH PHONETIC CONFUSION.In:Proc. International Conference on Acoustics, Speech and signal Processing(ICASSP),pp.6669-6673(2013)
>  [ 4 ] Chellapilla, K.,Puri, S., Simard,P.:High Performance Convolutional Neural Networks for Document Processing.In: Tenth International Workshop on Frontiers in Handwriting Recognition(2006)
>  [ 5 ]Zhang, Y., Chan ,W., Jaitly, N.:VERY DEEP CONVOLUTIONAL NETWORKS FOR END-TO-END SPEECH RECOGNITION.In:Proc. International Conference on Acoustics, Speech and signal Processing(ICASSP 2017)

## 11.如何使用卷积神经网络从梅尔谱图检测 COVID-19 咳嗽

>[11] https://blog.csdn.net/woshicver/article/details/120387103

### 什么是梅尔谱图？

梅尔谱图是转换为梅尔标度的谱图。那么，什么是[频谱](https://so.csdn.net/so/search?q=频谱&spm=1001.2101.3001.7020)图和梅尔音阶？频谱图是信号频谱的可视化，其中信号的频谱是信号包含的频率范围。梅尔音阶模仿人耳的工作方式，研究表明人类不会在线性音阶上感知频率。与较高频率相比，人类更擅长检测较低频率的差异。

![c864166c1342825345afb20004046c1c.png](F:\毕业论文\Industrial-Machine-Investigation-and-Inspection\Papers_reading\Mel)

### 数据集条件

使用的语音数据可以在https://github.com/virufy/virufy-data下载。

使用的声音数据是 COVID-19 阳性和 COVID-19 阴性的咳嗽声音记录。此数据为 mp3 格式，具有采样率为 48000 Hz 的单声道，并已进行分段以使其具有相同的时间。

但是，系统可以使用这些数据来识别感染 COVID-19 的人的咳嗽声吗？mp3音频格式需要转换成wav格式。为什么？因为语音识别中处理的是频率波和幅度波，而wav是波（waveform）形式的音频格式。因此，需要对音频进行预处理以将格式从mp3 更改为wav 格式。完成这一步后，就可以得到光谱图mel。

从音频中获取图像梅尔频谱图
软件 Audacity 可用于将 mp3 音频格式转换为 wav 格式。然后使用python编程语言中的librosa包读取wav格式的音频。通过使用python自带的librosa包来分析音频，下采样得到的音频数据48000Hz的采样率遵循librosa包的默认采样率，这样音频的采样率就变成了22050Hz。可以在librosa 文档中查看有关如何获取 Mel 频谱图的文档

### 构建模型

让我们使用 Python 和 Google Colab 制作一个系统，该系统可以使用卷积神经网络从 Mel Spectrogram 中识别来自 COVID-19 的感染者和非感染者的咳嗽声。

详细见 https://blog.csdn.net/woshicver/article/details/120387103

## 12.卷积神经网络实战

> [12] https://cloud.tencent.com/developer/article/1031245

1. 输入层

   一张完整的向量

2. 卷积层

   一个过滤器。

   隐藏层的每个神经元并不是和上一个隐藏层所有的神经元都相连，而只是和上一个隐藏层某一小片相连。

   输入层到隐藏层的这种映射叫做**特征映射（Feature map）**。kernel→feature map

   **这就是卷积层的作用：给定一个过滤器，卷积层会扫描输入层来阐述一个特征映射。**

3. 激活函数

   Relu：负数为零，正数为一

   激活函数有两个主要优点：

   - 它们给网络引入了非线性。实际上，目前提到的运算都是线性的，像卷积、矩阵相乘和求和。如果我们没有非线性，那么最终我们会得到一个线性模型，不能完成分类任务。
   - 它们通过防止出现梯度消失来提升训练进程。

4. 池化层

   修订特征映射现在来到池化层了。池化是一个下采样操作，能让特征映射降维。

   最普通的池化操作是Max-pooling。它选取一个小窗口，一般是2x2 ，以stride（步幅）为2在修订特征映射上滑动，每一步都取最大值。

   Max-pooling有很多优点：

   - 减小了特征映射的尺寸，减少了训练参数数目，从而控制过度拟合。
   - **获取最重要特征来压缩特征映射。**
   - 对于输入图片的变形、失真和平移具有网络不变性，输入小的失真不会改变池化的输出，因为我们是取最大值。

5. 全连接层

   卷积神经网络也有一层全连接层，就是我们在经典全连接网络中看到的那种。它一般在网络的末尾，最后一层池化层已经扁平化为一个矢量，和输出层全连接。而输出层是预测矢量，尺寸和类别数目相同。

   ![Overview of CNN](F:\毕业论文\Industrial-Machine-Investigation-and-Inspection\Papers_reading\Overview of CNN.png)

   全连接层起着**分类**的作用，而前面那些层都是在**提取特征**。

   全连接层收到卷积、修订和池化的压缩产物后，整合它们来实施分类。

   除为了分类，添加一层全连接层也是一种学习这些特征非线性组合的方法。从卷积层和池化层来的大部分特征可能已经能够完成分类，但是整合这些特征后效果会更好。

   把全连接层看作是添加到网络的一种附加提取。

我们一般会设置2到3个全连接层，这样在实施分类前就可以学习非线性特征组合。

关于这点可以引用Andrej Karpathy的话：

> 最普通的卷积神经网络架构形式是包含好几层卷积-函数激活层的，池化层紧随其后，这种结构不断重复直到图片已经完全融合成小尺寸。在某个阶段，转化到全连接层也是很正常的事情。最后一层全连接层持有输出，如类别值。

## 13.可视化和可解释的CNN

> [13] Zeiler, M. D., & Fergus, R. (2014). *Visualizing and Understanding Convolutional Networks. Lecture Notes in Computer Science, 818–833.* doi:10.1007/978-3-319-10590-1_53

### Why former research succeed 

- the availability of much larger training sets, with millions of labeled examples; 
- powerful GPU implementations, making the training of very large models practical and
- better model regularization strategies, such as Dropout (Hinton et al., 2012).

### revealing which parts of the scene are important for classification.

## 14. **钨极气体保护电弧焊** CNN 多分类

> [14] Ren W, Wen G, Xu B, et al. A Novel Convolutional Neural Network Based on Time–Frequency Spectrogram of Arc Sound and Its Application on GTAW Penetration Classification[J]. IEEE Transactions on Industrial Informatics, 2020, 17(2): 809-819.

## 15.咳嗽分类

> [15] Nguyen D S, Dang K T. COVID-19 Detection Through Smartphone-recorded Coughs Using Artificial Intelligence: An Analysis of Applicability for Pre-screening COVID-19 Patients in Vietnam[C]//2021 IEEE International Conference on Industrial Engineering and Engineering Management (IEEM). IEEE, 2021: 1392-1396.

## 16.Pytorch入门

> [16] https://www.bilibili.com/video/BV1hE411t7RN?spm_id_from=333.999.0.0

0. Pytorch配置

   1. anaconda

   2. 显卡驱动 
   3.  

1. Pycharm & Jupyter

2. 两个重要的函数

   1. Package函数 

      dir() 打开

      help() 说明书

      dir(pytorch) :   1,2,3,4

      dir(pytorch.3)  :    a,b,c,d

      help(pytorch.3.a)   :What will a do

3. 数据加载

   1. Dataset

      提供一种方式去获取数据及其label

   2. Dataloader

      为后面的网络提供不同的数据形式

      告诉我们总共有多少的数据

4. Tensorboard

   1. ​	查找 Terminal:          tensorboard --logdir=logs

   2. 在log中删除记录

      

## 17. TFA综述

> [17] Yang Y, Peng Z, Zhang W, et al. Parameterised time-frequency analysis methods and their engineering applications: A review of recent advances[J]. Mechanical Systems and Signal Processing, 2019, 119: 182-221.

- Almost all classical TFAs are non-parameterised methods

i) the conventional non-parameterised TFA

ii) the state-of-art adaptive non-parameterised TFA

iii) parameterised TFA. 

- STFT

![STFT](F:\毕业论文\Industrial-Machine-Investigation-and-Inspection\Papers_reading\STFT.png)

- CWT

![CWT](F:\毕业论文\Industrial-Machine-Investigation-and-Inspection\Papers_reading\CWT.png)

- WVD

![WVD](F:\毕业论文\Industrial-Machine-Investigation-and-Inspection\Papers_reading\WVD.png)

## 18.基于深度学习的音频场景分类研究 

> [18]  乔高杰. 基于深度学习的音频场景分类研究[D]. 武汉邮电科学研究院, 2021.

为了解决音频场景分类准确率低的问题，本文主要是单模型进行改进，研究工作从三个方面进行展开，分别是：以对数梅尔谱图为基础，通过改变滤波器的数量、使用不同通道的音频以及谐波冲击源分离(Harmonic Percussive Source Separation, HPSS)的增强方法来提取不同信息的音频特征；在卷积神经网络作为分类器的基础上，通过添加压缩激励(Squeeze Excitation，SE)模块将能够关注到卷积模块输出特征通道间的信息，并且利用压缩激励来提取不同频率间的信息；以经典的卷积神经网络结构 VGG(Visual Geometry Group)和 Inception 中的基本结构单元为基础，将1个 Inception 和 2 个 VGG 基本结构单元组成混合网络作为分类器。 

1. 如何更好的提取音频特征：通过改变滤波器的数量、使用不同通道的音频以及谐波冲击源分离(Harmonic Percussive Source Separation, HPSS)的增强方法来提取不同信息的音频特征；
2. 计算机听觉场景分析的主要研究内容包含音频场景分类(Acoustic Scene Clas-sification, ASC)和声音事件检测(Acoustic Event Detection, AED)[2]两个方面。
3. 梅尔频谱图：
4. ![image-20220403214727792](F:\毕业论文\Industrial-Machine-Investigation-and-Inspection\Papers_reading\.git\image-20220403214727792.png)
   - 预加重
   - 分帧
   - 离散傅里叶变换
   - 梅尔滤波器
   - 利用离散余弦变换提取MFCC
5.  不同粒度特征提取 
6. 本章共包含四个音频场景分类系统，分别为**基线系统**、**基于音频特征的音频场景分类系统**、**基于压缩激励机制的音频场景分类系统**以及**基于混合卷积网络的音频场景分类系统**。本章通过实验的方式来检验不同的系统中不同模型的分类效果，并对实验的结果进行了分析。 

## 19.基于改进同步挤压小波变换识别信号瞬时频率

> [19]刘景良,郑锦仰,郑文婷,黄文金.基于改进同步挤压小波变换识别信号瞬时频率[J].振动.测试与诊断,2017,37(04):814-821+848.DOI:10.16450/j.cnki.issn.1004-6801.2017.04.028.

