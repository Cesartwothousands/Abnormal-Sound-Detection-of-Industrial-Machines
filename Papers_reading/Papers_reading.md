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

**异常声音检测 (ASD) 是识别目标机器发出的声音是正常还是异常的任务。**. 自动检测机械故障是第四次工业革命中的一项重要技术，包括基于人工智能 (AI) 的工厂自动化。通过观察机器的声音来及时检测机器异常可能对机器状态监测有用。

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

## 8. 

> [8] 

