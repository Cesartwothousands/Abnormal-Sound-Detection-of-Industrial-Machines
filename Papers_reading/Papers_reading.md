# Papers reading

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

