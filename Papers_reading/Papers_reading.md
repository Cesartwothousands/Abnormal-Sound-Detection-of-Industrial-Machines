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

![image-20220212174914246](F:\毕业论文\Industrial-Machine-Investigation-and-Inspection\Papers_reading\dataset content.png)

## 2. 机器学习基础---无监督学习之异常检测

> https://www.cnblogs.com/ssyfj/p/12940077.html 

