We sincerely thank the reviewers for the constructive comments. We clarify concerns below and will make them clear in the revision.

**About open-source.**
We guarantee all our results are solid, correct, and reproducible. We will release our code when this work is accepted. Besides, we had submitted an example code in supplementary materials with the paper.

**Reviewer A:**

**1. Question about why better than white-box.**
- We did not try to claim that we are definitely better than all white-box solutions. Since their goal and metric (e.g. neuron coverage) cannot be used in black-box, we adopt different goals and metrics following previous works in our paper. We can perform better than white-box solutions only considering our goal and metric. 
- Moreover, our advantage than existing white-box solutions are analyzed in Sec. 4.3 **Analysis**: (1) Achieving high neuron coverage by the baseline methods does not definitely mean they can find more error-inducing inputs. (2) Neuron coverage metrics may have a bias as pointed out by [1, 2]. For instance, the neuron coverage metric in DeepXplore [3] promotes neurons whose values are below a threshold, in order to get higher values for better neuron coverage. It may fail to test inputs that will cause neurons with low values.
- Besides, we also found white-box baselines suffer the inactivation issue in Sec. 4.3, **Result of Inact-Rate**, but our method did not. White-box baselines first preprocess initial test cases (e.g., pixel values of images are integers in [0,255]) to lift them to the continuous domain, such as decimals within [0,1] before generating error-inducing inputs. However, after discretization, many error-inducing inputs would become inactive (i.e., can no longer mislead target CNNs) due to information loss. In contrast, perturbations generated by us are directly added to pixel values as integers. Therefore, our method does not suffer the inactivation issue.

**2. Question about the novelty of objective function and efficiency-centric policy.**
-	Tunable objective function. To the best of our knowledge, we are the first black-box method that considers both differential testing (DT) and single model testing (ST) scenarios. We designed novel objective functions for both scenarios. Particularly, our tunable objective function is dynamically adjusted to explore different decision boundaries to thoroughly test CNNs. For DT: in Eq. (5), the first term $C(x)[l]$ means fully exploring the target model $C$, where $l$ is dynamically adjusted for exploring different decision boundaries of the model. The second term $\sum_{i=1}^{n}|C(x)[l_0]-\hat{C}_i(x)[l_0]|$ aims to find disagreements between the target model $C$ with a batch of other models $\hat{C}=\{\hat{C}_1,\hat{C}_2,...\hat{C}_n\}$, as such disagreements indicate error-inducing inputs are found. For ST, in Eq. (6), we only optimize $C(x)[l]$, as ST only focuses on the target model $C$, and labels are provided to indicate whether error-inducing inputs are found.  
-	Efficiency-centric policy. We are the first to establish this policy, and analyze why such policy makes the testing efficient in Sec. 3.4. We did consider alternatives and verify the function of our policy in Sec. 4.5. Results show that our policy outperforms alternatives.

**3. Question about why 30,000 are chosen.**
-	We first run all baseline methods under different query budgets for multiple times, and then we choose values (30,000) that make them perform relatively well. DiffChaser is also constrained to 30,000 queries.

**4. Question about efficiency and testing time.**
-	First, testing time is highly related with complexity of model structure and dataset. Also, remote black-box testing time is also related with sample transfer time. Thus, we focus on query efficiency but not time efficiency following baseline black-box methods [4,5] by restricting max iteration numbers which is essentially the same as restricting queries. 
- Besides, we also compared DiffChaser and white-box baselines in terms of testing time. For instance, we allocate 1 hour per image as ADAPT [6]. For BET, we allocate the time budget equally to 50 priority labels. Other hyperparameters of all methods are the same as before. Taking ImageNet dataset as an example, we show results in terms of testing time as follows and we will add these results in revision.
- Results in DT scenario: under the same time budget, BET outperforms DiffChaser on all our metrics. 

| Method | Model | Err-Num | Label-Num | SR (%) |
| :--------: | :------------: | :-----: | :-------: | :----: |
| **BET** | VGG19/VGG19-q8 | **213.5** | **8.1** | **100.0** |
| DiffChaser | VGG19/VGG19-q8 | 9.7 | 0.4 | 38.0 |
| **BET** | VGG19/VGG19-q16 | **106.2** | **3.6** | **100.0** |
| DiffChaser | VGG19/VGG19-q16 | 0.2 | 0.1 | 10.2 |
| **BET** | ResNet50/ResNet50-q8| **1654.8** | **16.2** | **100.0** |
| DiffChaser | ResNet50/ResNet50-q8| 59.0 | 0.8 | 68.8 |
| **BET** | ResNet50/ResNet50-q16| **246.4** | **7.9** | **100.0** |
| DiffChaser | ResNet50/ResNet50-q16| 24.6 | 0.3 | 30.4 |
  - Results in ST scenario: under the same time budget, BET outperforms white-box baselines on all our metrics. 

| Method | Model | Err-Num | Label-Num | SR (%) | Inact-Rate (%) |
| :--: | :--: | :--: | :--: | :--: | :--: |
| **BET** | **VGG19** | **5942.8** | **38.4** | **100.0** | **0.0** |
| ADAPT | VGG19 | 2405.8 | 16,8 | 100.0 | 98.8 |
| DLFuzz-Best | VGG19 | 122.6 | 0.7 | 52.2 | 51.6 |
| DLFuzz-RR | VGG19 | 824.4 | 5.4 | 92.0 | 90.1 |
| DeepXplore | VGG19 | 197.8 | 1.8 | 40.2 | 38.8 |
| **BET** | **ResNet50** | **6447.4** | **29.8** | **100.0** | **0.0** |
| ADAPT | ResNet50 | 2957.5 | 2.6 | 89.0 | 93.2 |
| DLFuzz-Best | ResNet50 | 390.4 | 0.3 | 36.4 | 34.8 |
| DLFuzz-RR | ResNet50 | 902.8 | 0.8 | 54.7 | 50.2 |
| DeepXplore | ResNet50 | 1549.4 | 0.9 | 62.4 | 59.6 |


**5. Question about redundant error-inducing inputs.**
-	BET and all baselines do not suffer redundant error-inducing inputs. Given an original test input, our method will generate different perturbations without any duplicity, as we have declared in Sec 3.4 **P2** (line 479-482) and Sec. 3.5 (line 545-547).

**6. Question about overlapping with baseline methods.**
-	Most of our error-inducing inputs have little overlap with inputs generated by baseline methods. Particularly, for black-box baselines, take VGG19/VGG19-q16 as an example, we generate 231.5 times more error-inducing inputs than Diffchaser’s inputs. Therefore, calculating overlapping error-inducing inputs is not very meaningful. For white-box baselines, although some baselines have competitive Err-Num, we find their error-inducing inputs have relatively high Inact-Rate in Table 8. Such results indicate many error-inducing inputs generated by them are inactivate, and different with error-inducing inputs generated by us which do not suffer the inactivate problem. We will further confirm and show overlapping cases in appendix.

**7. Question about white-box methods.**
-	For ADAPT, we have run its two coverage strategies respectively and choose the relatively better results for comparison, as we have declared in Sec. 4.3 (line 861-863). Besides，taking ImageNet as an example, ADAPT’s default time budget is 1 hour per image in their paper. We have run their code and found the corresponding average queries are 23288 on VGG-19 and 18431 on ResNet-50, which are both below 30,000. Therefore, we believe our query budget makes ADAPT run to completion, and our comparisons are fair. We also compare white-box baselines in terms of testing time in Q4 of Reviewer A. Results show that we still perform better than baseline methods. 

**8. Question about ACC improvement.**
-	DeepXplore can increase the top-1 ACC on MNIST dataset (10 classes) by $2-3\%$ as their contribution. However, our Table 9 used TinyImageNet dataset (200 classes) and we can improve top-1 ACC by $2-3\%$ which is a significantly improvement than DeepXplore. Besides, if we use the same setting with DeepXplore, our ACC improvement can achieve $6\%$ (we have declared this in Sec. 4.4, line 1039-1043).

**Reviewer B:**

**1. Question about comparison with time budget.**
-	Please refer to Q4 of Reviewer A.

**2. Question about continuous perturbations.**
-	Each part of the mutator module supports the workflow for generating perturbations. These parts are in different links and play their respective roles. According to our observation, the sensing requires about $1.3\%$ query budget. And nearly $69\%$ cases choose square slicing and $31\%$ choose linear slicing.

**Reviewer C:**

**1. Question about continuous zones.**
-	The continuous zones are defined for the first convolution in the network. We try to maximize the change in the first convolution layer, and hope such change propagates and may eventually let the prediction result cross decision boundaries of CNNs to generate error-inducing inputs.

**2. Question about “$l$”.**
-	Given an input $x$, we define (line 405) labels $L=\{l_1,l_2,...l_n\}$ (including all classes except for $x$'s original label), and $l \in L$. During the testing process, our tunable objective function is dynamically adjusted to explore different decision boundaries and thoroughly test CNNs. Particularly, taking DT as an example: in Eq. (5), the first term $C(x)[l]$ indicates the confidence score of label $l$. Maximizing $C(x)[l]$ aims to find error-inducing inputs in label $l$’s decision boundaries. $l$ is dynamically adjusted for exploring different decision boundaries of the model. 

**3. Question about quantifying vulnerability.**
-	We use higher confidence scores to quantify.

**4. Question about diversifying different parts of one image.**
-	We split the test case into slices, and shuffle slice orders, then add perturbations to these slices one by one (each iteration only modifies one slice). Therefore, we explore different parts of the image, and make sure no parts of the image are missed. When all slices with the fixed larger size are iterated, again, we reduce the slice size, and re-split the test case into smaller slices, and shuffle slice orders, then add perturbations to these slices one by one. We iterate the whole process until the query budget exhausts. We rerun the whole experiment for multiple times to mitigate the effects of randomness.

**5. Question about solution $A$.**
-	When the greedy algorithm is finished, $A$ is the final solution of the greedy algorithm.

**6. Question about one by one.**
-	In line 544-545, we did declare “ All slices will be updated one by one in **multiple iterations**.”In each iteration, it is one slice. But for multiple iterations, slices will be updated one by one.

**7. Question about interesting follow-up.**
-	Thanks for your advice. This work only considers the black-box scenarios, therefore we do not consider the point you mentioned. We will follow your advice to explore this point.

**8. Question about the prime score.**
-	Prime score means the original maximum (prime) confidence score of the test case.



**References**

[1] Harel-Canada, Fabrice, et al. "Is neuron coverage a meaningful measure for testing deep neural networks?" ACM ESEC/SIGSOFT 2020.

[2] Li, Zenan, et al. "Structural coverage criteria for neural networks could be misleading." ICSE (NIER) 2019.

[3] Pei, Kexin, et al. "Deepxplore: Automated white-box testing of deep learning systems." SOSP 2017.

[4] Xie, Xiaofei, et al. "DiffChaser: Detecting Disagreements for Deep Neural Networks." IJCAI 2019.

[5] Odena, Augustus, et al. "Tensorfuzz: Debugging neural networks with coverage-guided fuzzing." ICML 2019.

[6] Seokhyun Lee, et al. “Effective white-box testing of deep neural networks with adaptive neuron-selection strategy.” ISSTA 2020.