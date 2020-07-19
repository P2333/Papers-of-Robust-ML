# Papers-of-Robust-ML
Related papers for robust machine learning (we mainly focus on defenses).
 
# Statement
Since there are tens of new papers on adversarial defense in each conference, we are only able to update those we just read and consider as insightful.

Anyone is welcomed to submit a pull request for the related and unlisted papers on adversarial defense, which are pulished on peer-review conferences (ICML/NeurIPS/ICLR/CVPR etc.) or released on arXiv.

## Contents 
- <a href="#General_training">General Defenses (training phase)</a><br>
- <a href="#General_inference">General Defenses (inference phase)</a><br>
- <a href="#Detection">Adversarial Detection</a><br>
- <a href="#Verification">Verification</a><br>
- <a href="#Theoretical">Theoretical Analysis</a><br>
- <a href="#Empirical">Empirical Analysis</a><br>
- <a href="#Seminal_work">Seminal Work</a><br>
- <a href="#Benchmark_Datasets">Benchmark Datasets</a><br>


<a id='General_training'></a>
## General Defenses (training phase)
* [Understanding and Improving Fast Adversarial Training](https://arxiv.org/pdf/2007.02617.pdf) <br/> A systematic study of catastrophic overfitting in adversarial training, its reasons, and ways of resolving it. The proposed regularizer, *GradAlign*, helps to prevent catastrophic overfitting and scale FGSM training to high Linf-perturbations.

* [Smooth Adversarial Training](https://arxiv.org/pdf/2006.14536.pdf) <br/> This paper advocate using smooth variants of ReLU during adversarial training, which can achieve state-of-the-art performance on ImageNet.  

* [Rethinking Softmax Cross-Entropy Loss for Adversarial Robustness](https://openreview.net/forum?id=Byg9A24tvB) (ICLR 2020) <br/> This paper rethink the drawbacks of softmax cross-entropy in the adversarial setting, and propose the MMC method to induce high-density regions in the feature space.

* [Jacobian Adversarially Regularized Networks for Robustness](https://openreview.net/pdf?id=Hke0V1rKPS) (ICLR 2020) <br/> This paper propose to show that a generally more interpretable model could potentially be more robust against adversarial attacks.

* [Fast is better than free: Revisiting adversarial training](https://openreview.net/forum?id=BJx040EFvH&noteId=BJx040EFvH) (ICLR 2020) <br/> This paper proposes several tricks to make FGSM-based adversarial training effective.

* [Adversarial Training and Provable Defenses: Bridging the Gap](https://openreview.net/forum?id=SJxSDxrKDr) (ICLR 2020) <br/> This paper proposes the layerwise adversarial training method, which gradually optimizes on the latent adversarial examples from low-level to high-level layers.

* [Adversarial Interpolation Training: A Simple Approach for Improving Model Robustness](https://openreview.net/forum?id=Syejj0NYvr&noteId=r1e432RzoS) <br/> This paper introduces the mixup method into adversarial training to improve the model performance on clean images.

* [Are labels required for improving adversarial robustness?](https://arxiv.org/pdf/1905.13725.pdf) (NeurIPS 2019) <br/> This paper exploit unlabeled data to better improve adversarial robustness.

* [Adversarial Robustness through Local Linearization](https://arxiv.org/pdf/1907.02610.pdf) (NeurIPS 2019) <br/> This paper introduce local linearization in adversarial training process.

* [Provably Robust Boosted Decision Stumps and Trees against Adversarial Attacks](https://arxiv.org/pdf/1906.03526.pdf) (NeurIPS 2019) <br/> A method to efficiently certify the robustness of GBDTs and to integrate the certificate into training (leads to an upper bound on the worst-case loss). The obtained certified accuracy is higher than for other robust GBDTs and is competitive to provably robust CNNs.

* [You Only Propagate Once: Accelerating Adversarial Training via Maximal Principle](https://arxiv.org/pdf/1905.00877.pdf) (NeurIPS 2019) <br/> This paper provides a fast method for adversarial training from the perspective of optimal control.

* [Adversarial Training for Free!](https://arxiv.org/pdf/1904.12843.pdf) (NeurIPS 2019) <br/> A fast method for adversarial training, which shares the back-propogation gradients of updating weighs and crafting adversarial examples.

* [ME-Net: Towards Effective Adversarial Robustness with Matrix Estimation](https://arxiv.org/abs/1905.11971) (ICML 2019) <br/> This paper demonstrates the global low-rank structures within images, and leverages matrix estimation to exploit such underlying structures for better adversarial robustness.

* [Using Pre-Training Can Improve Model Robustness and Uncertainty](https://arxiv.org/abs/1901.09960) (ICML 2019) <br/>
This paper shows adversarial robustness can transfer and that adversarial pretraining can increase adversarial robustness by ~10% accuracy.

* [Theoretically Principled Trade-off between Robustness and Accuracy](https://arxiv.org/pdf/1901.08573.pdf) (ICML 2019) <br/> A variant of adversarial training: TRADES, which won the defense track of NeurIPS 2018 Adversarial Competation.

* [Robust Decision Trees Against Adversarial Examples](http://web.cs.ucla.edu/~chohsieh/ICML_2019_TreeAdvAttack.pdf) (ICML 2019) <br/> A method to enhance the robustness of tree models, including GBDTs.

* [Improving Adversarial Robustness via Promoting Ensemble Diversity](https://arxiv.org/pdf/1901.08846.pdf) (ICML 2019) <br/> Previous work constructs ensemble defenses by individually enhancing each memeber and then directly average the predictions. In this work, the authors propose the adaptive diversity promoting (ADP) to further improve the robustness by promoting the ensemble diveristy, as an orthogonal methods compared to other defenses.

* [Feature Denoising for Improving Adversarial Robustness](https://arxiv.org/pdf/1812.03411.pdf) (CVPR 2019) <br/> This paper applies non-local neural network and large-scale adversarial training with 128 GPUs (with training trick in 'Accurate, large minibatch SGD: Training ImageNet in 1 hour'), which shows large improvement than previous SOTA trained with 50 GPUs.

* [Improving the Generalization of Adversarial Training with Domain Adaptation](https://arxiv.org/pdf/1810.00740.pdf) (ICLR 2019) <br/> This work proposes to use additional regularization terms to match the domains between clean and adversarial logits in adversarial training.

* [A Spectral View of Adversarially Robust Features](http://papers.nips.cc/paper/8217-a-spectral-view-of-adversarially-robust-features.pdf) (NeurIPS 2018) <br/> Given the entire dataset X, use the eigenvectors of spectral graph as robust features. [[Appendix](http://papers.nips.cc/paper/8217-a-spectral-view-of-adversarially-robust-features-supplemental.zip)]

* [Adversarial Logit Pairing](https://arxiv.org/pdf/1803.06373.pdf) <br/> Adversarial training by pairing the clean and adversarial logits.

* [Deep Defense: Training DNNs with Improved Adversarial Robustness](http://papers.nips.cc/paper/7324-deep-defense-training-dnns-with-improved-adversarial-robustness.pdf) (NeurIPS 2018) <br/> They follow the linear assumption in DeepFool method. DeepDefense pushes decision boundary away from those correctly classified, and pull decision boundary closer to those misclassified.

* [Max-Mahalanobis Linear Discriminant Analysis Networks](http://proceedings.mlr.press/v80/pang18a/pang18a.pdf) (ICML 2018) <br/> This is one of our work. We explicitly model the feature distribution as a Max-Mahalanobis distribution (MMD), which has max margin among classes and can lead to guaranteed robustness.

* [Ensemble Adversarial Training- Attacks and Defenses](https://arxiv.org/pdf/1705.07204.pdf) (ICLR 2018) <br/> Ensemble adversarial training use sevel pre-trained models, and in each training batch, they randomly select one of the currently trained model or pre-trained models to craft adversarial examples.

* [Pixeldefend: Leveraging generative models to understand and defend against adversarial examples](https://arxiv.org/abs/1710.10766) (ICLR 2018) <br/> This paper provided defense by moving adversarial examples back towards the distribution seen in the training data.

<a id='General_inference'></a>
## General Defenses (inference phase)
* [Mixup Inference: Better Exploiting Mixup to Defend Adversarial Attacks](https://openreview.net/forum?id=ByxtC2VtPB) (ICLR 2020) <br/> This paper exploit the mixup mechanism in the inference phase to improve robustness.

* [Barrage of Random Transforms for Adversarially Robust Defense](http://openaccess.thecvf.com/content_CVPR_2019/papers/Raff_Barrage_of_Random_Transforms_for_Adversarially_Robust_Defense_CVPR_2019_paper.pdf) (CVPR 2019) <br/> This paper applies a set of different random transformations as an off-the-shelf defense.

* [Mitigating Adversarial Effects Through Randomization](https://arxiv.org/pdf/1711.01991.pdf) (ICLR 2018) <br/> Use random resizing and random padding to disturb adversarial examples, which won the 2nd place in th defense track of NeurIPS 2017 Adversarial Competation.

* [Countering Adversarial Images Using Input Transformations](https://arxiv.org/pdf/1711.00117.pdf) (ICLR 2018) <br/> Apply bit-depth reduction, JPEG compression, total variance minimization and image quilting as input preprocessing to defend adversarial attacks.

<a id='Detection'></a>
## Adversarial Detection
* [Towards Robust Detection of Adversarial Examples](http://papers.nips.cc/paper/7709-towards-robust-detection-of-adversarial-examples.pdf) (NeurIPS 2018) <br/> This is one of our work. We train the networks with reverse cross-entropy (RCE), which can map normal features to low-dimensional manifolds, and then detectors can better separate between adversarial examples and normal ones.

* [A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks](http://papers.nips.cc/paper/7947-a-simple-unified-framework-for-detecting-out-of-distribution-samples-and-adversarial-attacks.pdf) (NeurIPS 2018) <br/> Fit a GDA on learned features, and use Mahalanobis distance as the detection metric.

* [Robust Detection of Adversarial Attacks by Modeling the Intrinsic Properties of Deep Neural Networks](http://papers.nips.cc/paper/8016-robust-detection-of-adversarial-attacks-by-modeling-the-intrinsic-properties-of-deep-neural-networks.pdf) (NeurIPS 2018) <br/> They fit a GMM on learned features, and use the probability as the detection metric.

* [Detecting adversarial samples from artifacts](https://arxiv.org/abs/1703.00410) <br/> This paper proposed the kernel density (K-density) metric on the learned features to detect adversarial examples.

<a id='Verification'></a>
## Verification
* [Towards Verifying Robustness of Neural Networks against Semantic Perturbations](https://arxiv.org/abs/1912.09533) (CVPR 2020) <br/> This paper generalize the pixel-wise verification methods into the semantic transformation space.

* [Neural Network Branching for Neural Network Verification](https://arxiv.org/abs/1912.01329) (ICLR 2020) <br/> This paper use GNN to adaptively construct branching strategy for model verification.

* [Towards Stable and Efficient Training of Verifiably Robust Neural Networks](https://openreview.net/forum?id=Skxuk1rFwB) (ICLR 2020) <br/> This paper combines the previous IBP and CROWN methods.

* [A Convex Relaxation Barrier to Tight Robustness Verification of Neural Networks](http://papers.nips.cc/paper/9176-a-convex-relaxation-barrier-to-tight-robustness-verification-of-neural-networks.pdf) (NeurIPS 2019) <br/> This paper makes a conprehensive studies on existing robustness verification methods based on convex relaxation.

* [Tight Certificates of Adversarial Robustness for Randomly Smoothed Classifiers](https://guanghelee.github.io/pub/Lee_etal_neurips19.pdf) (NeurIPS 2019) <br/> This word extends the robustness certificate of random smoothing from L2 to L0 norm bound.

* [On the Effectiveness of Interval Bound Propagation for Training Verifiably Robust Models](https://arxiv.org/pdf/1810.12715.pdf) (ICCV 2019) <br/> This paper proposes the scalable verificatin method with interval bound propagation (IBP).

* [Evaluating Robustness of Neural Networks with Mixed Integer Programming](https://arxiv.org/abs/1711.07356) (ICLR 2019) <br/> This paper use mixed integer programming (MIP) method to solve the verification problem.

* [Efficient Neural Network Robustness Certification with General Activation Functions](https://arxiv.org/abs/1811.00866) (NeurIPS 2018) <br/> This paper proposes the verification method CROWN for general activation with locally linear or quadratic approximation.

* [A Unified View of Piecewise Linear Neural Network Verification](https://arxiv.org/abs/1711.00455) (NeurIPS 2018) <br/> This paper presents a unified framework and an empirical benchmark on previous verification methods

* [Scaling Provable Adversarial Defenses](http://papers.nips.cc/paper/8060-scaling-provable-adversarial-defenses.pdf) (NeurIPS 2018) <br/> They add three tricks to improve the scalability (to CIFAR-10) of previously proposed method in ICML.

* [Provable Defenses against Adversarial Examples via the Convex Outer Adversarial Polytope](https://arxiv.org/pdf/1711.00851.pdf) (ICML 2018) <br/> By robust optimization (via a linear program), they can get a point-wise bound of robustness, where no adversarial example exists in the bound. Experiments are done on MNIST.

* [Towards Fast Computation of Certified Robustness for ReLU Networks](https://arxiv.org/abs/1804.09699) (ICML 2018) <br/> This paper proposes the Fast-Lin and Fast-Lip methods.

* [Evaluating the Robustness of Neural Networks: An Extreme Value Theory Approach](https://arxiv.org/abs/1801.10578) (ICLR 2018) <br/> This paper proposes the CLEVER method to estimate the upper bound of specification.

* [Certified Defenses against Adversarial Examples](https://arxiv.org/abs/1801.09344) (ICLR 2018) <br/> This paper proposes the certified training with semidefinite relaxation.

* [A Dual Approach to Scalable Verification of Deep Networks](https://arxiv.org/abs/1803.06567) (UAI 2018) <br/> This paper solves the dual problem to provide an upper bound of the primary specification problem for verification.

* [Reluplex: An efficient SMT solver for verifying deep neural networks](https://arxiv.org/pdf/1702.01135.pdf) (CAV 2017) <br/> This paper use satisfiability modulo theory (SMT) solvers for the verification problem.

* [Automated Verification of Neural Networks: Advances, Challenges and Perspectives](https://arxiv.org/pdf/1805.09938.pdf) <br/> This paper provides an overview of main verification methods, and introduces previous work on combining automated verification with machine learning. They also give some insights on future tendency of the combination between these two domains.

<a id='Theoretical'></a>
## Theoretical Analysis
* [Towards Deep Learning Models Resistant to Large Perturbations](https://arxiv.org/pdf/2003.13370.pdf) <br/> This paper prove that the weight initialization of a already robust model on small perturbation can be helpful for training on large perturbations.

* [Improved Sample Complexities for Deep Neural Networks and Robust Classification via an All-Layer Margin](https://openreview.net/forum?id=HJe_yR4Fwr) (ICLR 2020) <br/> This paper connect the generalization gap w.r.t all-layer margin, and propose a variant of adversarial training, where the perturbations can be imposed on each layer in network.

* [Adversarial Examples Are Not Bugs, They Are Features](https://arxiv.org/pdf/1905.02175.pdf) (NeurIPS 2019) <br/> They claim that adversarial examples can be directly attributed to the presence of non-robust features, which are highly predictive but locally quite sensitive.

* [First-order Adversarial Vulnerability of Neural Networks and Input Dimension](https://arxiv.org/pdf/1802.01421.pdf) (ICML 2019) <br/> This paper demonsrate the relations among adversarial vulnerability and gradient norm and input dimension with comprehensive empirical experiments.

* [Adversarial Examples from Computational Constraints](https://arxiv.org/pdf/1805.10204.pdf) (ICML 2019) <br/> The authors argue that the exsitence of adversarial examples could stem from computational constrations.

* [Adversarial Examples Are a Natural Consequence of Test Error in Noise](https://arxiv.org/pdf/1901.10513.pdf) (ICML 2019) <br/> This paper connects the relation between the general corruption robustness and the adversarial robustness, and recommand the adversarial defenses methods to be also tested on general-purpose noises.

* [PAC-learning in the presence of evasion adversaries](https://arxiv.org/pdf/1806.01471.pdf) (NeurIPS 2018) <br/> The authors analyze the adversarial attacks from the PAC-learning framework.

* [Adversarial Vulnerability for Any Classifier](http://papers.nips.cc/paper/7394-adversarial-vulnerability-for-any-classifier.pdf) (NeurIPS 2018) <br/> Uniform upper bound of robustness for any classifier on the data sampled from smooth genertive models.

* [Adversarially Robust Generalization Requires More Data](http://papers.nips.cc/paper/7749-adversarially-robust-generalization-requires-more-data.pdf) (NeurIPS 2018) <br/> This paper show that robust generalization requires much more sample complexity compared to standard generlization on two simple data distributional models. 

* [Robustness of Classifiers:from Adversarial to Random Noise](http://papers.nips.cc/paper/6331-robustness-of-classifiers-from-adversarial-to-random-noise.pdf) (NeurIPS 2016)

<a id='Empirical'></a>
## Empirical Analysis
* [Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks](https://arxiv.org/abs/2003.01690) (ICML 2020) <br/> An comprehensive empirical evaluations on some of the existing defense methods. 

* [Attacks Which Do Not Kill Training Make Adversarial Learning Stronger](https://arxiv.org/pdf/2002.11242.pdf) (ICML 2020) <br/> This paper also advovate for early-stop during adversarial training.

* [Overfitting in adversarially robust deep learning](https://arxiv.org/pdf/2002.11569.pdf) (ICML 2020) <br/> This paper shows the phenomena of overfitting when training robust models with sufficient empirical experiments (codes provided in paper).

* [When NAS Meets Robustness: In Search of Robust Architectures against Adversarial Attacks](https://arxiv.org/abs/1911.10695) <br/> This paper leverages NAS to understand the influence of network architectures against adversarial attacks. It reveals several useful observations on designing robust network architectures.

* [Adversarial Examples Improve Image Recognition](https://arxiv.org/pdf/1911.09665.pdf) <br/> This paper shows that an auxiliary BN for adversarial examples can improve generalization performance.

* [Intriguing Properties of Adversarial Training at Scale](https://openreview.net/forum?id=HyxJhCEFDS&noteId=rJxeamAAKB) (ICLR 2020) <br/> This paper investigates the effects of BN and deeper models for adversarial training on ImageNet.

* [Interpreting Adversarially Trained Convolutional Neural Networks](https://arxiv.org/pdf/1905.09797.pdf) (ICML 2019) <br/> This paper show that adversarial trained models can alleviate the texture bias and learn a more shape-biased representation.

* [On Evaluating Adversarial Robustness](https://arxiv.org/pdf/1902.06705.pdf) <br/> Some analyses on how to correctly evaluate the robustness of adversarial defenses.

* [Adversarial Example Defenses: Ensembles of Weak Defenses are not Strong](https://arxiv.org/pdf/1706.04701.pdf) <br/> This paper tests some ensemble of existing detection-based defenses, and claim that these ensemble defenses could still be evaded by white-box attacks.

<a id='Seminal_work'></a>
## Seminal Work
* [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/pdf/1706.06083.pdf) (ICLR 2018) <br/> This paper proposed projected gradient descent (PGD) attack, and the PGD-based adversarial training.

* [Adversarial examples are not easily detected: Bypassing ten detection methods](https://dl.acm.org/citation.cfm?Id=3140444) (AISec 17) <br/> This paper first desgined different adaptive attacks for detection-based methods.

* [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572) (ICLR 2015) <br/> This paper proposed fast gradient sign method (FGSM), and the framework of adversarial training.

* [Intriguing properties of neural networks](https://arxiv.org/abs/1312.6199) (ICLR 2014) <br/> This paper first introduced the concept of adversarial examples in deep learning, and provided a L-BFGS based attack method.

<a id='Benchmark_Datasets'></a>
## Benchmark Datasets
* [Natural adversarial examples](https://arxiv.org/pdf/1907.07174.pdf) <br/> ImageNet-A dataset.

* [Benchmarking Neural Network Robustness to Common Corruptions and Perturbations](https://arxiv.org/pdf/1903.12261.pdf) (ICLR 2019) <br/> ImageNet-C dataset.

* [Imagenet-trained cnns are biased towards texture; increasing shape bias improves accuracy and robustness](https://arxiv.org/pdf/1811.12231.pdf) (ICLR 2018) <br/> This paper empirically demonstrate that shape-based features lead to more robust models. They also provide the Styled-ImageNet dataset.
