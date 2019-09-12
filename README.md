# Papers-of-Robust-ML
 Related papers for robust machine learning

## Contents 
- <a href="#General_training">General Defenses (training phase)</a><br>
- <a href="#General_inference">General Defenses (inference phase)</a><br>
- <a href="#Detection">Adversarial Detection</a><br>
- <a href="#Verification">Verification</a><br>
- <a href="#Theoretical">Theoretical Analysis</a><br>
- <a href="#Empirical">Empirical Analysis</a><br>

<a id='General_training'></a>
## General Defenses (training phase)
* [You Only Propagate Once: Accelerating Adversarial Training via Maximal Principle](https://arxiv.org/pdf/1905.00877.pdf) (NeurIPS 2019) <br/> This paper provides a fast method for adversarial training from the perspective of optimal control.

* [Rethinking Softmax Cross-Entropy Loss for Adversarial Robustness ](https://arxiv.org/pdf/1905.10626.pdf) <br/> This paper rethink the drawbacks of softmax cross-entropy in the adversarial setting, and propose the MMC method to induce high-density regions in the feature space.

* [Interpolated Adversarial Training: Achieving Robust Neural Networks without Sacrificing Accuracy](https://arxiv.org/pdf/1906.06784.pdf) <br/> This paper introduces the mixup method into adversarial training to improve the model performance on clean images.

* [Theoretically Principled Trade-off between Robustness and Accuracy](https://arxiv.org/pdf/1901.08573.pdf) (ICML 2019) <br/> A variant of adversarial training: TRADES, which won the defense track of NeurIPS 2018 Adversarial Competation.

* [Robust Decision Trees Against Adversarial Examples](http://web.cs.ucla.edu/~chohsieh/ICML_2019_TreeAdvAttack.pdf) (ICML 2019) <br/> A method to enhance the robustness of tree models, including GBDTs.

* [Adversarial Training for Free!](https://arxiv.org/pdf/1904.12843.pdf) (NeurIPS 2019) <br/> A fast method for adversarial training, which shares the back-propogation gradients of updating weighs and crafting adversarial examples.

* [Improving Adversarial Robustness via Promoting Ensemble Diversity](https://arxiv.org/pdf/1901.08846.pdf) (ICML 2019) <br/> Previous work constructs ensemble defenses by individually enhancing each memeber and then directly average the predictions. In this work, the authors propose the adaptive diversity promoting (ADP) to further improve the robustness by promoting the ensemble diveristy, as an orthogonal methods compared to other defenses.

* [Ensemble Adversarial Training- Attacks and Defenses](https://arxiv.org/pdf/1705.07204.pdf) (ICLR 2018) <br/> Ensemble adversarial training use sevel pre-trained models, and in each training batch, they randomly select one of the currently trained model or pre-trained models to craft adversarial examples.

* [Max-Mahalanobis Linear Discriminant Analysis Networks](http://proceedings.mlr.press/v80/pang18a/pang18a.pdf) (ICML 2018) <br/> This is one of our work. We explicitly model the feature distribution as a Max-Mahalanobis distribution (MMD), which has max margin among classes and can lead to guaranteed robustness.

* [A Spectral View of Adversarially Robust Features](http://papers.nips.cc/paper/8217-a-spectral-view-of-adversarially-robust-features.pdf) (NeurIPS 2018) <br/> Given the entire dataset X, use the eigenvectors of spectral graph as robust features. [[Appendix](http://papers.nips.cc/paper/8217-a-spectral-view-of-adversarially-robust-features-supplemental.zip)]

* [Deep Defense: Training DNNs with Improved Adversarial Robustness](http://papers.nips.cc/paper/7324-deep-defense-training-dnns-with-improved-adversarial-robustness.pdf) (NeurIPS 2018) <br/> They follow the linear assumption in DeepFool method. DeepDefense pushes decision boundary away from those correctly classified, and pull decision boundary closer to those misclassified.

* [Feature Denoising for Improving Adversarial Robustness](https://arxiv.org/pdf/1812.03411.pdf) (CVPR 2019) <br/> This paper applies non-local neural network and large-scale adversarial training with 128 GPUs (with training trick in 'Accurate, large minibatch SGD: Training ImageNet in 1
hour'), which shows large improvement than previous SOTA trained with 50 GPUs.

<a id='General_inference'></a>
## General Defenses (inference phase)
* [Barrage of Random Transforms for Adversarially Robust Defense](http://openaccess.thecvf.com/content_CVPR_2019/papers/Raff_Barrage_of_Random_Transforms_for_Adversarially_Robust_Defense_CVPR_2019_paper.pdf) (CVPR 2019) <br/> This paper applies a set of different random transformations as an off-the-shelf defense.

* [Mitigating Adversarial Effects Through Randomization](https://arxiv.org/pdf/1711.01991.pdf) (ICLR 2018) <br/> Use random resizing and random padding to disturb adversarial examples, which won the 2nd place in th defense track of NeurIPS 2017 Adversarial Competation.

* [Countering Adversarial Images Using Input Transformations](https://arxiv.org/pdf/1711.00117.pdf) (ICLR 2018) <br/> Apply bit-depth reduction, JPEG compression, total variance minimization and image quilting as input preprocessing to defend adversarial attacks.

<a id='Detection'></a>
## Adversarial Detection
* [Towards Robust Detection of Adversarial Examples](http://papers.nips.cc/paper/7709-towards-robust-detection-of-adversarial-examples.pdf) (NeurIPS 2018) <br/> This is one of our work. We train the networks with reverse cross-entropy (RCE), which can map normal features to low-dimensional manifolds, and then detectors can better separate between adversarial examples and normal ones.

* [A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks](http://papers.nips.cc/paper/7947-a-simple-unified-framework-for-detecting-out-of-distribution-samples-and-adversarial-attacks.pdf) (NeurIPS 2018) <br/> Fit a GDA on learned features, and use Mahalanobis distance as the detection metric.

* [Robust Detection of Adversarial Attacks by Modeling the Intrinsic Properties of Deep Neural Networks](http://papers.nips.cc/paper/8016-robust-detection-of-adversarial-attacks-by-modeling-the-intrinsic-properties-of-deep-neural-networks.pdf) (NeurIPS 2018) <br/> They fit a GMM on learned features, and use the probability as the detection metric.

<a id='Verification'></a>
## Verification
* [Automated Verification of Neural Networks: Advances, Challenges and Perspectives](https://arxiv.org/pdf/1805.09938.pdf) <br/> This paper provides an overview of main verification methods, and introduces previous work on combining automated verification with machine learning. They also give some insights on future tendency of the combination between these two domains.

* [Provable Defenses against Adversarial Examples via the Convex Outer Adversarial Polytope](https://arxiv.org/pdf/1711.00851.pdf) (ICML 2018) <br/> By robust optimization (via a linear program), they can get a point-wise bound of robustness, where no adversarial example exists in the bound. Experiments are done on MNIST.

* [Scaling Provable Adversarial Defenses](http://papers.nips.cc/paper/8060-scaling-provable-adversarial-defenses.pdf) (NeurIPS 2018) <br/> They add three tricks to improve the scalability of previously proposed method. Experiments are done on MNIST and CIFAR-10.

<a id='Theoretical'></a>
## Theoretical Analysis
* [Adversarial Examples Are a Natural Consequence of Test Error in Noise](https://arxiv.org/pdf/1901.10513.pdf) (ICML 2019) <br/> This paper connects the relation between the general corruption robustness and the adversarial robustness, and recommand the adversarial defenses methods to be also tested on general-purpose noises.

* [Adversarial Examples Are Not Bugs, They Are Features](https://arxiv.org/pdf/1905.02175.pdf) <br/> They claim that adversarial examples can be directly attributed to the presence of non-robust features, which are highly predictive but locally quite sensitive.

* [On Evaluating Adversarial Robustness](https://arxiv.org/pdf/1902.06705.pdf) <br/> Some analyses on how to correctly evaluate the robustness of adversarial defenses.

* [Robustness of Classifiers:from Adversarial to Random Noise](http://papers.nips.cc/paper/6331-robustness-of-classifiers-from-adversarial-to-random-noise.pdf) (NeurIPS 2016)

* [Adversarial Vulnerability for Any Classifier](http://papers.nips.cc/paper/7394-adversarial-vulnerability-for-any-classifier.pdf) (NeurIPS 2018) <br/> Uniform upper bound of robustness for any classifier on the data sampled from smooth genertive models.

* [Adversarially Robust Generalization Requires More Data](http://papers.nips.cc/paper/7749-adversarially-robust-generalization-requires-more-data.pdf) (NeurIPS 2018) <br/> This paper show that robust generalization requires much more sample complexity compared to standard generlization on two simple data distributional models. 

<a id='Empirical'></a>
## Empirical Analysis
* [Adversarial Example Defenses: Ensembles of Weak Defenses are not Strong](https://arxiv.org/pdf/1706.04701.pdf) <br/> This paper tests some ensemble of existing detection-based defenses, and claim that these ensemble defenses could still be evade by white-box attacks.
