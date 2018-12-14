# Papers-of-Robust-ML
 Related papers for robust machine learning

## Contents 
- <a href="#General">General Defenses</a><br>
- <a href="#Detection">Adversarial Detection</a><br>
- <a href="#Verification">Verification</a><br>
- <a href="#Theoretical">Theoretical Analysis</a><br>

## General Defenses
<a id='General'></a>
* [A Spectral View of Adversarially Robust Features](http://papers.nips.cc/paper/8217-a-spectral-view-of-adversarially-robust-features.pdf) (NeurIPS 2018) <br/> Given the entire dataset X, use the eigenvectors of spectral graph as robust features. [[Appendix](http://papers.nips.cc/paper/8217-a-spectral-view-of-adversarially-robust-features-supplemental.zip)]
## Adversarial Detection
<a id='Detection'></a>
* [A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks](http://papers.nips.cc/paper/7947-a-simple-unified-framework-for-detecting-out-of-distribution-samples-and-adversarial-attacks.pdf) (NeurIPS 2018) <br/> Fit a GDA on learned features, and use Mahalanobis distance as the detection metric.
## Verification
<a id='Verification'></a>
* [Provable Defenses against Adversarial Examples via the Convex Outer Adversarial Polytope](https://arxiv.org/pdf/1711.00851.pdf) (ICML 2018) <br/> By robust optimization (via a linear program), they can get a point-wise bound of robustness, where no adversarial example exists in the bound. Experiments are done on MNIST.
## Theoretical Analysis
<a id='Theoretical'></a>
* [Adversarial Vulnerability for Any Classifier](http://papers.nips.cc/paper/7394-adversarial-vulnerability-for-any-classifier.pdf) (NeurIPS 2018) <br/> Uniform upper bound of robustness for any classifier on the data sampled from smooth genertive models.
