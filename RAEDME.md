# Memorization in Self-supervised Learning Improve Downstream Generalization

*Keywords:* self-supervised learning, memorization, encoders

*TL;DR:* We introduce a formal definition of memorization in self-supervised learning and provide a thorough empirical evaluation that suggests that encoders require memorization to generalize well to downstream tasks.

*Abstract:* Self-supervised learning (SSL) has recently received significant attention due to its ability to train high-performance encoders purely on unlabeled data, often scraped from the internet. This data can still be sensitive and empirical evidence suggests that SSL encoders memorize private information of their training data and can disclose them at inference time. Since existing theoretical definitions of memorization from supervised learning rely on labels, they do not transfer to SSL. To address this gap, we propose a theoretical framework for defining memorization within the context of SSL. Our definition compares the difference in alignment of representations for data points and their augmented views returned by both encoders that were trained on these data points and encoders that were not. Through comprehensive empirical analysis on diverse encoder architectures and datasets, we highlight that even though SSL relies on large datasets and strong augmentations—both known in supervised learning as regularization techniques that reduce overfitting—still significant fractions of training data points experience high memorization. Although this may be disconcerting with regard to privacy, our research demonstrates that the process of memorization is essential for encoders to generalize to different downstream tasks.

## Description of the code

1. Candidate models training. Please first install the requirements.txt for the specific model. Then make sure the train_XXX.py is in the same folder with the model files (in the models folder). Then modify the datapath, savingpath, and other parameters according to your device and experiment needs. The output will be the final models and checkpoints for both candidate and independent ones.

2. Memorization measurement. Once the canary and independent model pairs are trained, use memorization_scores.py to test the memorization score for candidate samples. Make sure the tested candidate sample should be the same as the candidate samples during training. Also modify the datapath, savingpath, model name, and other parameters before using.

3. KNN evaluation of dejavu. Before using, please follow the instruction from https://github.com/facebookresearch/DejaVu to install FFCV-SSL or directly use the command below:

***************************
git clone git@github.com:facebookresearch/FFCV-SSL.git
cd FFCV-SSL
pip install -e .
***************************

Then, use dejavu.py to evaluate the dejavu memory on candidate samples. It's very  ****important**** to keep the data augmentation during testing the same as the one you used during training. For example, to evaluate the MAE model trained under 75% masking ratio, only apply 75% masking ratio as image augmentation during testing. Also modify the datapath, savingpath, model name, and other parameters before using.
 
4. Linear Probing/ Use linear_probing_train.py to do the linear probing for the pretrained models. The default classifier model is a FC-layer. Make sure that this file is under the same folder with the model_XXX.py as well as the model.pt. Also modify the datapath, savingpath, model name, and other parameters before using.

5. Sample removing. Use the remove_samples.py do test the memorization scores for all training samples and remove the top XX percentages accordingly. Also modify the datapath, savingpath, model name, persentage you want to remove, and other parameters before using.