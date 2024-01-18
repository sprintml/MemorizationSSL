# Memorization in Self-Supervised Learning Improves Downstream Generalization

Authors: Wenhao Wang, Muhammad Ahmad Kaleem, Adam Dziedzic, Michael Backes, Nicolas Papernot, Franziska Boenisch

## Keywords

self-supervised learning, memorization, encoders, generalization, ssl

## TL;DR: 

We introduce a formal definition for memorization in self-supervised learning and provide a thorough empirical evaluation that suggests that encoders require memorization to generalize well to downstream tasks.
Abstract:

## Abstract
Self-supervised learning (SSL) has recently received significant attention due to its ability to train high-performance encoders purely on unlabeled data---often scraped from the internet. This data can still be sensitive and empirical evidence suggests that SSL encoders memorize private information of their training data and can disclose them at inference time. Since existing theoretical definitions of memorization from supervised learning rely on labels, they do not transfer to SSL. To address this gap, we propose a framework for defining memorization within the context of SSL. Our definition compares the difference in alignment of representations for data points and their augmented views returned by both encoders that were trained on these data points and encoders that were not. Through comprehensive empirical analysis on diverse encoder architectures and datasets we highlight that even though SSL relies on large datasets and strong augmentations---both known in supervised learning as regularization techniques that reduce overfitting---still significant fractions of training data points experience high memorization. Through our empirical results, we show that this memorization is essential for encoders to achieve higher generalization performance on different downstream tasks.

URL to the paper: https://openreview.net/forum?id=KSjPaXtxP8
