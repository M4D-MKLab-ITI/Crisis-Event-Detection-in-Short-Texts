# Leveraging Transformer Self Attention Encoder for Crisis Event Detection in Short Texts

This is the python implementation of the models presented in "Leveraging Transformer Self Attention Encoder for Crisis Event Detection in Short Texts" (2022) (Pantelis Kyriakidis, Despoina Chatzakou, Theodora TsikrikaStefanos Vrochidis, Ioannis Kompatsiaris). 

The paper introduces novel architectures that surpass the performance of the previous SOTA (Multichannel Convolutional Neural Network, MCNN) on CrisisLexT26.

To do that, we leverage Transformer Self Attention Encoders that are able to extract features from the input embeddings unaffected from the word n-grams (CNNs) or the distance between separate words (RNNs). By paying attention to the input, each word distributes its attention over all the other words. Residually adding the attention context to the input, reduces the noise and enhances important contextualized features. 

Read more about Transformers and their Self Attention Encoder in the original paper ["Attention is all you need"](https://arxiv.org/abs/1706.03762) and this educational [blog post](https://jalammar.github.io/illustrated-transformer/).

The following figure illustrates the architecture of the best performing model, AD-MCNN:

<p align="center">
  <img src="admcnn.png" />
</p>
![AD-MCNN_img](admcnn.png "AD-MCNN")
