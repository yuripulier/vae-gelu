# Variational Autoencoder - An Approach Using GELUs 

## 1. Introduction

<div align="justify">
&nbsp;&nbsp; Even over the years, the Rectified Linear Unit (ReLU) activation function remains a competitive approach to creating Deep Learning models because it is faster and demonstrates better convergence compared to sigmoid. For generate image problems such as Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs), a variation called LeakyReLU (LReLU) has been shown to be more efficient.
<br><br>
&nbsp;&nbsp; The idea of this work is to show the improvements in a VAE network caused by the use of GELUs, replacing the LReLU used in the base model. GELUs were introduced in the work of Hendricks D. - Gaussian Error Linear Units (GELUs) (2016), corresponding to better results in computer vision tasks, natural language processing and automatic speech recognition compared to models with ReLUs or ELUs.
<br><br>  
&nbsp;&nbsp; The code is based on the approach taken by Foster at <a href="https://github.com/davidADSP/GDL_code">github.com/davidADSP/GDL_code</a>.
</div>

## 2. Variational Autoencoder

<div align="justify">
&nbsp;&nbsp; Before getting into the concept of the Variational Autoencoder, it is important to highlight the operation of an Autoencoder (AE), which is a neural network composed of two parts:
<br><br>
&nbsp;&nbsp;&nbsp;&nbsp; 1. The encoder network compresses high-dimensional input data into high-dimensional data lower dimension. <br>
&nbsp;&nbsp;&nbsp;&nbsp; 2. The decoder network decompresses the low-dimensional representation, reconstructing the input data.
<br><br>

&nbsp;&nbsp; Foster (2019) explains that the autoencoder network is trained to find weights that minimize the loss between the original input and the input reconstruction. The representation vector (representation vector) shown in Figure 13 demonstrates the compression of the input image in a smaller dimension called latent space, and it is from there that the decoder starts the reconstruction to obtain the input image. By choosing a point in the latent space represented to the right of the image, the decoder should be able to generate images within the distribution of the original data. However, we can notice that, depending on the point chosen in this two-dimensional latent space, the decoder will not be able to generate the images correctly. There is also a problem of lack of symmetry, which we notice by looking at the y axes of latent space and we see that the number of points in 𝑦 < 0 is much greater than in 𝑦 > 0 and there is a large concentration at the point (0, 0). Finally, through the coloring, we noticed that some digits are represented in very small and overlapping areas.
<br><br>
&nbsp;&nbsp; In addition to the aforementioned problems, the decoder must be able to generate different types of digits. According to Foster (2019), if the autoencoder is too free to choose how it will use the latent space to encode the images, there will be huge gaps between groups of similar points without these spaces between the numbers being able to generate images correctly. The Variational Autoencoder is a model that can be used to solve these problems demonstrated in an autoencoder to become a generative model. In an autoencoder, each image is mapped directly as a point in latent space, while in a VAE, each image is mapped as a multivariate normal distribution around a point.
<br><br>
&nbsp;&nbsp; The encoder only cares about mapping the input to a mean vector and a variance vector, not worrying about the covariance (numerical interdependence between two random variables) between the dimensions. As the output of the neural network can be any real number in the range (−∞, ∞), it is preferable to map the variance logarithm (FOSTER, 2019).
</div>

## 3. Gaussian Error Linear Units (GELUs)

<div align="justify">
&nbsp;&nbsp; To better understand the complete formulation of the GELUs access the original paper at <a href="https://arxiv.org/pdf/1606.08415.pdf">Gaussian Error Linear Units (GELUs)</a>.
</div>

## 4. Experiments

### 4.1 Use of GELUs
<div align="justify">
&nbsp;&nbsp; We train a Variational Autoencoder on MNIST. We use a network with layers of width 28, 14, 7, 2, 7, 14, 28, in that order. We use the Adam optimizer, a batch size of 32 and loss is the root mean squared error. Our learning rate is 0.0005 and we trained for 200 epochs. We do not use dropout or any normalization layer like batch normalization or layer normalization. It might be interesting to test batch normalization in this model, as neuron inputs tend to follow a normal distribution, especially in this case. These tests using dropout and batch normalization will be done in future works.
<br><br>
&nbsp;&nbsp; In our experiment, we did tests with the base model using LReLU, a model replacing the LReLUs by GELUs in the encoder, a model doing the same replacement only in the decoder and a last model replacing both the encoder and the decoder (full). Left are reconstruction loss curves and right are KL loss curves. Light, thin curves correspond to test set log losses. In the last figure, the general loss of VAE.
<br><br>
<div align="center"><img src="https://github.com/yuripulier/vae-gelu/blob/main/img/vae_losses.png", height= 300, width=900 /></div>
<div align="center"><img src="https://github.com/yuripulier/vae-gelu/blob/main/img/model_loss.png", height= 300, width=450 /></div>
<br>
&nbsp;&nbsp; We can observe that replacing the activation layers only in the encoder or decoder already obtains a better performance than the base model. It is important to note that when using GELUs in the encoder, there was an improvement compared to using them in the decoder. Possibly due to the behavior of the decoder input which tends to be similar to a normal distribution due to the regularization of the encoder through the Kullback-Leibler Divergence, inserting the data representations as a normal distribution in the latent space. In the Full-GELU model, in which we replaced all LReLUs with GELUs, we noticed a significant improvement in relation to the base model and also to the others.
</div>

### 4.2 Use of Dropout
<div align="justify">
&nbsp;&nbsp; We trained the same network setup as the experiment shown in 4.1, adding only a dropout layer (rate = 0.25) after all the GELUs activation layers. We can see that there was a deterioration in the model in the training stage, but the result in validation stage is optimistic reducing overfitting problems obtaining lower loss values than in training.
<br><br>
<div align="center"><img src="https://github.com/yuripulier/vae-gelu/blob/main/img/vae_losses_dropout.png", height= 300, width=450 /></div>
<br>
&nbsp;&nbsp; Perhaps the dropout rate may have influenced a deterioration in relation to the Full-GELU model. Therefore, in the future we will test different dropout values that maintain the overfitting improvement and optimize the result of the Full-GELU model.
</div>  
  
### 4.3 Use of Batch Normalization

#### 4.3.1 After GELUs
<div align="justify">
&nbsp;&nbsp; We trained the same network setup as the experiment shown in 4.1, adding a batch normalization layer before all the GELUs activation layers. We observed a good improvement in network training and a reasonable improvement in validation. Both converge.
<br><br>
<div align="center"><img src="https://github.com/yuripulier/vae-gelu/blob/main/img/vae_losses_bn.png", height= 300, width=450 /></div>
<br>
</div>

#### 4.3.2 Before GELUs
<div align="justify">
&nbsp;&nbsp; This time we used the batch normalization layers before the GELUs activation layers. We observed a better results then the experiment 4.2.1, evidenced after epoch 100. The training and validation curve lines converge.
<br><br>
<div align="center"><img src="https://github.com/yuripulier/vae-gelu/blob/main/img/vae_losses_bn_before_gelu.png", height= 300, width=450 /></div>
<br>
</div>

## 5. Conclusion
<div align="justify"> 
&nbsp;&nbsp; The use of GELUs in a VAE has a superior performance compared to LReLU or ReLU used in most imaging models, therefore it becomes an excellent alternative for nonlinearity in this type of model. The best results were found for the model with batch normalization being used before the GELUs, both in the encoder and in the decoder. This is due to the normalization performed on the input data, creating batches that follow a normal distribution, favoring later GELUs.
</div>

## References
  FOSTER, D. Generative Deep Learning: Teaching Machines to Paint, Write, Compose, and Play. 1ª Edição. O’Reilly Media: Michele Cronin, 2019.
<br><br>
  HENDRYCKS, Dan; GIMPEL, Kevin. Gaussian error linear units (GELUs). arXiv preprint arXiv:1606.08415, 2016.
