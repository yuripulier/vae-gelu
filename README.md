# Variational Autoencoder - An Approach Using GELUs 
## 1. Introduction

<div align="justify">
&nbsp;&nbsp; Even over the years, the Rectified Linear Unit (ReLU) activation function remains a competitive approach to creating Deep Learning models because it is faster and demonstrates better convergence compared to sigmoid. For generate image problems such as Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs), a variation called LeakyReLU (LReLU) has been shown to be more efficient.
<br><br>
&nbsp;&nbsp; The idea of this work is to show the improvements in a VAE network caused by the use of GELUs, replacing the LReLU used in the base model. GELUs were introduced in the work of Hendricks D. - Gaussian Error Linear Units (GELUs) (2020), corresponding to better results in computer vision tasks, natural language processing and automatic speech recognition compared to models with ReLUs or ELUs.
</div>

## 2. Variational Autoencoder

<div align="justify">
&nbsp;&nbsp; Before getting into the concept of the Variational Autoencoder, it is important to highlight the operation of an Autoencoder (AE), which is a neural network composed of two parts:
<br><br>
&nbsp;&nbsp;&nbsp;&nbsp; 1. The encoder network compresses high-dimensional input data into high-dimensional data lower dimension. <br>
&nbsp;&nbsp;&nbsp;&nbsp; 2. The decoder network decompresses the low-dimensional representation, reconstructing the input data.
<br><br>

&nbsp;&nbsp; Foster (2019) explains that the autoencoder network is trained to find weights that minimize the loss between the original input and the input reconstruction. The representation vector (representation vector) shown in Figure 13 demonstrates the compression of the input image in a smaller dimension called latent space, and it is from there that the decoder starts the reconstruction to obtain the input image. By choosing a point in the latent space represented to the right of the image, the decoder should be able to generate images within the distribution of the original data. However, we can notice that, depending on the point chosen in this two-dimensional latent space, the decoder will not be able to generate the images correctly. There is also a problem of lack of symmetry, which we notice by looking at the y axes of latent space and we see that the number of points in 𝑦 < 0 is much greater than in 𝑦 > 0 and there is a large concentration at the point (0, 0 ). Finally, through the coloring, we noticed that some digits are represented in very small and overlapping areas.
<br><br>
&nbsp;&nbsp; In addition to the aforementioned problems, the decoder must be able to generate different types of digits. According to Foster (2019), if the autoencoder is too free to choose how it will use the latent space to encode the images, there will be huge gaps between groups of similar points without these spaces between the numbers being able to generate images correctly. The Variational Autoencoder is a model that can be used to solve these problems demonstrated in an autoencoder to become a generative model. In an autoencoder, each image is mapped directly as a point in latent space, while in a VAE, each image is mapped as a multivariate normal distribution around a point.
<br><br>
&nbsp;&nbsp; The encoder only cares about mapping the input to a mean vector and a variance vector, not worrying about the covariance (numerical interdependence between two random variables) between the dimensions. As the output of the neural network can be any real number in the range (−∞, ∞), it is preferable to map the variance logarithm (FOSTER, 2019).
</div>

## 3. Gaussian Error Linear Units (GELUs)

<div align="justify">

&nbsp;&nbsp; 


</div>


