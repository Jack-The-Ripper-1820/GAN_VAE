# Comparison of GAN and VAE on MNIST and Face Image Generation (Machine Learning Course Project)

WIP - Only the initial GAN on the MNIST has been implemented, still have to optimize it. 

# REPORT

## REPORT 2 

Introduction
Variational autoencoders (VAEs) are a popular generative model used for image generation, anomaly detection, and data compression. VAEs are a type of autoencoder that can learn a low-dimensional representation of high-dimensional data. The low-dimensional representation is known as the latent space, which can be used to generate new data similar to the training data. In this report, I will discuss how I came up with a VAE implementation for image generation using TensorFlow.

Background
Autoencoders are neural networks that learn to reconstruct input data. They consist of two parts: an encoder that maps the input data to a latent representation and a decoder that maps the latent representation to the output data. The latent representation is typically lower dimensional than the input data, and it is learned through a process of minimizing the reconstruction loss between the input data and the output data.

VAEs extend the basic autoencoder by learning a probability distribution over the latent space. The goal is to make the latent space "smooth" so that nearby points in the latent space produce similar output data. This is achieved by learning the mean and variance of the distribution over the latent space, and then sampling from this distribution to generate new data.

Implementation
My implementation of the VAE for image generation is based on the TensorFlow framework. I started with a naive implementation of a basic autoencoder for image reconstruction. I then extended this implementation to include the VAE architecture.

The VAE consists of two parts: an encoder and a decoder. The encoder consists of three convolutional layers followed by a fully connected layer that outputs the mean and variance of the distribution over the latent space. The decoder consists of a fully connected layer followed by three transposed convolutional layers that generate the output image.

To train the VAE, I used the Adam optimizer with a learning rate of 1e-4. I also used the mean squared error as the reconstruction loss and the Kullback-Leibler (KL) divergence as the regularization loss. The total loss is the sum of the reconstruction loss and the KL loss.

The loss function for a VAE consists of two components: the reconstruction loss and the KL divergence loss. The reconstruction loss measures the difference between the original input and the output of the decoder network. The KL divergence loss measures the difference between the distribution of the latent variables and a unit Gaussian distribution.

The reconstruction loss can be computed as the mean squared error (MSE) between the original input and the reconstructed output:

$$\mathcal{L}{\text{recon}} = \frac{1}{N}\sum{i=1}^{N}\left(\boldsymbol{x}_i - \boldsymbol{x}_i^{\text{recon}}\right)^2$$

where $N$ is the number of training examples, $\boldsymbol{x}_i$ is the $i$-th training example, and $\boldsymbol{x}_i^{\text{recon}}$ is the reconstructed output for the $i$-th training example.

The KL divergence loss can be computed as follows:

$$\mathcal{L}{\text{KL}} = -\frac{1}{2N}\sum{i=1}^{N}\left(1 + \log\left(\boldsymbol{\sigma}_i^2\right) - \boldsymbol{\mu}_i^2 - \boldsymbol{\sigma}_i^2\right)$$

where $\boldsymbol{\mu}_i$ and $\boldsymbol{\sigma}_i$ are the mean and standard deviation of the latent variables for the $i$-th training example.

The total loss for the VAE is then the sum of the reconstruction loss and the KL divergence loss:

$$\mathcal{L} = \mathcal{L}{\text{recon}} + \beta \cdot \mathcal{L}{\text{KL}}$$

where $\beta$ is a hyperparameter that controls the weighting of the KL divergence loss.


Trained the VAE on the MNIST dataset, which consists of 60,000 28x28 grayscale images of handwritten digits. Iused a batch size of 128 and trained the model for 30 epochs.

However, it is important to note that the training process is still ongoing and I expect to obtain even better results with more training epochs. The use of VAE has shown promising results in image generation tasks and can be applied in various other domains for generating new data points. In conclusion, this implementation highlights the effectiveness of VAE in generating high-quality images and can serve as a baseline for future research in this field.

## previous report

Generative Adversarial Networks (GANs) are a class of machine learning models designed for generative tasks, such as generating new, synthetic data instances. The main idea behind GANs is to train two neural networks, a generator network and a discriminator network, in a two-player game framework. The generator network aims to generate new, synthetic data instances that are indistinguishable from real data, while the discriminator network aims to distinguish between real data instances and synthetic data instances generated by the generator network.

This code is an implementation of a GAN that generates new, synthetic images of handwritten digits from the MNIST dataset. The generator network is defined using the Keras library, with the TensorFlow backend. The generator network is a sequential model that consists of several dense and convolutional layers. The dense layers are used to learn a compact representation of the noise data, which is then upscaled to generate the synthetic images. The discriminator network is also defined using the Keras library, with the TensorFlow backend. The discriminator network is a convolutional neural network (CNN) that is designed to distinguish between real images and synthetic images.


The MNIST dataset is loaded, and the training images are preprocessed by reshaping them into 28x28x1 arrays and normalizing the pixel values to the range [-1, 1]. The generator and discriminator networks are then trained using the Adam optimization algorithm. The generator network is trained to generate synthetic images that can fool the discriminator network, while the discriminator network is trained to correctly identify real and synthetic images. The loss functions used to train the networks are binary cross-entropy loss functions, which measure the dissimilarity between the predicted probabilities and the true labels.


The code contains a for loop that trains a Generative Adversarial Network (GAN) by alternating between training the generator and discriminator. The generator and discriminator are defined as TensorFlow models and the generator loss and discriminator loss functions are defined outside of the loop.
Inside the loop, the training process occurs in two parts: first, the discriminator is trained on a batch of real images and a batch of fake images generated by the generator. The loss for the discriminator is calculated as the sum of the log loss for the real images and the log loss for the fake images. The gradients of the discriminator with respect to the loss are computed using the TensorFlow GradientTape context and the discriminator's trainable variables are updated using the Adam optimizer.


Next, the generator is trained by passing noise to it and the loss is calculated based on the discriminator's predictions on the generated images. The gradients of the generator with respect to the loss are computed using the TensorFlow GradientTape context and the generator's trainable variables are updated using the Adam optimizer.

tf.GradientTape is a TensorFlow mechanism that allows you to record the operations performed during forward pass and automatically compute the gradients in the backward pass. In this code, tf.GradientTape is used to calculate the gradients of the loss functions with respect to the trainable variables of the discriminator and generator models, so that these gradients can be used to update the weights of the models during training.


Here, there are two separate instances of tf.GradientTape: one for training the discriminator, and another for training the generator. In each case, the with tf.GradientTape() block is used to record the forward pass operations, and the gradients are then calculated using tape.gradient(loss, variables), where loss is the output of the loss function and variables are the trainable variables of the model. The resulting gradients are then passed to the optimizer using optimizer.apply_gradients(zip(grads, variables)).

Finally, the code prints the loss values every 10 epochs, saves the model weights every 20 epochs, and generates and plots fake images every 30 epochs.

This implementation demonstrates the power of GANs in generating new, synthetic data instances, and highlights the importance of using appropriate loss functions and optimization algorithms for training GANs. It can be used as a starting point for further research and development of GANs for a variety of generative tasks, including image generation, audio generation, and text generation.


## Loss Function in Generative Adversarial Networks (GANs)
The loss function in GANs plays a crucial role in determining the quality of the generated images. It is a measure of how well the generator and discriminator are performing in their respective roles.
The loss function in GANs consists of two parts: the generator loss and the discriminator loss. The generator loss measures the difference between the generated images and the ground truth images, while the discriminator loss measures the ability of the discriminator to correctly classify the generated images and the real images.

The generator loss is calculated using the cross-entropy loss function, as follows:

### L_G = -E[log(D(G(z)))]

where G(z) is the generated image, 

D is the discriminator, and 

z is the random noise vector. 

The goal of the generator is to maximize this loss, which means it wants to produce images that the discriminator can't distinguish from the real images.


The discriminator loss is calculated using the cross-entropy loss function as well, as follows:


### L_D = -E[log(D(x))] - E[log(1 - D(G(z)))]

where x is the real image and G(z) is the generated image. 

The goal of the discriminator is to minimize this loss, which means it wants to correctly classify the real and generated images.
In summary, the loss function in GANs plays a crucial role in the training process, as it determines the quality of the generated images and the ability of the discriminator to correctly classify the generated and real images.

This leads to a minmax Loss function between the generator and the discriminator

The Minimax loss function is a crucial component in the training of GANs. It is used to train the generator and discriminator networks in a GAN architecture. The loss function can be seen as a two-player game between the generator and the discriminator, where the generator is trying to generate samples that the discriminator cannot differentiate from real samples, and the discriminator is trying to distinguish real samples from generated ones.

In the game, the generator tries to minimize the loss function, while the discriminator tries to maximize it. The loss function can be written as a min-max optimization problem, where the generator is trying to minimize its negative expected reward, and the discriminator is trying to maximize its expected reward. The expected reward is defined as the probability that the discriminator correctly classifies a sample as either real or generated.

The Minimax loss function can be written as:

### L(G,D) = E_x[log(D(x))] + E_z[log(1-D(G(z)))]

where G is the generator, D is the discriminator, x is a real sample from the dataset, and z is a random noise vector that is used as input to the generator. The first term in the loss function is the expected log-likelihood that the discriminator assigns a high probability to real samples, and the second term is the expected log-likelihood that the discriminator assigns a low probability to generated samples.

By alternating the optimization of the generator and the discriminator, the Minimax loss function trains the two networks to play a minimax game, where the generator tries to produce samples that the discriminator cannot distinguish from real samples, and the discriminator tries to classify samples as accurately as possible. Over time, the generator and discriminator improve, and the generated samples become increasingly realistic.


