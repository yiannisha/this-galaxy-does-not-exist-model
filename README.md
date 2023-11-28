# this-galaxy-does-not-exist-model

⚠️ This model is still in training.

# Inspiration
I was inspired to create this model from this [human face generator project](https://thispersondoesnotexist.com/) and this [galaxy stable diffusion model](https://huggingface.co/Supermaxman/hubble-diffusion-2)

# Methods Used
I used the classic GANs approach of training a Generator and a Discriminator model interchangeably, until the output of the Generator model was visually successful.

# Pitfalls
The Generator model is very sensitive to the learning rate hyperparameter, resulting in hitting a lot of local minima during training.
The size of the model itself also results to weights of greater sizes.

A few ways we can optimise are:
  - Learning Rate Scheduling
  - Gradient Clipping for weight reduction
