# Welcome

This is some practice I did with creating a dynamic UNET architecture model. It started when I wanted to learn stable
diffusion and with all the tutorials I saw, it always involved some kind of UNET architecture that looked confusing
upon first glance. 

It made me quit for a second and after a couple of months, I gave it another shot, starting with just the basic UNET
at first. Starting with just a UNET let me learn how that worked, and after getting something to work, I created a
structure that created the encoder and decoder as a list of blocks based on whatever list of channels I fed it. That
way, I wouldn't have to manually add blocks to my model every time I wanted to alter it.

From there, I learned what Attention was and created my own simple version of that with some guidance. From there, I
beefed up the model with some batch modularization, activation functions, and added the ability to enable and disable
stuff through optional parameters. I am using the MNIST dataset to mess around with it and see how it works.

I'm currently working on the diffusion part of it, but feel free to explore. Here is the colab I am using since
my computer does not have cuda enabled for pytorch and installing it would...be a pain I am not dealing with right now.

# Google Colab

You can use GPUs for free as of February 2025. You can just copy my stuff into your own notebook if you want. There
is also a requirements folder to timestamp what library versions I use.

https://colab.research.google.com/drive/1lOn-XvXf3O6jaqZSc4LVkkmBC5E6EHf7?usp=sharing
