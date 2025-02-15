# Major Credits

Thank you to DeepFindr for the video on creating a relatively simple diffusion model in PyTorch. 
https://www.youtube.com/watch?v=a4Yfz2FxXiY

Thank you to FernandoPC25 for the article on UNETs and inspiring my dynamic architecture - https://medium.com/@fernandopalominocobo/mastering-u-net-a-step-by-step-guide-to-segmentation-from-scratch-with-pytorch-6a17c5916114

Thank you to PvJosue for trying this before me - https://github.com/pvjosue/pytorch_convNd/blob/master/LICENSE

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
stuff through optional parameters. Lastly, I added the ability to use time embeddings and created a diffusion "shell"
model that can sit around the UNET to handle all the denoising steps for a complete diffusion model.

I am still working on adding more and more to it, and currently it is kind of a trash image generator, but the more 
options I add to it, the more customizable I can make everything at the click of a button which is nice.

# Google Colab Link

You can use GPUs for free as of February 2025, but it has a limited runtime. You can just copy my stuff into your own 
notebook if you want. There is also a requirements folder to timestamp what library versions I use.

https://colab.research.google.com/drive/1lOn-XvXf3O6jaqZSc4LVkkmBC5E6EHf7?usp=sharing

After I found out about the limited runtime, I had to create my own local environment. I have an nvidia drive, and
pytorch has resources on how to download the necessary stuff. https://pytorch.org/get-started/locally/

My suggestion is to pick a drive with a decent amount of space you can play around and create a virtual environment,
and when you are installing, pick a place with a good connection. The stuff in my requirements folder took up around
2.5GB of space.
