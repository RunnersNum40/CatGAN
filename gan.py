import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import glob
from sklearn.utils import shuffle
from PIL import Image
import time

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from keras.datasets.mnist import load_data

from keras.layers import Dense, Reshape, Conv2D, Conv2DTranspose, Dropout, Input, merge
from keras.layers.core import Activation, Flatten
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.advanced_activations import LeakyReLU

from keras.models import Model, load_model, Sequential
from keras.optimizers import SGD, Adam, RMSprop
from keras.initializers import RandomNormal

import keras.backend as K
from tensorflow.python.keras.backend import set_session
import tensorflow as tf

K.image_data_format()
K.set_image_data_format("channels_last")

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


# A function to normalize image pixels.
def norm_img(img):
    """A function to Normalize Images.
    Input:
        img : Original image as numpy array.
    Output: Normailized Image as numpy array
    """
    img = img/127.5 - 1
    return img


def denorm_img(img):
    """A function to Denormailze, i.e. recreate image from normalized image
    Input:
        img : Normalized image as numpy array.
    Output: Original Image as numpy array
    """
    img = (img+1)*127.5
    return img.astype(np.uint8) 

(images1, _), (images2, _) = load_data()

images1.reshape(28, 28, 60000)
images2.reshape(28, 28, 10000)

images = images1
def sample_from_dataset(batch_size, image_shape, data_dir=None):
    """Create a batch of image samples by sampling random images from a data directory.
    Resizes the image using image_shape and normalize the images.
    Input:
        batch_size : Sample size required
        image_size : Size that Image should be resized to
        data_dir : Path of directory where training images are placed.

    Output:
        sample : batch of processed images 
    """
    sample_dim = (batch_size, ) + image_shape
    sample = np.empty(sample_dim, dtype=np.float32)
    all_data_dirlist = list(glob.glob(data_dir))
    sample_imgs_paths = np.random.choice(all_data_dirlist, batch_size)
    for index, img_filename in enumerate(sample_imgs_paths):
        image = Image.open(img_filename)
        image = image.resize(image_shape[:-1])
        image = image.convert("RGB") 
        image = np.asarray(image)
        image = norm_img(image)
        sample[index, ...] = image
    return sample


def savegrid(ims, rows=None, cols=None, fill=True, showax=False):
    if rows is None != cols is None:
        raise ValueError("Set either both rows and cols or neither.")

    if rows is None:
        rows = len(ims)
        cols = 1

    gridspec_kw = {"wspace": 0, "hspace": 0} if fill else {}
    fig,axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw)

    if fill:
        bleed = 0
        fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

    for ax,im in zip(axarr.ravel(), ims):
        ax.imshow(im)
        if not showax:
            ax.set_axis_off()

    kwargs = {"pad_inches": .01} if fill else {}
    fig.savefig("faces.png", **kwargs)


def save_img_batch(img_batch, img_save_dir):
    """Takes as input a image batch and a img_save_dir and saves 16 images from the batch in a 4x4 grid in the img_save_dir
    """
    plt.close("all")
    plt.figure(figsize=(16,16))
    gs1 = gridspec.GridSpec(4, 4)
    gs1.update(wspace=0, hspace=0)
    rand_indices = np.random.choice(img_batch.shape[0],16,replace=False)
    for i in range(16):
        ax1 = plt.subplot(gs1[i])
        ax1.set_aspect("equal")
        rand_index = rand_indices[i]
        image = img_batch[rand_index, :,:,:]
        fig = plt.imshow(denorm_img(image))
        plt.axis("off")
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig(img_save_dir,bbox_inches="tight",pad_inches=0)
    # plt.show() 


def gen_noise(batch_size, noise_shape):
    """ Generates a numpy vector sampled from normal distribution of shape                                (batch_size, noise_shape)
    Input:
        batch_size : size of batch
        noise_shape: shape of noise vector, normally kept as 100 
    Output:a numpy vector sampled from normal distribution of shape                                  (batch_size, noise_shape)     
    """
    return np.random.randn(batch_size, *noise_shape)


def get_gen_normal(noise_shape, conv_layers=4):
    """ This function takes as input shape of the noise vector and creates the Keras generator    architecture.
    """
    kernel_init = "glorot_uniform"

    generator = Sequential(name="Generator")

    generator.add(Input(shape=noise_shape))
    generator.add(Conv2DTranspose(filters=512, kernel_size=(4, 4), strides=(1, 1), padding="valid", data_format="channels_last", kernel_initializer=kernel_init))
    generator.add(BatchNormalization(momentum=0.5))
    generator.add(LeakyReLU(0.2))

    for i in range(conv_layers-2, -1, -1):
        generator.add(Conv2DTranspose(filters=64*2**i, kernel_size=(4, 4), strides=(2, 2), padding="same", data_format="channels_last", kernel_initializer=kernel_init))
        generator.add(BatchNormalization(momentum=0.5))
        generator.add(LeakyReLU(0.2))

    # conv 2D layer 1.
    generator.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", data_format="channels_last", kernel_initializer=kernel_init))
    generator.add(BatchNormalization(momentum = 0.5))
    generator.add(LeakyReLU(0.2))

    # Final Transpose 2D conv layer 5 to generate final image. Filter size 3 for 3 image channel
    generator.add(Conv2DTranspose(filters=3, kernel_size=(4, 4), strides=(2, 2), padding="same", data_format="channels_last", kernel_initializer=kernel_init))

    # Tanh activation to get final normalized image
    generator.add(Activation("tanh"))

    # defining the optimizer and compiling the generator model.
    gen_opt = Adam(lr=0.0002, beta_1=0.5)
    generator.compile(loss="binary_crossentropy", optimizer=gen_opt, metrics=["accuracy"])
    generator.summary()
    return generator


def get_disc_normal(image_shape=(64, 64, 3), conv_layers=4):
    dropout_prob = 0.4
    kernel_init = "glorot_uniform"

    discriminator = Sequential(name="Discriminator")
    discriminator.add(Input(shape=image_shape))

    for i in range(conv_layers):
        discriminator.add(Conv2D(filters=64*2**i, kernel_size=(4, 4), strides=(2, 2), padding="same", data_format="channels_last", kernel_initializer=kernel_init))
        discriminator.add(BatchNormalization(momentum = 0.5))
        discriminator.add(LeakyReLU(0.2))

    # Flatten
    discriminator.add(Flatten())
    # Dense Layer
    discriminator.add(Dense(1))
    # Sigmoid Activation
    discriminator.add(Activation("sigmoid"))
    # Optimizer and Compiling model
    dis_opt = Adam(lr=0.0002, beta_1=0.5)
    discriminator.compile(loss="binary_crossentropy", optimizer=dis_opt, metrics=["accuracy"])
    # discriminator.summary()
    return discriminator


class GAN:
    def __init__(self, image_shape, noise_shape, discriminator_save=None, generator_save=None):
        self.image_shape = image_shape
        self.noise_shape = noise_shape

        self.discriminator = get_disc_normal(self.image_shape, 4) if discriminator_save == None else load_model(discriminator_save)
        self.generator = get_gen_normal(self.noise_shape, int(np.log2(image_shape[0]))-2) if generator_save == None else load_model(generator_save)

        self.discriminator.trainable = False

        # Optimizer for the GAN
        opt = Adam(lr=0.00015, beta_1=0.5) #same as generator
        # Input to the generator
        gen_inp = Input(shape=self.noise_shape)

        GAN_inp = self.generator(gen_inp)
        GAN_opt = self.discriminator(GAN_inp)

        # Final GAN
        self.gan = Model(inputs=gen_inp, outputs=GAN_opt)
        self.gan.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

        self.print_status = False



    def train(self, data_dir, num_steps=1000, batch_size=64, save_intervals=(10, 500), save_model_dir=None, img_save_dir=None, log_dir=None):
        # Use a fixed noise vector to see how the GAN Images transition through time on a fixed noise. 
        fixed_noise = gen_noise(16, self.noise_shape)

        # To keep Track of losses
        self.avg_disc_fake_loss = []
        self.avg_disc_real_loss = []
        self.avg_GAN_loss = []

        # We will run for num_steps iterations
        for step in range(num_steps): 
            tot_step = step
            if self.print_status: print("Begin step: ", tot_step)
            # to keep track of time per step
            step_begin_time = time.time() 

            # sample a batch of normalized images from the dataset
            real_data_X = sample_from_dataset(batch_size, self.image_shape, data_dir=data_dir)

            # Genearate noise to send as input to the generator
            noise = gen_noise(batch_size, self.noise_shape)

            # Use generator to create(predict) images
            fake_data_X = self.generator.predict(noise)

            # Save predicted images from the generator every 10th step
            if img_save_dir != None and (tot_step % save_intervals[0]) == 0:
                save_img_batch(fake_data_X, f"{img_save_dir}{tot_step:05d}_image.png")

            # Create the labels for real and fake data. We don"t give exact ones and zeros but add a small amount of noise. This is an important GAN training trick
            real_data_Y = np.random.random_sample(batch_size)*0.2/np.log(tot_step+2)+1
            fake_data_Y = np.random.random_sample(batch_size)*0.2/np.log(tot_step+2)

            real_data_Y[batch_size//16:] -= 1
            fake_data_Y[batch_size//16:] += 1

            # train the discriminator using data and labels
            self.discriminator.trainable = True
            self.generator.trainable = False

            # Training Discriminator seperately on real data
            dis_metrics_real = self.discriminator.train_on_batch(real_data_X, real_data_Y) 

            # training Discriminator seperately on fake data
            dis_metrics_fake = self.discriminator.train_on_batch(fake_data_X, fake_data_Y) 

            # Save the losses to plot later
            self.avg_disc_fake_loss.append(dis_metrics_fake[0])
            self.avg_disc_real_loss.append(dis_metrics_real[0])

            # Train the generator using a random vector of noise and its labels (1"s with noise)
            self.generator.trainable = True
            self.discriminator.trainable = False

            GAN_X = gen_noise(batch_size, self.noise_shape)
            GAN_Y = real_data_Y

            gan_metrics = self.gan.train_on_batch(GAN_X, GAN_Y)

            # Log results by opening a file in append mode
            log_dir = log_dir if log_dir != None else img_save_dir
            message = f"Step: {tot_step} Disc: real loss: {dis_metrics_real[0]} fake loss: {dis_metrics_fake[0]} GAN loss: {gan_metrics[0]}\n"
            text_file = open(log_dir+"\\training_log.txt", "a")
            text_file.write(message)
            text_file.close()
            if self.print_status: print(f"Step: {tot_step} Disc: real loss: {dis_metrics_real[0]} fake loss: {dis_metrics_fake[0]} GAN loss: {gan_metrics[0]}\n")

            # save GAN loss to plot later
            self.avg_GAN_loss.append(gan_metrics[0])

            end_time = time.time()
            diff_time = int(end_time - step_begin_time)

            # save model at every 500 steps
            if tot_step > 0 and (tot_step % save_intervals[1]) == 0 or step == num_steps-1:
                print("-----------------------------------------------------------------")
                print(f"Average Disc_fake loss: {np.mean(self.avg_disc_fake_loss)}") 
                print(f"Average Disc_real loss: {np.mean(self.avg_disc_real_loss)}") 
                print(f"Average GAN loss: {np.mean(self.avg_GAN_loss)}")
                print("-----------------------------------------------------------------")
                self.discriminator.trainable = False
                self.generator.trainable = False
                # predict on fixed_noise
                noise = gen_noise(batch_size, self.noise_shape)
                fixed_noise_generate = self.generator.predict(noise)
                step_num = str(tot_step).zfill(4)
                save_img_batch(fixed_noise_generate, img_save_dir+step_num+"fixed_image.png")
                save_model_dir = save_model_dir if save_model_dir != None else log_dir
                self.generator.save(save_model_dir+str(tot_step)+"_GENERATOR_weights_and_arch.hdf5")
                self.discriminator.save(save_model_dir+str(tot_step)+"_DISCRIMINATOR_weights_and_arch.hdf5")


    def generate_images(self, save_dir):
        noise = gen_noise(batch_size, self.noise_shape)
        fake_data_X = self.generator.predict(noise)
        print("Displaying generated images")
        plt.figure(figsize=(16, 16))
        gs1 = gridspec.GridSpec(4, 4)
        gs1.update(wspace=0, hspace=0)
        rand_indices = np.random.choice(fake_data_X.shape[0], 16, replace=False)
        for i in range(16):
            ax1 = plt.subplot(gs1[i])
            ax1.set_aspect("equal")
            rand_index = rand_indices[i]
            image = fake_data_X[rand_index, :, :, :]
            fig = plt.imshow(denorm_img(image))
            plt.axis("off")
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)

        plt.tight_layout()
        plt.savefig(save_dir+str(time.time())+"_GENERATEDimage.png", bbox_inches="tight", pad_inches=0)
        plt.show()

if __name__ == "__main__":
    # Shape of noise vector to be input to the Generator
    noise_shape = (1,1,100)
    # Number of steps for training. num_epochs = num_steps*batch_size/data_size
    num_steps = 50000
    # batch size for training.
    batch_size = 128
    # Location to save images and logs 
    img_save_dir = "images/"
    # Image size to reshape to
    image_shape = (32, 32, 3)
    # Location of data directory
    data_dir = "archive/CAT_*/*.jpg"
    # set up log and save directories
    log_dir = "saves/"
    save_model_dir = "saves/"

    gan = GAN(image_shape, noise_shape)
    # gan.print_status = True
    gan.train(data_dir, num_steps, batch_size, (100, 500), save_model_dir, img_save_dir, log_dir)