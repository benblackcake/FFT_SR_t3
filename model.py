

import tensorflow as tf
from utils import fft,L2_loss
import numpy as np
from utils import imshow,imshow_spectrum,plt_imshow

class FFTSR:

    def __init__(self, sess, learning_rate, epoch):
        self.sess =sess
        self.epoch = epoch
        # self.images = images
        self.learning_rate = learning_rate
        self.build_model()


    def build_model(self):
        self.images = tf.placeholder(tf.float32, [256, 256], name='input_img')
        self.label = tf.placeholder(tf.float32, [256, 256], name='HR_img')

        # self.image_matrix = tf.reshape(self.images, shape=[-1, 256, 256, 1])
        self.source_fft = tf.fft2d(tf.complex(self.images, 0.0 * self.images))
        self.label_fft = tf.fft2d(tf.complex(self.label, 0.0 * self.label))

        # self.source_fft = tf.signal.rfft2d(self.images,fft_length=[tf.shape(self.images)[0],tf.shape(self.images)[1]])
        # self.label_fft = tf.signal.rfft2d(self.label)

        # self.label_fft = tf.fft2d(tf.complex(self.label, 0.0 * self.label))
        self.label_risidual = self.label_fft - self.source_fft
        self.pred = (self.model())

        # self.label_risidual_fft = tf.complex(self.label_risidual, 0.0 * self.label_risidual) #self.label - self.images

        self.pred_risidual = self.label_risidual - (self.label_fft- self.pred)
        # self.pred_risidual = tf.real(tf.ifft2d(self.pred_risidual))
        self.pred = tf.real(tf.ifft2d(self.pred))

        # self.pred = tf.squeeze(self.model())

        # self.loss_min = self.label_risidual_fft - tf.ifft2d(self.pred)
        # self.loss_min = tf.real(self.loss_min)

        # self.predict = tf.real(tf.ifft2d(self.pred))
        # self.predict =tf.real(tf.ifft2d(self.pred))

        # print('label_risidual',self.loss_min)
        # print('pred',self.pred)

        # loss_complex = self.label_risidual - self.pred
        self.loss = tf.nn.l2_loss(tf.real(tf.ifft2d(self.pred_risidual)))
        # squared_deltas = tf.square(self.label - self.pred)
        # self.loss = L2_loss(self.label, self.pred)
        # print(self.pred)
        # self.loss = tf.reduce_mean(tf.square(self.label - self.pred))
        # print('build_model_image_shape',self.images)


    def model(self):
        # x = None
        # print('source_fft',source_fft)
        self.f1,self.spectral_c1 = self.fft_conv_pure(self.source_fft,filters=5,width=256,height=256,stride=1, name='conv1')
        # f1_smooth,self.spatial_s1,self.spectral_s1 = self.fft_conv(self.spectral_c1,filters=5,width=5,height=5,stride=1, name='f1_smooth')

        self.f2,self.spectral_c2 = self.fft_conv_pure(self.f1,filters=5,width=256,height=256,stride=1, name='conv2')
        self.f3,self.spectral_c3 = self.fft_conv_pure(self.f2,filters=5,width=256,height=256,stride=1, name='conv3')
        self.f4,self.spectral_c4 = self.fft_conv_pure(self.f3,filters=5,width=256,height=256,stride=1, name='conv4')
        self.f5,self.spectral_c5 = self.fft_conv_pure(self.f4,filters=5,width=256,height=256,stride=1, name='conv5')
        self.f6,self.spectral_c6 = self.fft_conv_pure(self.f5,filters=5,width=256,height=256,stride=1, name='conv6')

        # self.f2,self.spectral_c2 = self.fft_conv_pure(self.f1,filters=5,width=256,height=256,stride=1, name='conv2')
        # self.f2,self.spectral_c2 = self.fft_conv_pure(self.f1,filters=5,width=256,height=256,stride=1, name='conv2')

        # f2_smooth,self.spatial_s2,self.spectral_s2 = self.fft_conv(f2,filters=5,width=5,height=5,stride=1, name='f2_smooth')

        # f1_smooth,_,_ = self.fft_conv(f1,filters=5,width=5,height=5,stride=1,name='f1_smooth')
        print('f1',self.f1)
        f_ = self.f1+self.f2+self.f3+self.f4+self.f5+self.f6
        p_ = f_ *self.f1
        # i_ = p_+self.f1
        # f_=self
        # f_ = tf.real(tf.ifft2d(f_))
        print('f_',f_)
        print('__debug__spatial_c1',self.spectral_c1)

        return p_
    #

    def fft_conv_pure(self, source, filters, width, height, stride, activation='relu', name='fft_conv'):
        # This function applies the convolutional filter, which is stored in the spectral domain, as a element-wise
        # multiplication between the filter and the image (which has been transformed to the spectral domain)
        source = tf.reshape(source,shape=[-1,256,256,1])
        _, input_height, input_width, channels = source.get_shape().as_list()

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            init = self.random_spatial_to_spectral(channels, filters, height, width) # (1,5,256,256)
            init_smooth = self.random_spatial_to_spectral(filters, filters, filters, filters)

            # Option 1: Over-Parameterize fully in the spectral domain
            w_real = tf.Variable(init.real, dtype=tf.float32, name='real')
            w_imag = tf.Variable(init.imag, dtype=tf.float32, name='imag')
            w = tf.cast(tf.complex(w_real, w_imag), tf.complex64)

            w_smooth_real = tf.Variable(init_smooth.real, dtype=tf.float32, name='real')
            w_smooth_imag = tf.Variable(init_smooth.imag, dtype=tf.float32, name='imag')

            b = tf.Variable(tf.constant(0.1, shape=[filters]))

        # Add batch as a dimension for later broadcasting
        w = tf.expand_dims(w, 0)  # batch, channels, filters, height, width
        w_smooth_real = tf.transpose(w_smooth_real, [2, 3, 0, 1])
        w_smooth_imag = tf.transpose(w_smooth_imag, [2, 3, 0, 1]) #(5,5,1,5)

        # Prepare the source tensor for FFT (1,256,256,1)
        source = tf.transpose(source, [0, 3, 1, 2])  # batch, channel, height, width (1,1,256,256)
        # source_fft = tf.fft2d(tf.complex(source, 0.0 * source))

        # Prepare the FFTd input tensor for element-wise multiplication with filter
        source_fft = tf.expand_dims(source, 2)  # batch, channels, filters, height, width
        source_fft = tf.tile(source_fft, [1, 1, filters, 1, 1]) # (1,1,5,256,256)

        # Shift, then pad the filter for element-wise multiplication, then unshift
        w_shifted = self.batch_fftshift2d(w)
        height_pad = (input_height - height) // 2
        width_pad = (input_width - width) // 2

        # Pads with zeros
        w_padded = tf.pad(
            w_shifted,
            [[0, 0], [0, 0], [0, 0], [height_pad, height_pad], [width_pad, width_pad]],
            mode='CONSTANT'
        )
        w_padded = self.batch_ifftshift2d(w_padded)

        # Convolve with the filter in spectral domain
        conv = source_fft * tf.conj(w_padded) # (1,1,5,256,256)
        print(conv)

        # Sum out the channel dimension, and prepare for bias_add (1,5,256,256)
        # Note: The decision to sum out the channel dimension seems intuitive, but
        #	   not necessarily theoretically sound.
        c_r = tf.real(conv)
        c_i = tf.imag(conv)

        c_r = tf.reduce_sum(c_r, reduction_indices=1)  # batch, filters, height, width (1,5,256,256)
        c_i = tf.reduce_sum(c_i, reduction_indices=1)  # batch, filters, height, width (1,5,256,256)

        c_r = tf.transpose(c_r, [0, 2, 3, 1]) # (1,256,256,5)
        c_i = tf.transpose(c_i, [0, 2, 3, 1])
        print('w_smooth_real',w_smooth_real)
        c_r = tf.nn.conv2d(c_r, w_smooth_real, strides=[1, stride, stride, 1], padding='SAME')
        c_i = tf.nn.conv2d(c_i, w_smooth_imag, strides=[1, stride, stride, 1], padding='SAME')

        c_r = tf.nn.bias_add(c_r, b)
        c_i = tf.nn.bias_add(c_i, b)

        c_r = tf.reduce_sum(c_r, reduction_indices=3)
        c_i = tf.reduce_sum(c_i, reduction_indices=3)

        c_r = tf.nn.relu(c_r)
        c_i = tf.nn.relu(c_i)

        feature_map = tf.cast(tf.complex(c_r, c_i), tf.complex64)
        print(feature_map)
        print('feature_map',feature_map)
        print('c_r',c_r)
        print('c_i',c_i)

        w_smooth = tf.cast(tf.complex(w_smooth_real, w_smooth_imag), tf.complex64)
        print(w_smooth)
        # conv = tf.real(tf.ifft2d(conv))
        # conv = tf.reduce_sum(conv, reduction_indices=1)  # batch, filters, height, width
        # conv = tf.transpose(conv, [0, 2, 3, 1])  # batch, height, width, filters
        #
        # # Drop the batch dimension to keep things consistent with the other conv_op functions
        w = tf.squeeze(w, [0])  # channels, filters, height, width
        w = tf.reduce_sum(w, reduction_indices=1)

        #
        # # Compute a spatial encoding of the filter for visualization
        # spatial_filter = tf.ifft2d(w)
        # spatial_filter = tf.transpose(spatial_filter, [2, 3, 0, 1])  # height, width, channels, filters
        #
        # # Add the bias (in the spatial domain)
        # output = tf.nn.bias_add(conv, b)
        # output = tf.nn.relu(output) if activation is 'relu' else output
        # print('out',output)
        # w = tf.reduce_sum(w, reduction_indices=1)
        # print(w)
        return feature_map, w


    def random_spatial_to_spectral(self, batch_size, height, width, filters):
        # Create a truncated random image, then compute the FFT of that image and return it's values
        # used to initialize spectrally parameterized filters
        # an alternative to this is to initialize directly in the spectral domain
        w = tf.truncated_normal([batch_size,height,width,filters], mean=0, stddev=0.01)
        fft_ = tf.fft2d(tf.complex(w, 0.0 * w), name='spectral_initializer')
        return fft_.eval(session=self.sess)

    def batch_fftshift2d(self, tensor):
        # Shifts high frequency elements into the center of the filter
        indexes = len(tensor.get_shape()) - 1
        top, bottom = tf.split(tensor, 2, indexes - 1)
        tensor = tf.concat([bottom, top], indexes - 1)
        left, right = tf.split(tensor, 2, indexes)
        tensor = tf.concat([right, left], indexes)
        print(tensor)
        return tensor

    def batch_ifftshift2d(self, tensor):
        # Shifts high frequency elements into the center of the filter
        indexes = len(tensor.get_shape()) - 1
        left, right = tf.split(tensor, 2, indexes)
        tensor = tf.concat([right, left], indexes)
        top, bottom = tf.split(tensor, 2, indexes - 1)
        tensor = tf.concat([bottom, top], indexes - 1)

        return tensor

    def run(self,hr_img,lr_img):
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())
        print('run: ->',hr_img.shape)
        # shape = np.zeros(hr_img.shape)
        # err_ = []
        # print(shape)
        for er in range(self.epoch):
            # image = tf.reshape(image,[image.shape[0],image.shape[1]])
            _,x = self.sess.run([self.train_op,self.loss],feed_dict={self.images: lr_img, self.label:hr_img})
            source = self.sess.run([self.source_fft],feed_dict={self.images: lr_img, self.label:hr_img})
            imshow_spectrum(np.squeeze(source))

            # _residual = self.sess.run([self.label_risidual],feed_dict={self.images: lr_img, self.label:hr_img})
            # print(np.abs(_residual))
            # plt_imshow(np.squeeze(np.abs(_residual)))

            print(x)
        # w = self.sess.run([self.spectral_c1],feed_dict={self.images: lr_img, self.label:hr_img})
        # w = np.asarray(w)
        # # w =np.squeeze(w)
        # # w = w /(1e3*1e-5)
        # print(w)
        # print('----')
        # print(w[:,:,:,0])
        # # imshow_spectrum(w)
    # #
        result = self.pred.eval({self.images: lr_img,self.label:hr_img})
        result = np.squeeze(result)
        # result = result*255/(1e3*1e-5)
        # result = np.clip(result, 0.0, 255.0).astype(np.uint8)
        plt_imshow(((result)))
        print(np.abs(result))