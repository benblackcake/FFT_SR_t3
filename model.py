

import tensorflow as tf
from utils import fft,ifft
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
        self.images = tf.placeholder(tf.complex128, [256, 256], name='input_img')
        self.label = tf.placeholder(tf.complex128, [256, 256], name='HR_img')

        # self.image_matrix = tf.reshape(self.images, shape=[-1, 256, 256, 1])
        # self.source_fft = tf.fft2d(tf.complex(self.images, 0.0 * self.images))
        # self.label_fft = tf.fft2d(tf.complex(self.label, 0.0 * self.label))

        # self.source_fft = tf.signal.rfft2d(self.images,fft_length=[tf.shape(self.images)[0],tf.shape(self.images)[1]])
        # self.label_fft = tf.signal.rfft2d(self.label)

        # self.label_fft = tf.fft2d(tf.complex(self.label, 0.0 * self.label))
        self.label_risidual = self.label - self.images
        self.pred = self.model()

        # self.label_risidual_fft = tf.complex(self.label_risidual, 0.0 * self.label_risidual) #self.label - self.images

        self.pred_risidual = self.label_risidual - self.pred

        r = tf.real(self.pred_risidual)
        i = tf.imag(self.pred_risidual)

        self.concat_r_i = tf.concat([r,i],axis=0)
        print(self.concat_r_i)
        self.pred_risidual = tf.real(tf.ifft2d(self.pred_risidual))
        # self.pred = tf.real(tf.ifft2d(self.pred))
        # print(self.pred_risidual.eval(session=self.sess))
        # r = tf.real(self.pred_risidual)
        # i = tf.imag(self.pred_risidual)
        #
        # tf.cast(tf.complex(r, i), tf.complex128)
        # a = ifft(self.pred_risidual.eval(session=self.sess))
        # pred_risidual = tf.spectral.irfft2d(tf.dtypes.cast(self.pred_risidual,tf.complex64))

        print('pred_risidual',(self.pred_risidual))
        self.loss = tf.nn.l2_loss(self.pred_risidual)
        # self.loss = tf.nn.l2_loss(ifft(self.sess.run(self.pred_risidual)))



    def model(self):
        # x = None
        # print('source_fft',source_fft)
        self.f1, self.spectral_c1 = self.fft_conv_pure(self.images,filters=5,width=256,height=256,stride=1, name='conv1')
        self.f2, self.spectral_c2 = self.fft_conv_pure(self.f1,filters=5,width=256,height=256,stride=1, name='conv1')


        return self.f1+self.f2
    #

    def fft_conv_pure(self, source, filters, width, height, stride, activation='relu', name='fft_conv'):
        # This function applies the convolutional filter, which is stored in the spectral domain, as a element-wise
        # multiplication between the filter and the image (which has been transformed to the spectral domain)

        source = tf.reshape(source,shape=[-1,256,256,1])

        weight_fft = self.random_spatial_to_spectral(1, filters, height, width)
        smooth_fft = self.random_spatial_to_spectral(filters, filters, filters, filters)

        w = tf.expand_dims(weight_fft, 0,)
        b = tf.Variable(tf.constant(0.1, shape=[filters],dtype=tf.float64))

        source = tf.transpose(source, [0, 3, 1, 2])  # batch, channel, height, width (1,1,256,256)

        source_fft = tf.expand_dims(source, 2)
        source_fft = tf.tile(source_fft, [1, 1, filters, 1, 1])

        weight_product = source_fft * w

        c_r = tf.real(weight_product)
        c_i = tf.imag(weight_product)

        c_r = tf.reduce_sum(c_r, reduction_indices=1)  # batch, filters, height, width (1,5,256,256)
        c_i = tf.reduce_sum(c_i, reduction_indices=1)  # batch, filters, height, width (1,5,256,256)

        c_r = tf.transpose(c_r, [0, 2, 3, 1]) # (1,256,256,5)
        c_i = tf.transpose(c_i, [0, 2, 3, 1])

        # print(smooth_fft)
        smooth_fft_r = np.real(smooth_fft)
        smooth_fft_i = np.imag(smooth_fft)

        c_r = tf.nn.conv2d(c_r, smooth_fft_r, strides=[1, stride, stride, 1], padding='SAME',name='conv_real_part')
        c_i = tf.nn.conv2d(c_i, smooth_fft_i, strides=[1, stride, stride, 1], padding='SAME',name='conv_imag_part')

        c_r = tf.nn.bias_add(c_r, b)
        c_i = tf.nn.bias_add(c_i, b)

        c_r = tf.reduce_sum(c_r, reduction_indices=3)
        c_i = tf.reduce_sum(c_i, reduction_indices=3)

        c_r = tf.nn.relu(c_r)
        c_i = tf.nn.relu(c_i)


        feature_map = tf.cast(tf.complex(c_r, c_i), tf.complex128, name = 'feature_map')

        w = tf.squeeze(w, [0])  # channels, filters, height, width
        w = tf.reduce_sum(w, reduction_indices=1)
        print(source_fft)
        print(w)
        print(weight_product)

        print(c_r)
        print(c_i)

        print(feature_map)

        return feature_map,w


    def random_spatial_to_spectral(self, batch_size, height, width, filters):
        # Create a truncated random image, then compute the FFT of that image and return it's values
        # used to initialize spectrally parameterized filters
        # an alternative to this is to initialize directly in the spectral domain
        w = tf.truncated_normal([batch_size,height,width,filters], mean=0, stddev=0.01)
        fft_ = fft(self.sess.run(w))
        return fft_

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
            # source = self.sess.run([self.model()],feed_dict={self.images: lr_img, self.label:hr_img})
            # imshow_spectrum(np.squeeze(source))

            # _residual = self.sess.run([self.label_risidual],feed_dict={self.images: lr_img, self.label:hr_img})
            # _r = ifft(_residual)
            # # print(np.abs(_r))
            # plt_imshow(np.squeeze(_r))

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
        imshow_spectrum(np.squeeze(result))
        result = ifft(result)
        # result = result*255/(1e3*1e-5)
        # result = np.clip(result, 0.0, 255.0).astype(np.uint8)
        plt_imshow(np.squeeze(result))
        print(np.abs(result))