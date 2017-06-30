import numpy as np
import matplotlib.pyplot as plt

#own libraries
import copy

np.random.seed(seed=0)


class Data():
    # TODO make im_out optional
    def __init__(self, im_in, im_out, bool_tophat = True, bool_residue = False, colors_sep = True, big_patches = 32, ext = 0):
        """ Bool residue: set to true if you want to train the residue instead of desired image
        colors_sep: separate rgb
        gausshat:
        """
        
        # TODO, extension
        
        self.im_in = im_in
        self.im_out = im_out
        
        if bool_tophat:
            import data_net_sr
            self.im_in_gausshat = data_net_sr.tophatblur(self.im_in)
        
        self.bool_residue = bool_residue
        self.colors_sep = colors_sep
        self.big_patches = big_patches
        if self.big_patches:
            self.width = big_patches
        
        # Train on residue
        if self.bool_residue == True:
            self.im_residue = im_out - im_in
        
        self.shape = np.shape(im_in)
        
        # settings
        self.ext = ext
        self._bool_tophat = bool_tophat
        
        # generate the patches
        self.images2patches()
                    
    def patches2images(self, out_patches, normalize = True):
        
        if self.colors_sep:
            shape = np.shape(out_patches)
            out_patches = np.reshape(out_patches, (-1, 3, shape[1], shape[2]))
            shape = np.shape(out_patches)
            out_patches_star = np.zeros((shape[0], shape[2], shape[3], shape[1]))
            for color_i in range(3):
                out_patches_star[..., color_i] = out_patches[:, color_i , :, :]
            out_patches = out_patches_star
        
        if self.big_patches:
            shape = list(self.shape)
            shape_patches =  np.shape(out_patches)
            width = shape_patches[-2]
            shape[-1] = shape_patches[-1]

            out_patches_star = np.zeros(shape=shape)
            h_range = shape[0]//width
            w_range = shape[1]//width
            
            for h_i in range(h_range):
                for w_i in range(w_range):
                    foo = out_patches[h_i*w_range + w_i, ...]
                    out_patches_star[width*h_i: width*(h_i + 1), width*w_i: width*(w_i + 1), :] = foo[:, :, :]
                    # self.patches_input.append(im_in[h_i: h_i + width, w_i: w_i + width])

            im_gen = out_patches_star

        else:
            im_gen = np.reshape(out_patches, newshape=self.shape)

        if self.bool_residue == True:   # add residue to input
            im_gen = self.im_in + im_gen

        if normalize:
            return self._normalization(im_gen)
        else:
            return im_gen
    
    def right_patches2images(self, out_patches, normalize = True):
    
        if self.colors_sep:
            shape = np.shape(out_patches)
            out_patches = np.reshape(out_patches, (-1, 3, shape[1], shape[2]))
            shape = np.shape(out_patches)
            out_patches_star = np.zeros((shape[0], shape[2], shape[3], shape[1]))
            for color_i in range(3):
                out_patches_star[..., color_i] = out_patches[:, color_i, :, :]
            out_patches = out_patches_star
            
        if normalize:
            return self._normalization(out_patches)
        else:
            return out_patches
    
    def bot_patches2images(self, out_patches, normalize = True):
        
        if self.colors_sep:
            out_patches = self._color_sep(out_patches)
    
        if normalize:
            return self._normalization(out_patches)
        else:
            return out_patches
    
    def botright_patch2image(self, out_patch, normalize = True):
        if self.colors_sep:
            out_patch = self._color_sep(out_patch)
            
        if normalize:
            return self._normalization(out_patch)
        else:
            return out_patch
    
    def _color_sep(self, out_patches):
        shape = np.shape(out_patches)
        out_patches = np.reshape(out_patches, (-1, 3, shape[1], shape[2]))
        shape = np.shape(out_patches)
        out_patches_star = np.zeros((shape[0], shape[2], shape[3], shape[1]))
        for color_i in range(3):
            out_patches_star[..., color_i] = out_patches[:, color_i, :, :]
        return out_patches_star
    
    def _normalization(self, im):
        # Removes some outliers
        im[im > 1.0] = 1.0
        im[im < 0.0] = 0.0
        return im

    def in_patches(self):
        return self.patches_input
    
    def in_patches_gausshat(self):
        return self.patches_input_gausshat
    
    def out_patches(self):
        return self.patches_output
    
    # with the rightside not contained in the patches, they are generated independently
    def right_patches(self):
        a = DataPlaceholder()
        a.x = self.patches_input_right
        if self._bool_tophat:
            a.x_tophat = self.patches_input_right_gausshat
        return a
    
    def bot_patches(self):
        # return self.patches_input_bot
        a = DataPlaceholder()
        a.x = self.patches_input_bot
        if self._bool_tophat:
            a.x_tophat = self.patches_input_bot_gausshat
        return a
    
    def botright_patch(self):
        # return self.patch_input_botright
        a = DataPlaceholder()
        a.x = self.patch_input_botright
        if self._bool_tophat:
            a.x_tophat = self.patch_input_botright_gausshat
        return a
  
    def images2patches(self):
        
        im_in = self.im_in

        im_in_seg = SegmentedImage(im_in, ext = self.ext)
        if self._bool_tophat:
            im_in_gausshat_seg = SegmentedImage(self.im_in_gausshat, ext = self.ext)
                   
        if self.bool_residue == True:
            im_out = self.im_residue
        else:
            im_out = self.im_out
            
        im_out_seg = SegmentedImage(im_out, ext = 0)

        shape = np.shape(im_in)
        
        if self.big_patches:

            self.patches_input = []
            self.patches_input_gausshat = []
            self.patches_output = []
            self.patches_input_right = []
            self.patches_input_right_gausshat = []
            self.patches_input_bot = []
            self.patches_input_bot_gausshat = []
            
            width = self.width
            
            for h_i in range(shape[0]//width ):
                for w_i in range(shape[1]//width ):
                    patch_in = im_in_seg.get_patch(width * h_i, width * w_i, width)
                    self.patches_input.append(patch_in)
                    if self._bool_tophat:
                        patch_in = im_in_gausshat_seg.get_patch(width * h_i, width * w_i, width)
                        self.patches_input_gausshat.append(patch_in)
                    patch_i = im_out_seg.get_patch(width * h_i, width * w_i, width)
                    self.patches_output.append(patch_i)
                    
            for h_i in range(shape[0]//width):
                patch_in = im_in_seg.get_patch(width * h_i, shape[1]-width, width)
                self.patches_input_right.append(patch_in)
                if self._bool_tophat:
                    patch_in = im_in_gausshat_seg.get_patch(width * h_i, shape[1]-width, width)
                    self.patches_input_right_gausshat.append(patch_in)

            for w_i in range(shape[1] // width):
                patch_in = im_in_seg.get_patch(shape[0] - width, width * w_i, width)
                self.patches_input_bot.append(patch_in)
                if self._bool_tophat:
                    patch_in = im_in_gausshat_seg.get_patch(shape[0] - width, width * w_i, width)
                    self.patches_input_bot_gausshat.append(patch_in)

            patch_in = im_in_seg.get_patch(shape[0] - width, shape[1] - width, width)
            self.patch_input_botright = np.reshape(patch_in, newshape=(1, width + 2*self.ext,
                                                                       width + 2*self.ext, shape[2]))
            if self._bool_tophat:
                patch_in = im_in_gausshat_seg.get_patch(shape[0] - width, shape[1] - width, width)
                self.patch_input_botright_gausshat = np.reshape(patch_in, newshape=(1, width + 2 * self.ext,
                                                                           width + 2 * self.ext, shape[2]))

            self.patches_input = np.asarray(self.patches_input)
            self.patches_input_right = np.asarray(self.patches_input_right)
            self.patches_input_bot = np.asarray(self.patches_input_bot)
            if self._bool_tophat:
                self.patches_input_gausshat = np.asarray(self.patches_input_gausshat)
                self.patches_input_right_gausshat = np.asarray(self.patches_input_right_gausshat)
                self.patches_input_bot_gausshat = np.asarray(self.patches_input_bot_gausshat)
            
            self.patches_output = np.asarray(self.patches_output)


        else:
            res = 'highres'
            if res == 'lowres':
                (self.patches_input, self.patches_output) = generate_patches(im_in, im_out, width = self.width )
            elif res == 'highres':
                (self.patches_input, self.patches_output) = generate_patches_hres(im_in, im_out, width = self.width )
            
        # TODO can be done better
        def color_sep(patches):
            shape_patches = np.shape(patches)

            patches_star = np.empty((shape_patches[0], shape_patches[3], shape_patches[1], shape_patches[2], 1))

            for color_i in range(shape_patches[-1]):
                patches_star[:, color_i, :, :, :] = patches[..., color_i:color_i + 1]
                
            return np.reshape(patches_star, newshape=(-1, shape_patches[1], shape_patches[2], 1))
            
        if self.colors_sep:
            self.patches_input = color_sep(self.patches_input)
            self.patches_output = color_sep(self.patches_output)
            self.patches_input_right = color_sep(self.patches_input_right)
            self.patches_input_bot = color_sep(self.patches_input_bot)
            self.patch_input_botright = color_sep(self.patch_input_botright)
            if self._bool_tophat:
                self.patches_input_gausshat = color_sep(self.patches_input_gausshat)
                self.patches_input_right_gausshat = color_sep(self.patches_input_right_gausshat)
                self.patches_input_bot_gausshat = color_sep(self.patches_input_bot_gausshat)
                self.patch_input_botright_gausshat = color_sep(self.patch_input_botright_gausshat)

                # shape_in = np.shape(self.patches_input)
            # shape_out = np.shape(patches_output)
            # shape_in_right = np.shape(self.patches_input_right)
            #
            # a = np.empty((shape_in[0], shape_in[3], shape_in[1], shape_in[2], 1))
            # b = np.empty((shape_out[0], shape_out[3], shape_out[1], shape_out[2], 1))
            # a_right = np.empty((shape_in_right[0], shape_in_right[3], shape_in_right[1], shape_in_right[2], 1))
            #
            # for color_i in range(3):
            #     a[:, color_i, :, :, :] = self.patches_input[..., color_i:color_i+1]
            #     b[:, color_i, :, :, :] = patches_output[..., color_i:color_i+1]
            #
            #
            # self.patches_input = np.reshape(a, newshape=(-1, shape_in[1], shape_in[2], 1))
            # patches_output = np.reshape(b, newshape=(-1, shape_out[1], shape_out[2], 1))
        
        # print("input patch: \n{}".format(self.patches_input[-1]))
        # print("output patch: \n{}".format(patches_output[-1]))

        # return shuffle_patches(self.patches_input, self.patches_output)
    
    # Reference image that we want to outperform
    def ref_im(self):
        return self.im_in   # Bicubic at this point****
    
    def goal_im(self):
        return self.im_out


def print_here(text):
    print("HERE :{}".format(text))

# Shuffle the patches
def shuffle_patches(patches_input, patches_output):
    shape_size = np.shape(patches_input)[0]
    idx_shuffled = np.random.permutation(shape_size)
    patches_input_shuffled = patches_input[idx_shuffled, ...]
    patches_output_shuffled = patches_output[idx_shuffled, ...]
    
    return (patches_input_shuffled, patches_output_shuffled)


def generate_patches_hres(im_input, im_output, width = 5):
    patches_input = []
    patches_output = []
    
    im_in_extend = SegmentedImage(im_input, width)
    im_out_extend = SegmentedImage(im_output, 1)
    
    shape_input = np.shape(im_input)
    
    for h_i in range(shape_input[0]):
        for w_i in range(shape_input[1]):
            in_ij = im_in_extend.get_segm(h_i, w_i)
            patches_input.append(in_ij)
            out_ij = im_out_extend.get_segm(h_i, w_i)
            patches_output.append(out_ij)
    
    patches_input = np.asarray(patches_input)
    patches_output = np.asarray(patches_output)
    
    print("input_images: {}".format(np.shape(patches_input)))
    print("output_images: {}".format(np.shape(patches_output)))
    
    return (patches_input, patches_output)


def generate_patches(im_input, im_output, width = 5):
    patches_input = []
    patches_output = []
    
    image_extended = SegmentedImage(im_input, width)
    
    shape_input = np.shape(im_input)
    
    for h_i in range(shape_input[0]):
        for w_i in range(shape_input[1]):
            foo = image_extended.get_segm(h_i, w_i)
            patches_input.append(foo)
            patches_output.append(im_output[2 * h_i:2 * h_i + 2, 2 * w_i:2 * w_i + 2, ...])
    
    patches_input = np.asarray(patches_input)
    patches_output = np.asarray(patches_output)
    
    print("input_images: {}".format(np.shape(patches_input)))
    print("output_images: {}".format(np.shape(patches_output)))
    
    return (patches_input, patches_output)


def extend_image(image_orig, ext):
    width = 1 + 2*ext
    shape_orig = np.shape(image_orig)
    shape_ext = list(shape_orig)
    shape_ext[0] = shape_orig[0] + 2 * ext
    shape_ext[1] = shape_orig[1] + 2 * ext
    
    if len(shape_orig) == 2:
        image_extended = np.zeros((shape_orig[0] + width - 1, shape_orig[1] + width - 1))
    elif len(shape_orig) == 3:
        image_extended = np.zeros((shape_orig[0] + width - 1, shape_orig[1] + width - 1, shape_orig[2]))
    else:
        raise LookupError
    
    image_extended[ext: shape_orig[0] + ext, ext: shape_orig[1] + ext, ...] = image_orig
       
    if ext != 0:
        image_extended[:ext, :, ...] = image_extended[2 * ext: ext:-1, :, ...]
        image_extended[-ext:, :, ...] = image_extended[-ext - 2: -2 * ext - 2:-1, :, ...]
        image_extended[:, :ext, ...] = image_extended[:, 2 * ext: ext:-1, ...]
        image_extended[:, -ext:, ...] = image_extended[:, -ext - 2: -2 * ext - 2:-1, ...]

    return image_extended

class SegmentedImage():
    def __init__(self, image_orig, ext = 0):
        self.image_extended = extend_image(image_orig, ext)
        self.ext = ext
    
    def get_segm(self, h_i, w_i):
        return self.image_extended[h_i: h_i + self.width, w_i: w_i + self.width, ...]

    def get_patch(self, h_i, w_i, orig_width):
        return self.image_extended[h_i: h_i + orig_width + 2 * self.ext, w_i: w_i + orig_width + 2 *self.ext, ...]

class DataGen():
    """
    Class that contains all training data
    """
    # TODO contains , input patches, output patches (general container of this), image extended etc
    def __init__(self, images = None, images_that = None, labels =  None):
        self._images = images
        self._images_that = images_that
        self._labels = labels
               
        self._num_examples = images.shape[0]
        self._epochs_completed = 0
        
        self.shuffle()
        
        test_batch_size = 100
        
        self.test_images = self._images[:test_batch_size]
        if self._images_that:
            # self.test_images_that = np.empty(shape=np.shape(self._labels[:test_batch_size]))
            self.test_images_that = self._images_that[:test_batch_size]
        self.test_labels = self._labels[:test_batch_size]
        
        self.data_placeholder = DataPlaceholder()
        
    def shuffle(self):
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._images = self._images[perm]
        if self._images_that:
            self._images_that = self._images_that[perm]
        self._labels = self._labels[perm]

        self.batch_zero()
        
    def get_test_data(self):
        self.data_placeholder.x = self.test_images
        if self._images_that:
            self.data_placeholder.x_tophat = self.test_images_that
        self.data_placeholder.y = self.test_labels
        
        return copy.copy(self.data_placeholder)
    
    def next_batch(self, batch_size):
        """ Return the next `batch_size` examples from this data set."""
        
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            self.shuffle()

            # Start next epoch
            start = self._index_in_epoch
            self._index_in_epoch += batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        
        self.data_placeholder.x = self._images[start:end]
        if self._images_that:
            self.data_placeholder.x_tophat = self._images_that[start:end]
        self.data_placeholder.y = self._labels[start:end]
        
        return self.data_placeholder

    def batch_zero(self):
        """ Reset batches to first one """
        self._index_in_epoch = 0
    
    @property
    def images(self):
        return self._images
    
    @property
    def labels(self):
        return self._labels
    
    @property
    def num_examples(self):
        return self._num_examples
    
class DataPlaceholder():
    x = None
    x_tophat = None
    y = None
