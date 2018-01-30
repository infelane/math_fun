from keras import optimizers, callbacks, losses, backend as K
import numpy as np
from link_to_keras_ipi import metrics, losses as losses_ipi
from link_to_keras_ipi.preprocessing.image import ImageDataGenerator as ImageDataGenerator2
from link_to_soliton.metrics import roc
from keras_contrib.callbacks.dead_relu_detector import DeadReluDetector


def test(model, data, args):
    """
    Testing a model
    :param model:
    :param data:
    :param args:
    :return:
    """
    
    # weights = '/home/lameeus/data/general/weights/unet_complex/weights-'
    weights = '/home/lameeus/data/general/weights/unet_complex_shift/weights-'
    weights += 'best.h5'
    model.load_weights(weights)

    x_val, y_val = data
    y_pred = model.predict(x_val, batch_size=100)
    
    jaccard = np.mean(K.eval(metrics.jaccard_with_0_labels(y_val, y_pred)))
    print('jaccard = {}'.format(jaccard))

    roc.curve2(y_pred, y_val, save_data=True)

    from f2018_01 import builder_net
    
    if 0:
        w = 10          # output is 10x10
        ext_double = 36- w    # input size is 36x36
    else:
        w = 100
        ext_double = 26

        x_builder1 = np.empty(shape=[1, w+ext_double, w+ext_double, 3])
        x_builder3 = np.empty(shape=[1, w+ext_double, w+ext_double, 1])
        x_builder = [x_builder1]*2 + [x_builder3]
        y_builder = np.empty(shape=[1, w, w, 2])
    model = builder_net.complex_unet_shift(x=x_builder, y=y_builder)
    model.load_weights(weights)
    segmentation_whole_image(model, w, ext_double)
    
    
def segmentation_whole_image(model, w, ext_double):
    # TODO clean up, a lot is used from other file
    import matplotlib.pyplot as plt
    from f2017_09 import main_lamb
    from f2017_08.hsi import tools_data
    
    dict_data = main_lamb.MainData(set='zach_big', w=w)

    img_clean = dict_data.get_img_clean()
    img_rgb = dict_data.get_img_rgb()
    img_ir = dict_data.get_img_ir()
    
    data = tools_data.Data(img_clean, w=w)
    
    ext_zoom = (ext_double) // 2
    ext_tuple = (ext_zoom, ext_double-ext_zoom)
    
    x_clean = data.img_to_x2(img_clean, ext=ext_tuple)
    x_rgb = data.img_to_x2(img_rgb, ext=ext_tuple)
    x_ir = data.img_to_x2(img_ir, ext=ext_tuple)
    
    x_in = [x_clean, x_rgb, x_ir]

    y_pred = model.predict(x_in)

    pred_imgs = data.y_to_img2(y_pred)
    
    def get_pred_rgb(img_clean, pred_img):
        cyan = [0, 1, 1]
        pred_rgb = np.copy(img_clean)
        pred_rgb[pred_img[:, :, 1] > 0.5, :] = cyan
        return pred_rgb
    
    segmentation_rgb = get_pred_rgb(img_clean, pred_imgs)

    plt.imsave('input.png', img_clean)
    plt.imsave('prediction.png', segmentation_rgb)
    
    from link_to_soliton.paint_tools.image_tools import save_im
    grey_single = pred_imgs[..., 1]
    # grey = np.stack([grey_single]*3, axis=2)
    # save_im(grey, 'pred_certainty.png')
    save_im(grey_single, 'pred_certainty.png')
    
    plt.figure()
    plt.imshow(segmentation_rgb)
    plt.show()


def generate_data_generator_list(x, y, batch_size=32):
    # own version
    datagen = ImageDataGenerator2(horizontal_flip=True,
                                  vertical_flip=True,
                                  diagonal_flip=True,
                                  )
    
    seed = 1337
    assert type(x) == list
    
    n_in = len(x)
    
    gens_in = []
    
    for i in range(n_in):
        gens_in.append(datagen.flow(x[i], seed=seed, batch_size=batch_size))
    
    gens_out = datagen.flow(y, seed=seed, batch_size=batch_size)
    
    while True:
        xi = [a.next() for a in gens_in]
        yi = gens_out.next()
        
        yield xi, yi


def train(model, data, args, bool_generator=True):
    """
    Training a Network
    :param model: the model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data

    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_loss',
                                           save_best_only=True, save_weights_only=True, verbose=1)

    tb = callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False,
                                write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                embeddings_metadata=None)
    
    x_train_part = [x_train_i[:1000, ...] for x_train_i in x_train]
    drd = DeadReluDetector(x_train_part)

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr, decay=args.decay),
                  # loss='mse',
                  loss=losses.categorical_crossentropy,
                  metrics=[metrics.jaccard_with_0_labels])  # accuracy is super bad metric with some outputs being [0, 0] (no annotation)
    
    if bool_generator:
        # With data generator for data augmentation on the spot
        batch_size = 32     # default
        n_train = np.shape(x_train[0])[0]  # x is a list (use x[0] to get first image).
        n_test = np.shape(x_test[0])[0]  # x is a list (use x[0] to get first image).
        model.fit_generator(generate_data_generator_list(x_train, y_train),
                            epochs=args.epochs,
                            validation_data=generate_data_generator_list(x_test, y_test), # [x_test, y_test],
                            callbacks=[checkpoint, tb, drd],
                            steps_per_epoch=n_train // batch_size,    # needed with data_generator (unless we do 1 update = 1 epoch)
                            validation_steps=n_test // batch_size,
                            )
    else:
        model.fit(x_train, y_train,
                  epochs=args.epochs,
                  validation_data=[x_test, y_test],
                  callbacks=[checkpoint, tb, drd]
                  )
