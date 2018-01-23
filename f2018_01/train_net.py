from keras import optimizers, callbacks, losses
import numpy as np
from link_to_keras_ipi import metrics, losses as losses_ipi
from link_to_keras_ipi.preprocessing.image import ImageDataGenerator as ImageDataGenerator2

from keras_contrib.callbacks.dead_relu_detector import DeadReluDetector




def test(model, data, args):
    """
    Testing a model
    :param model:
    :param data:
    :param args:
    :return:
    """

    x_test, y_test = data
    y_pred = model.predict(x_test, batch_size=100)
    mse = np.mean(np.square(y_test - y_pred))
    print(mse)


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


def train(model, data, args):
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
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  # loss='mse',
                  loss=losses.categorical_crossentropy,
                  metrics=[metrics.jaccard_with_0_labels])  # accuracy is super bad metric with some outputs being [0, 0] (no annotation)
    
    bool_generator = True
    if bool_generator:
        # With data generator for data augmentation on the spot
        batch_size = 32     # default
        n_data = np.shape(x_train[0])[0]  # x is a list (use x[0] to get first image).
        model.fit_generator(generate_data_generator_list(x_train, y_train),
                            epochs=args.epochs,
                            validation_data=[x_test, y_test],
                            callbacks=[checkpoint, tb, drd],
                            steps_per_epoch=n_data // batch_size    # needed with data_generator (unless we do 1 update = 1 epoch)
                            )
    else:
        model.fit(x_train, y_train,
                  epochs=args.epochs,
                  validation_data=[x_test, y_test],
                  callbacks=[checkpoint, tb, drd]
                  )
