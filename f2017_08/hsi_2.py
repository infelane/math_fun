import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.neighbors import NearestNeighbors


from f2017_08 import main_multimodal


class NN(object):
    def __init__(self, x, n):
        self.n = n
        self.depth = np.shape(x)[-1]

        # self.random_points(x)
        
    def random_points(self, x):
        n_samples = np.shape(x)[0]
        idx = np.arange(n_samples)
        np.random.seed(123)
        np.random.shuffle(idx)
        
        self.points = x[idx[:self.n], :]
        
        # self.points = np.random.uniform(0, 1, size=(self.n, self.depth))
        
    def get_nn(self, x):
        dist = []
        
        for i in range(self.n):
            dist_i = np.sum(np.square(x - self.points[i:i+1, :]), axis = -1)
            dist.append(dist_i)
            
        dist = np.stack(dist, axis = 0)
        return np.argmin(dist, axis = 0)
    
    def update_points(self, x):
        nn_out = self.get_nn(x)
        
        for i in range(self.n):
            self.points[i, :] = np.mean(x[nn_out == i], axis = 0)

        print(np.mean(self.points, axis=1))
        self.save_points()
        
    def save_points(self):
        np.save('points.npy', self.points)
        
    def load_points(self):
        self.points = np.load('points.npy')
        

def main():
    hsi_data = main_multimodal.HsiData()
    img = hsi_data.get_img()
    
    if 1:
        img_rgb = hsi_data.to_rgb(img)
        plt.figure()
        plt.imshow(img_rgb)
    
    if 0:
        rgb = hsi_data.to_rgb(img)
        plt.imsave('/home/lameeus/data/multimodal/rgb.png', rgb)
        
    if 0:
        # img = main_multimodal.example_crop(img)
        
        flat = main_multimodal.Flat(img)
        img_flat = flat.flatten(img)
        
        # print(np.shape(mask_01))
        # print(np.shape(img))
        
        # img_data = img[mask_01 == 1, :]
        
        img_data = hsi_data.get_data()
        
        n = 7
        nn = NN(img_data, n)
        
        if 0:
            nn.random_points(img_data)
        else:
            nn.load_points()
        
        black = [0., 0., 0.]
        red = [1, 0, 0]
        blue = [0, 0, 1.]
        green = [0, 1., 0.]
        cyan = [0., 1., 1.]
        purple = [1., 0, 1.]
        yellow = [1., 1., 0.]
        colors = [black, red, blue, green, cyan, purple, yellow]
        
        def thingy():
            nn_out = nn.get_nn(img_flat)
    
            nn_out_img = flat.deflatten(nn_out)
            
            shape = np.shape(nn_out_img)
    
            nr0 = np.zeros(shape=(shape[0], shape[1], 3))
            for i in range(n):
            
                nr0[nn_out_img[:, :, 0] == i, :] = colors[i]
            # nr0[nn_out_img[:, :, 0] == 1, :] = 1.
            # nr0[nn_out_img[:, :, 0] == 2, :] = 1.
            
            plt.figure()
            plt.imshow(nr0)
            
        plt.figure()
        for i in range(n):
            plt.plot(nn.points[i, :], color = colors[i], label = '{}'.format(i))
        # plt.show()
        
        thingy()
        for i in range(0):
            
            nn.update_points(img_data)
    
            thingy()
      
    plt.show()
    
    # """ Other example """
    #y
    # n_neighbors = 15
    #
    # # import some data to play with
    # iris = datasets.load_iris()
    #
    # print(np.shape(iris.data))
    #
    # # we only take the first two features. We could avoid this ugly
    # # slicing by using a two-dim dataset
    # X = iris.data[:, :2]
    # y = iris.target
    #
    # h = .02  # step size in the mesh
    #
    # # Create color maps
    # cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    # cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    #
    # for weights in ['uniform', 'distance']:
    #     # we create an instance of Neighbours Classifier and fit the data.
    #     clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    #     clf.fit(X, y)
    #
    #     # Plot the decision boundary. For that, we will assign a color to each
    #     # point in the mesh [x_min, x_max]x[y_min, y_max].
    #     x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    #     y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    #     xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
    #                          np.arange(y_min, y_max, h))
    #     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    #
    #     # Put the result into a color plot
    #     Z = Z.reshape(xx.shape)
    #     plt.figure()
    #     plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    #
    #     # Plot also the training points
    #     plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
    #                 edgecolor='k', s=20)
    #     plt.xlim(xx.min(), xx.max())
    #     plt.ylim(yy.min(), yy.max())
    #     plt.title("3-Class classification (k = %i, weights = '%s')"
    #               % (n_neighbors, weights))
    #
    # plt.show()


if __name__ == '__main__':
    main()