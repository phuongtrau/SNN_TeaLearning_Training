from keras.datasets import fashion_mnist
import numpy as np 
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
print(x_train.shape)
x_train = x_train.reshape((-1,1))
print(x_train.shape)
bins = np.arange(256)
# print(bins)
# hist = np.histogram(x_train,bins=bins,density=False)
# print(hist)
plt.hist(x_train,bins=bins,density=True)
plt.show()