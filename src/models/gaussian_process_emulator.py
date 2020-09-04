import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import itertools
import gpflow
from gpflow.kernels import White, Constant, Linear, SquaredExponential, Exponential, ChangePoints
import tensorflow as tf
from gpflow.ci_utils import ci_niter
from tqdm import tqdm
import gpflow.utilities
import pickle

class Brownian(gpflow.kernels.Kernel):
    def __init__(self,active_dims=None):
        super().__init__(active_dims=active_dims)
        self.variance = gpflow.Parameter(1.0, transform=gpflow.utilities.positive())

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        return self.variance * tf.minimum(X, tf.transpose(X2))  # this returns a 2D tensor

    def K_diag(self, X):
        return self.variance * tf.reshape(X, (-1,))  # this returns a 1D tensor

class ExponentialDecay(gpflow.kernels.Kernel):
    def __init__(self, variance=1.0, alpha=1.0, beta=1.0, active_dims=None):
        super().__init__(active_dims=active_dims)
        self.variance = gpflow.Parameter(variance, transform=gpflow.utilities.positive())
        self.alpha    = gpflow.Parameter(alpha, transform=gpflow.utilities.positive())
        self.beta     = gpflow.Parameter(beta, transform=gpflow.utilities.positive())

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        return self.variance * tf.pow(self.beta/(tf.nn.relu(X) + tf.nn.relu(tf.transpose(X2)) + self.beta),self.alpha)  # this returns a 2D tensor
        #return tf.multiply(self.variance,tf.pow(tf.divide(self.beta,tf.add(tf.add(X,tf.transpose(X2)),self.beta),self.alpha)))  # this returns a 2D tensor

    def K_diag(self, X):
        #return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))
        #return tf.reduce_sum(self.variance * (self.beta/(X + tf.transpose(X) + self.beta))**self.alpha, axis=-1)
        return tf.reshape(self.variance * tf.pow(self.beta/(2*tf.nn.relu(X) + self.beta),self.alpha), (-1,))  # this returns a 1D tensor
        #return self.variance * tf.reshape(X, (-1,))  # this returns a 1D tensor

class GaussianProcessEmulator:
    """
    A class for Gaussian Process Regression

    """

    def __init__(self):
        self.model = None
        self.kernel = None

    def initialize(self,X,y,method,test_size=0.2,maxiter=1000,learning_rate=1e-3,scale_X=True,random_state=42,multiple_kernel_dims=False):
        self.method = method
        self.X = X
        self.y = y
        self.N       = len(y)
        self.num_train_data = int((1-test_size)*self.N)
        self.learning_rate = learning_rate
        self.num_features = X.shape[1]
        self.num_induce = 20*self.num_features

        if method[:4] == "user":
            kernels = eval(method[5:])
        else:
            kernels = Constant() + Constant()
            if method == "linear":
                k = Linear
            elif method == "non-linear":
                k = SquaredExponential
            if multiple_kernel_dims:
                for i in range(self.num_features):
                    kernels *= k(active_dims=[i])
            else:
                kernels *= k()

        self.kernel = kernels
        self.logf = []

        # training/test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)

        # pre-processing (normalizing /wrt training data set)
        self.scaler_X = StandardScaler(with_mean=scale_X,with_std=scale_X).fit(self.X_train)
        self.X_train = self.scaler_X.transform(self.X_train)
        self.X_test  = self.scaler_X.transform(self.X_test)
        self.X       = self.scaler_X.transform(self.X)
        self.scaler_y = StandardScaler().fit(self.y_train)
        self.y_train = self.scaler_y.transform(self.y_train)
        self.y_test  = self.scaler_y.transform(self.y_test)
        self.y       = self.scaler_y.transform(self.y)

        # construct SVGP model
        Z = self.X_train[:self.num_induce, :].copy()
        self.model = gpflow.models.SVGP(kernels, gpflow.likelihoods.Gaussian(), inducing_variable=Z, num_data=self.num_train_data)
        self.summary()

        self.batch_size            = 128
        self.prefetch_size         = tf.data.experimental.AUTOTUNE
        self.shuffle_buffer_size   = self.num_train_data // 2
        self.num_batches_per_epoch = self.num_train_data // self.batch_size
        self.maxiter               = maxiter

        # generate training set (tensorflow-style)
        self.train_dataset = tf.data.Dataset.from_tensor_slices(
            (self.X_train, self.y_train)).repeat().shuffle(buffer_size=self.shuffle_buffer_size)
        print(self.N,self.num_train_data,self.batch_size,self.shuffle_buffer_size,self.prefetch_size)


    def training(self):
        """
        Utility function running the Adam optimizer

        """
        # Create an Adam Optimizer action
        train_iter = iter(self.train_dataset.batch(self.batch_size))
        training_loss = self.model.training_loss_closure(train_iter, compile=True)
        optimizer = tf.optimizers.Adam(self.learning_rate)
        best_model = self.model
        best_elbo = 1.e40

        @tf.function
        def optimization_step():
            optimizer.minimize(training_loss, self.model.trainable_variables)

        for step in tqdm(range(self.maxiter)):
            optimization_step()
            elbo = training_loss().numpy()
            self.logf.append(elbo)
            if elbo < best_elbo:
                best_elbo = elbo
                best_model = self.model
            if step % 1000 == 0:
                print(elbo, best_elbo)

        self.model = best_model

    def save(self,save_dir):
        frozen_model = gpflow.utilities.freeze(self.model)
        module_to_save = tf.Module(name="GaussianProcessEmulatorModel")
        predict_fn = tf.function(
            frozen_model.predict_y, input_signature=[
                tf.TensorSpec(shape=[None, self.num_features], dtype=tf.float64)]
        )
        module_to_save.predict_y = predict_fn
        tf.saved_model.save(module_to_save, save_dir)
        d = {"scaler_X": self.scaler_X, "scaler_y": self.scaler_y}
        with open(save_dir+"scaler.pkl", "wb") as f:
            pickle.dump(d,f)

    def load(self,save_dir):
        self.model = tf.saved_model.load(save_dir)
        with open(save_dir+"scaler.pkl", "rb") as f:
            d = pickle.load(f)
        self.scaler_X = d["scaler_X"]
        self.scaler_y = d["scaler_y"]

    def predict(self,X):
        # scale input
        X = self.scaler_X.transform(X)
        # make prediction
        y_pred, y_pred_var = self.model.predict_y(X)
        # inverse scaling transformation
        y_pred = self.scaler_y.inverse_transform(y_pred)
        y_pred_var *= self.scaler_y.var_
        return y_pred, y_pred_var.numpy()

    def summary(self):
        gpflow.utilities.print_summary(self.model)


def main():

    X = np.arange(0,8.*np.pi,0.5)
    X += 0.1*np.random.random(size=len(X))
    y = 2*np.sin(X) + 2.
    y += np.random.random(size=len(y))
    y = 0.5*X + 2.
    y += 2*np.random.random(size=len(y))
    X = np.reshape(X,(-1,1))
    y = np.reshape(y,(-1,1))
    plt.plot(X,y,'k.')
    print(X.shape,y.shape)
    do_train = False
    gpe = GaussianProcessEmulator()
    if do_train:
        gpe.initialize(X,y,method="non-linear",maxiter=1000)
        gpe.training()
        #gpe.summary()
        gpe.save("./models/")
    gpe.load("./models/")
    X = np.arange(-2*np.pi,10*np.pi,0.5)
    X = np.reshape(X,(-1,1))
    y_pred, y_pred_var = gpe.predict(X)
    l, = plt.plot(X.flatten(),y_pred.flatten())
    plt.fill_between(X.flatten(),y_pred.flatten() - 2*y_pred_var.flatten()**0.5,y_pred.flatten() + 2*y_pred_var.flatten()**0.5,color=l.get_color(),alpha=0.25,lw=0)
    plt.show()

if __name__ == "__main__":
    main()

