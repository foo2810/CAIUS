import sys
sys.path.append('./')

# tensorflow messageの抑制
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import pandas as pd

from utils.datasets import load_data, train_test_split
from utils.train import training_mixup
from utils.common import time_counter

from models.simple import SimpleCNN
from models.wrapper import VGG16, ResNet50

tfk = tf.keras
tfk.backend.set_floatx('float32')

# Params
n_classes = 3
batch_size = 4
n_epochs = 10
lr = 0.001

# Dataset
print('[Dataset]')
with time_counter():
    # x, y, _ = load_data('data/dataset/', classes=['normal', 'nude', 'swimwear'], size=128, cache_path='data/data_cache/dataset_size128_autopad.pkl', auto_pad_val=True)
    x, y = np.random.randn(60, 128, 128, 3), np.random.randint(0, 3, 60)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_rate=0.5)
    n_train = len(x_train)
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(n_train).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
print('x_train: {}'.format(x_train.shape))
print('y_train: {}'.format(y_train.shape))
print('x_test: {}'.format(x_test.shape))
print('y_test: {}'.format(y_test.shape))

in_shape = x_train.shape[1:]
del x_train, y_train, x_test, y_test


import time
import pathlib
result_path_raw : str = "results/{}".format(int(time.time()))
result_path : pathlib.Path = pathlib.Path(result_path_raw)
# if not result_path.exists():
#     result_path.mkdir(parents=True)

def training_SeLa(encoder_model, train_ds, test_ds, loss, optimizer, n_epochs, batch_size, n_classes, output_best_weights=True, weight_name=None, train_weights=None, **kwargs):
    """
    https://github.com/yukimasano/self-label を参考に pytorch -> tensorflow2

    encoder_model: 

    **kwargs
        encoder_epochs:
    """

    encoder_epochs = kwargs.get("encoder_epochs", 10)

    def training_SeLa_Encoder(encoder_model=None, train_ds=None, n_epochs=20, n_classes=3):
        """
        Encoder 教師なしの部分
        """

        if train_ds is None:
            raise ValueError("train_ds most not be None")

        if encoder_model is None:
            # encoder_model = ResNet50(weights=None, include_top=False)
            raise ValueError("encoder_model most not be None")

            
        def optimize_labels(L, model, pseudo_train_ds, outs, hc, K):
            
            class MovingAverage():
                def __init__(self, intertia=0.9):
                    self.intertia = intertia
                    self.reset()

                def reset(self):
                    self.avg = 0.

                def update(self, val):
                    self.avg = self.intertia * self.avg + (1 - self.intertia) * val

            from scipy.special import logsumexp
            def py_softmax(x, axis=None):
                """stable softmax"""
                return np.exp(x - logsumexp(x, axis=axis, keepdims=True))

            def optimize_L_sk(L, PS, outs, dtype=np.float64, nh=0):
                N = tf.reduce_max(L.shape).numpy()
                tt = time.time()
                PS = PS.T # now it is K x N
                r = np.ones((outs[nh], 1), dtype=dtype) / outs[nh]
                c = np.ones((N, 1), dtype=dtype) / N
                lamb = 25 # default value
                PS **= lamb  # K x N
                inv_K = dtype(1./outs[nh])
                inv_N = dtype(1./N)
                err = 1e6
                _counter = 0
                while err > 1e-1:
                    r = inv_K / (PS @ c)          # (KxN)@(N,1) = K x 1
                    c_new = inv_N / (r.T @ PS).T  # ((1,K)@(KxN)).t() = N x 1
                    if _counter % 10 == 0:
                        err = np.nansum(np.abs(c / c_new - 1))
                    c = c_new
                    _counter += 1
                print("error: ", err, 'step ', _counter, flush=True)  # " nonneg: ", sum(I), flush=True)
                # inplace calculations.
                PS *= np.squeeze(c)
                PS = PS.T
                PS *= np.squeeze(r)
                PS = PS.T
                argmaxes = np.nanargmax(PS, 0) # size N
                L_numpy = L.numpy()
                L_numpy[nh] = argmaxes
                L = tf.convert_to_tensor(L_numpy)
                # newL = tf.convert_to_tensor(argmaxes)
                # # tf.print("\t", newL.shape)
                # # tf.print("\t", L[nh].shape)
                # # tf.print("\t", L.shape)
                # L[nh] = newL
                print('opt took {0:.2f}min, {1:4d}iters'.format(((time.time() - tt) / 60.), _counter), flush=True)

                return L

            def cpu_sk(model, pseudo_train_ds, L, outs, hc=1, K=3, dtype=np.float64):
                """ Sinkhorn Knopp optimization on CPU
                    * stores activations to RAM
                    * does matrix-vector multiplies on CPU
                    * slower than GPU
                """
                # 1. aggregate inputs:
                N = sum(1 for _ in pseudo_train_ds.unbatch())
                if hc == 1:
                    PS = np.zeros((N, K), dtype=dtype)
                else:
                    # activations of previous to last layer to be saved if using multiple heads.
                    presize = 2048
                    PS_pre = np.zeros((N, presize), dtype=dtype)
                now = time.time()
                l_dl = sum(1 for _ in pseudo_train_ds)
                time.time()
                batch_time = MovingAverage(intertia=0.9)
                # self.model.headcount = 1
                for batch_idx, ((data, _), _selected) in enumerate(pseudo_train_ds):
                    mass = data.shape[0]
                    if hc == 1:
                        p = tfk.activations.softmax(model(data), 1)
                        PS[_selected, :] = p.numpy().astype(dtype)
                    else:
                        p = model(data)
                        PS_pre[_selected, :] = p.numpy().astype(dtype)
                    batch_time.update(time.time() - now)
                    now = time.time()
                    if batch_idx % 50 == 0:
                        print(f"Aggregating batch {batch_idx:03}/{l_dl}, speed: {mass / batch_time.avg:04.1f}Hz",
                            end='\r', flush=True)
                # self.model.headcount = self.hc
                print("Aggreg of outputs  took {0:.2f} min".format((time.time() - now) / 60.), flush=True)

                # 2. solve label assignment via sinkhorn-knopp:
                if hc == 1:
                    L = optimize_L_sk(L, PS, outs, dtype=np.float64, nh=0)
                else:
                    for nh in range(hc):
                        print("computing head %s " % nh, end="\r", flush=True)
                        tl = getattr(model, "top_layer%d" % nh)
                        time_mat = time.time()

                        # clear memory
                        try:
                            del PS
                        except:
                            pass

                        # apply last FC layer (a matmul and adding of bias)
                        PS = (PS_pre @ tl.weight.cpu().numpy().T.astype(dtype)
                                + tl.bias.cpu().numpy().astype(dtype))
                        print(f"matmul took {(time.time() - time_mat)/60:.2f}min", flush=True)
                        PS = py_softmax(PS, 1)
                        L = optimize_L_sk(L, PS, outs, dtype=np.float64, nh=nh)
                return L
            """
            if not args.cpu and torch.cuda.device_count() > 1:
                sk.gpu_sk(self)
            else:
                self.dtype = np.float64
                sk.cpu_sk(self)

            # save Label-assignments: optional
            # torch.save(self.L, os.path.join(self.checkpoint_dir, 'L', str(niter) + '_L.gz'))

            # free memory
            self.PS = 0
            """
            return cpu_sk(model, pseudo_train_ds, L, outs, hc, K)
        
        @tf.function
        def train_step(data, selected, model, optimizer, criterion, L, hc=1):
            with tf.GradientTape() as tape:
                #################### train CNN ####################################################
                final = model(data)

                if hc == 1:
                    # loss = criterion(L[0, selected], final)
                    loss = criterion(tf.gather(L[0], selected), final)
                else:
                    # loss = tf.reduce_mean(tf.concat([criterion(L[h, selected], final[h]) for h in range(hc)], axis=0))
                    loss = tf.reduce_mean(tf.concat([criterion(tf.gather(L[h], selected), final[h]) for h in range(hc)], axis=0))

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            return loss

        def optimize_epoch(model, optimizer, train_ds, epoch, optimize_times, L, hc, ncl, outs, validation=False):
            print(f"Starting epoch {epoch}, validation: {validation} " + "="*30, flush=True)

            loss_value =tfk.metrics.Mean()
            # house keeping
            # XE = torch.nn.CrossEntropyLoss()
            XE = tfk.losses.SparseCategoricalCrossentropy()

            train_ds_batch_num = sum(1 for _ in train_ds)
            for iter, ((data, _), selected) in enumerate(train_ds): # for in batch
                niter = epoch * train_ds_batch_num + iter
                batch_size = data.shape[0]

                print("\t", L)
                if (niter * batch_size) >= optimize_times[-1]:
                    ############ optimize labels #########################################
                    # model.headcount = 1
                    print('Optimizaton starting', flush=True)
                    _ = optimize_times.pop()
                    L = optimize_labels(L, model, train_ds, outs, hc, K=ncl) # TODO update psudo labels
                print("\t", L)

                loss = train_step(data, selected, model, optimizer, XE, L, hc=hc)
                loss_value.update_state(loss)

                # print(niter, " Loss: {0:.3f}".format(loss.item()), flush=True)

            return {'loss': loss_value.result().numpy()}

        def optimize(model, train_ds, n_epochs, lr=1e-3, hc=1, ncl=3, nopts=100):
            """
            ここが表現学習部分?

            Perform full optimization.

            ncl', default=3000, type=int, 
            help='number of clusters per head (default: 3000)')

            hc', default=1, type=int, 
            help='number of heads (default: 1)')

            nopts', default=100, type=int, 
            help='number of pseudo-opts (default: 100)'
            """
            # preparetaion --------------------
            outs = [ncl]*hc
            N = sum(1 for _ in train_ds.unbatch()) # len(train_ds.dataset) # TODO 全データのサイズ
            # optimization times (spread exponentially), can also just be linear in practice (i.e. every n-th epoch)
            optimize_times = [(n_epochs+2)*N] + ((n_epochs+1.01)*N*(np.linspace(0, 1, nopts)**2)[::-1]).tolist()

            # print("\t", outs, N)
            # print("\t", optimize_times)

            # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
            #                             # weight_decay=self.weight_decay,
            #                             # momentum=self.momentum,
            #                             lr=lr)
            optimizer = tfk.optimizers.SGD(lr)

            print('We will optimize L at epochs:', [np.round(1.0 * t / N, 2) for t in optimize_times], flush=True)

            # initiate labels as shuffled.
            L = np.zeros((hc, N), dtype=np.int32) # 疑似ラベル
            for nh in range(hc):
                for _i in range(N):
                    L[nh, _i] = _i % outs[nh]
                L[nh] = np.random.permutation(L[nh])
            L = tf.convert_to_tensor(L, dtype=tf.int64)

            print("\t", L)
            # end preparetaion --------------------


            # Perform optmization ###############################################################
            train_loss_results = []
            for epoch in range(n_epochs):	
                m = optimize_epoch(model, optimizer, train_ds, epoch, optimize_times, L, hc, ncl, outs, validation=False)

                print("Epoch[{}/{}] Loss:{}".format(epoch+1, n_epochs, m['loss']))
                train_loss_results.append(m['loss'])
            # print(f"optimization completed. Saving model to {os.path.join(self.checkpoint_dir,'model_final.pth.tar')}")
            # torch.save(self.model, os.path.join(self.checkpoint_dir, 'model_final.pth.tar'))
            return train_loss_results, model

        
        # concat classifier
        inputs = tfk.Input((128, 128, 3))

        encoder = encoder_model
        encoder.trainable = True

        embeddings = encoder(inputs, training=True)
        embeddings = tfk.layers.GlobalAveragePooling2D()(embeddings)
        out = tfk.layers.Dense(n_classes)(embeddings)

        encoder_model = tfk.Model(inputs, out)

        # Training
        loss, sela_model = optimize(encoder_model, train_ds, n_epochs=n_epochs, ncl=n_classes)

        hist_sela = {"loss": loss}
        return sela_model, hist_sela

    # SeLa use Dataset with image index too
    train_size = sum(1 for _ in train_ds.unbatch())
    train_ds_with_index = tf.data.Dataset.zip( (train_ds.unbatch(), tf.data.Dataset.range(train_size)) ).batch(batch_size)
    # for i, ((data, label), selected) in enumerate(train_ds_with_index):
    #     print(data.shape, label.shape, selected.shape)
    #     print(data.shape, label, selected)
    #     if i > 1: sys.exit()

    SeLa_model, hist_sela = training_SeLa_Encoder(encoder_model, train_ds_with_index, n_epochs=encoder_epochs, n_classes=n_classes*2)

    # return hist_sela

    # Classifier
    
    # model
    def supervised_model(projection: tfk.Model, output_size):
        inputs = tfk.Input((128, 128, 3))
        projection.trainable = False

        r = projection(inputs, training=False)
        outputs = tfk.layers.Dense(output_size, activation='softmax')(r)

        supervised_model = tfk.Model(inputs, outputs)
    
        return supervised_model


    # SeLa_model.summary()

    # Encoder model with non-linear projections
    projection = tfk.Model(SeLa_model.input, SeLa_model.layers[-2].output)
    linear_model = supervised_model(projection, n_classes)

    # linear_model.summary()

    # Training
    hist = training_mixup(linear_model, train_ds, test_ds, loss, optimizer, n_epochs, batch_size, n_classes, 0.2, 
                            output_best_weights=output_best_weights, weight_name=weight_name, train_weights=train_weights)

    return hist




# Model
model = ResNet50(weights=None, include_top=False)
# inputs = tfk.Input((128, 128, 3))
# res = ResNet50(weights=None, include_top=True, input_shape=(128, 128, 3))(inputs)
# outputs = tfk.layers.Dense(n_classes, activation='softmax')(res)
# model = tfk.Model(inputs, outputs)
# model.summary()

# Loss
loss = tfk.losses.CategoricalCrossentropy()

# Optimizer
opt = tfk.optimizers.Adam(lr)

# Training
# train_ds = train_ds.unbatch().batch(4).take(2)
hist = training_SeLa(model, train_ds, test_ds, loss, opt, n_epochs, batch_size,
                        n_classes, output_best_weights=False, #weight_name=str(result_path / 'best_param'),
                        encoder_epochs=1)
print(hist)

# hist_file_path = str(result_path / 'history.csv')
# pd.DataFrame(hist).to_csv(hist_file_path)
