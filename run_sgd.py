import numpy as np
import tensorflow as tf
import sys
import pickle as pkl
from models import Dense_2Layer, Big_Dense, Big_Conv_net, Small_Conv_net
import random
import os
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, accuracy_score


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


# Reading Arguments
rand_seed = 1
tf.random.set_seed(rand_seed)

dataset = sys.argv[1]
name = sys.argv[2]
part = sys.argv[3]
swapped = sys.argv[4]
re_enter = sys.argv[5]
noisy = sys.argv[6]
save_logs = bool(int(sys.argv[7]))

per_sw = swapped
print(dataset, name, part, swapped, re_enter, noisy, save_logs)


# Loading Dataset
if dataset == "Ozone":
    x_train = np.load(name + "/" + dataset + "/x_train_part_" + part + ".npy")
    y_train = np.load(name + "/" + dataset + "/y_train_part_" + part + ".npy")

    x_test = np.load(name + "/" + dataset + "/x_test_part_" + part + ".npy")
    y_test = np.load(name + "/" + dataset + "/y_test_part_" + part + ".npy")
    
    logs = "./logs/" + name +  "/" + dataset + "/part_" + part + "/"

    x_test = (x_test - x_train.mean(axis=0)) / (x_train.std(axis=0) + np.finfo(np.float32).eps)
    x_train = (x_train - x_train.mean(axis=0)) / (x_train.std(axis=0) + np.finfo(np.float32).eps)
elif dataset == "adult":
    x_train = np.load(name + "/" + dataset + "/log_x_train.npy")
    y_train = np.load(name + "/" + dataset + "/log_y_train.npy")

    x_test = np.load(name + "/" + dataset + "/log_x_test.npy")
    y_test = np.load(name + "/" + dataset + "/log_y_test.npy")
    
    logs = "./logs/" + name +  "/" + dataset + "/"

elif dataset == "credit":
    x_train = np.load(name + "/" + dataset + "/x_train_part_" + part + ".npy")
    y_train = np.load(name + "/" + dataset + "/y_train_part_" + part + ".npy")

    x_test = np.load(name + "/" + dataset + "/x_test_part_" + part + ".npy")
    y_test = np.load(name + "/" + dataset + "/y_test_part_" + part + ".npy")
    
    logs = "./logs/" + name +  "/" + dataset + "/part_" + part + "/"

    x_test = (x_test - x_train.mean(axis=0)) / (x_train.std(axis=0) + np.finfo(np.float32).eps)
    x_train = (x_train - x_train.mean(axis=0)) / (x_train.std(axis=0) + np.finfo(np.float32).eps)


elif dataset == "mnist":
    x_train = np.load(name + "/" + dataset + "/1_minor_0.1/x_train_" + part + ".npy")
    y_train = np.load(name + "/" + dataset + "/1_minor_0.1/y_train_" + part + ".npy")

    x_test = np.load(name + "/" + dataset + "/1_minor_0.1/x_test.npy")
    y_test = np.load(name + "/" + dataset + "/1_minor_0.1/y_test.npy")

    logs = "./logs/" + name +  "/" + dataset + "/1_minor_0.1/" + part + "/"

    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
    
    x_test = x_test / 255.0
    x_train = x_train / 255.0

elif dataset == "cifar10":
    x_train = np.load(name + "/" + dataset + "/1_minor_0.1/x_train_" + part + ".npy")
    y_train = np.load(name + "/" + dataset + "/1_minor_0.1/y_train_" + part + ".npy")

    x_test = np.load(name + "/" + dataset + "/1_minor_0.1/x_test.npy")
    y_test = np.load(name + "/" + dataset + "/1_minor_0.1/y_test.npy")

    logs = "./logs/" + name +  "/" + dataset + "/1_minor_0.1/" + part + "/"

    x_test = x_test / 255.0
    x_train = x_train / 255.0


numcl = len(np.unique(y_train))
y_train = tf.keras.utils.to_categorical(y_train, num_classes=numcl)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=numcl)

classes = y_train.shape[1]



# Initializing models
opt = tf.keras.optimizers.Adam()
if (dataset == "Ozone") or (dataset == "adult"):
    epochs = 10
    model = Dense_2Layer(x_train.shape[1], numcl)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["acc"])
elif (dataset == "credit") :
    epochs = 10
    model = Big_Dense(x_train.shape[1], numcl)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["acc"])
elif dataset == "mnist":
    epochs = 15
    model = Small_Conv_net()
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["acc"])

elif dataset == "cifar10":
    epochs = 35
    model = Big_Conv_net()
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["acc"])


N = x_train.shape[0]
indices = np.arange(N)
np.random.shuffle(indices)

if N > 64:
    batch_size = 64
else:
    batch_size = int(N / 2)


num_batches = x_train.shape[0] // batch_size
swapped = round(float(swapped) * N)

ma_param = 0.6

full_batch_losses = np.zeros(N)
EMA_batch_losses = np.zeros(N)

train_results = []
test_results = []

re_enter = (int(re_enter) == 1)
noisy = (int(noisy) == 1)
if re_enter:
    sw_times = np.zeros(N)
if noisy:
    var_noise = 0.2


# Start training
for ep in range(epochs):
    print("Epoch: ", ep)
        
    
    if (noisy & (ep > 0)):
        noise_mask = np.round(np.random.uniform(0, 1, x_train.shape))
        mask = np.in1d(indices, batch_low_swap).astype(int)
        noise = np.random.normal(0, var_noise, x_train.shape)

        ax = list(range(len(x_train.shape)))[1:]
        x_train_temp = x_train[indices] + np.expand_dims(mask, axis=ax) * x_train[indices] * noise_mask * noise
        y_train_temp = y_train[indices]

        

    else:
        x_train_temp = x_train[indices]
        y_train_temp = y_train[indices]


    for b in range(num_batches+1):
        if b == num_batches:
            batch_x = x_train_temp[b * batch_size :]
            batch_y = y_train_temp[b * batch_size :]
                
        else:
            batch_x = x_train_temp[b * batch_size : (b+1) * batch_size]
            batch_y = y_train_temp[b * batch_size : (b+1) * batch_size]
            

        with tf.GradientTape() as tape:
            preds = model(batch_x, training=True)
            losses = tf.keras.losses.categorical_crossentropy(batch_y, preds)
        
        # print(losses.numpy().mean())
        grads = tape.gradient(losses, model.trainable_variables)

        opt.apply_gradients(zip(grads, model.trainable_variables))

        if b == num_batches:
            full_batch_losses[indices[b * batch_size : ]] = losses
            EMA_batch_losses[indices[b * batch_size : ]] = ma_param * EMA_batch_losses[indices[b * batch_size : ]] + (1 - ma_param) * losses
        else:
            full_batch_losses[indices[b * batch_size : (b+1) * batch_size]] = losses
            EMA_batch_losses[indices[b * batch_size : (b+1) * batch_size]] = ma_param * EMA_batch_losses[indices[b * batch_size : (b+1) * batch_size]] + (1 - ma_param) * losses
            


    # --------------------------------TRAIN SCORES---------------------------------------------

    tr_preds = model.predict(x_train)
    # [los, s] = model.evaluate(x_train, y_train)

    los = np.mean(tf.keras.losses.categorical_crossentropy(y_train, tr_preds))
    s = accuracy_score(y_train.argmax(axis=1), tr_preds.argmax(axis=1))

    f1_macro = f1_score(y_train.argmax(axis=1), tr_preds.argmax(axis=1), average='macro')
    f1_micro = f1_score(y_train.argmax(axis=1), tr_preds.argmax(axis=1), average='micro')
    balacc = balanced_accuracy_score(y_train.argmax(axis=1), tr_preds.argmax(axis=1))
    rocauc_macro = roc_auc_score(y_train, tr_preds, average='macro', multi_class="ovr")
    rocauc_micro = roc_auc_score(y_train, tr_preds, average='micro', multi_class="ovr")
    
    print(" Train Loss: ", los)
    print(" Train Accuracy: ", s)
    print(" Train F1: ", f1_macro, ", ", f1_micro)
    print(" Train Balanced Acc: ", balacc)
    print(" Train ROC AUC: ", rocauc_macro, ", ", rocauc_micro)
    
    train_results.append([los, s, f1_macro, f1_micro, balacc, rocauc_macro, rocauc_micro])

    # ---------------------------------TEST SCORES-----------------------------------------------

    tst_preds = model.predict(x_test)
    # [los, s] = model.evaluate(x_test, y_test)

    los = np.mean(tf.keras.losses.categorical_crossentropy(y_test, tst_preds))
    s = accuracy_score(y_test.argmax(axis=1), tst_preds.argmax(axis=1))

    f1_macro2 = f1_score(y_test.argmax(axis=1), tst_preds.argmax(axis=1), average='macro')
    f1_micro2 = f1_score(y_test.argmax(axis=1), tst_preds.argmax(axis=1), average='micro')
    balacc2 = balanced_accuracy_score(y_test.argmax(axis=1), tst_preds.argmax(axis=1))
    rocauc_macro2 = roc_auc_score(y_test, tst_preds, average='macro', multi_class="ovr")
    rocauc_micro2 = roc_auc_score(y_test, tst_preds, average='micro', multi_class="ovr")

    print(" Test Loss: ", los)
    print(" Test Accuracy: ", s)
    print(" Test F1: ", f1_macro2, ", ", f1_micro2)
    print(" Test Balanced Acc: ", balacc2)
    print(" Test ROC AUC: ", rocauc_macro2, ", ", rocauc_micro2)
    
    test_results.append([los, s, f1_macro2, f1_micro2, balacc2, rocauc_macro2, rocauc_micro2])

    # ------------------------------------------------------------------------------------

    #------------------Swapping--------------------------------------
    if swapped != 0:
        print("Swapping ", swapped, " samples.")
        if re_enter:
            # sw_thr = % of epochs
            re_inds = sw_times >= int(0.2 * epochs)
            mean_loss =  full_batch_losses.mean()
            full_batch_losses[re_inds] = mean_loss

        
        ind_batch_low = np.argpartition(full_batch_losses, swapped)
        # ind_ma_low = np.argpartition(EMA_batch_losses, swapped)
        
        # ind_batch_high = np.argpartition(full_batch_losses, N - swapped)
        ind_ma_high = np.argpartition(EMA_batch_losses, N - swapped)
        
        batch_low_swap = ind_batch_low[swapped:]
        ma_high_swap = ind_ma_high[-swapped:]
        
        indices = np.concatenate((batch_low_swap, ma_high_swap))
    
        if re_enter:
            swapped_ind = ind_batch_low[:swapped]
            sw_times[indices] = 0
            sw_times[swapped_ind] += 1
        

    np.random.shuffle(indices) 


train_results = np.array(train_results)
test_results = np.array(test_results)

# Saving results
if save_logs:
    if not os.path.exists(logs):
        os.makedirs(logs)
    np.save(logs + "train_results_part" + "_" + str(per_sw) + "_" + str(re_enter) + "_" + str(noisy), train_results)
    np.save(logs + "test_results__part" + "_" + str(per_sw) + "_" + str(re_enter) + "_" + str(noisy), test_results)