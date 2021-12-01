import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import robust_scale,minmax_scale
from keras.layers import Dropout, Input, Concatenate, Dense, \
    Conv2D, BatchNormalization, MaxPool2D, Flatten
from keras import losses
from keras.models import load_model, Model
from keras import optimizers
from sklearn.model_selection import KFold
import keras

bin_size = 32
def load_dict(path):
    res ={}
    lines = open(path).readlines()
    for line in lines:
        x_list = line.strip().split(",")
        name = x_list[0]
        vec = [np.float(x) for x in x_list[1:]]
        res[name]=vec
    return res

def get_model():
    inx = Input(shape=(bin_size, bin_size, 4))
    x = BatchNormalization()(inx)
    x = Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
    x = Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Conv2D(128, kernel_size=(3, 3), activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Conv2D(128, kernel_size=(3, 3), activation="relu")(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Conv2D(256, kernel_size=(3, 3), activation="relu")(x)
    x = Conv2D(256, kernel_size=(3, 3), activation="relu")(x)
    x = Flatten()(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    x_out = Dense(2, activation="softmax")(x)
    model = Model(inputs=[inx], outputs=[x_out])
    print(model.summary())
    return model

lnc_kmer_dict = load_dict("./embedding/lnc_kmer_dict.txt")
lnc_doc2vec_dict = load_dict("./embedding/lnc_doc2vec_dict.txt")
lnc_ctd_dict = load_dict("./embedding/lnc_ctd_dict.txt")
lnc_role2vec_dict = load_dict("./embedding/lnc_role2vec_dict.txt")

mir_kmer_dict = load_dict("./embedding/mir_kmer_dict.txt")
mir_doc2vec_dict = load_dict("./embedding/mir_doc2vec_dict.txt")
mir_ctd_dict = load_dict("./embedding/mir_ctd_dict.txt")
mir_role2vec_dict = load_dict("./embedding/mir_role2vec_dict.txt")

positive_pairs_path = "./data/lnc_mir_pairs.txt"
negative_pairs_path = "./data/negative_pairs.txt"

positive_pairs = [[line.strip().split(",")[0],line.strip().split(",")[1]] for line in open(positive_pairs_path,"r").readlines()]
negative_pairs = [[line.strip().split(",")[0],line.strip().split(",")[1]] for line in open(negative_pairs_path,"r").readlines()]
labels = [1] * len(positive_pairs)+[0] * len(negative_pairs)
labels_softmax = keras.utils.to_categorical(labels,num_classes=2)

all_pairs = positive_pairs+negative_pairs
all_mats =[]

for pair in all_pairs:

    lnc = pair[0]
    mir = pair[1]

    lnc_ver = lnc_kmer_dict[lnc]
    mir_vec = mir_kmer_dict[mir]
    H, xedges, yedges = np.histogram2d(lnc_ver, mir_vec, bins=bin_size,normed=True)
    # H = minmax_scale(H)
    H_kmer = H[:, :, np.newaxis]

    lnc_ver = lnc_doc2vec_dict[lnc]
    mir_vec = mir_doc2vec_dict[mir]
    H, xedges, yedges = np.histogram2d(lnc_ver, mir_vec, bins=bin_size,normed=True)
    # H = minmax_scale(H)
    H_doc2vec = H[:, :, np.newaxis]

    lnc_ver = lnc_ctd_dict[lnc]
    mir_vec = mir_ctd_dict[mir]
    H, xedges, yedges = np.histogram2d(lnc_ver, mir_vec, bins=bin_size,normed=True)
    # H = minmax_scale(H)
    H_ctd = H[:, :, np.newaxis]

    lnc_ver = lnc_role2vec_dict[lnc]
    mir_vec = mir_role2vec_dict[mir]
    H, xedges, yedges = np.histogram2d(lnc_ver, mir_vec, bins=bin_size,normed=True)
    # H = minmax_scale(H)
    H_role2vec = H[:, :, np.newaxis]

    mat = np.concatenate([H_kmer,H_doc2vec,H_ctd,H_role2vec],axis=2)
    all_mats.append(mat)

all_mats_np = np.array(all_mats)
labels_np = np.array(labels_softmax)

print("all_mats_np",all_mats_np.shape)
print("labels_np",labels_np.shape)

accs=[]
f1s=[]
sens=[]
spes=[]
mccs=[]
aucs=[]

kf = KFold(n_splits=5,shuffle=True,random_state=42)
for train_index, test_index in kf.split(list(range(len(all_mats_np)))):
    train_data = all_mats_np[train_index]
    train_label = labels_np[train_index]

    test_data = all_mats_np[test_index]
    test_label = labels_np[test_index]

    model = get_model()
    adma = optimizers.adam(lr=0.000001)
    model.compile(optimizer=adma,
                  loss=losses.binary_crossentropy,
                  metrics=['accuracy'])

    model.fit(x=[train_data],
              y=train_label,
              batch_size=1024,
              shuffle=True,
              epochs=256,
              verbose=2,
              validation_split=0.2)

    res_pred = model.predict([test_data])
    res_pred_s = res_pred[:, 1]
    res_label = np.argmax(res_pred, axis=1)
    test_label = np.argmax(test_label, axis=1)

    confusion = metrics.confusion_matrix(test_label, res_label)

    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]

    acc = metrics.accuracy_score(y_true=test_label, y_pred=res_label)
    f1_score = metrics.f1_score(y_true=test_label, y_pred=res_label)
    sensitivity = TP / float(FN + TP)
    specificity = TN / float(TN + FP)
    mcc = metrics.matthews_corrcoef(y_true=test_label, y_pred=res_label)
    auc = metrics.roc_auc_score(y_true=test_label, y_score=res_pred_s)

    print("acc", acc)
    print("auc", auc)
    print("f1_score", f1_score)
    print("sensitivity", sensitivity)
    print("specificity", specificity)
    print("mccs", mcc)

    accs.append(acc)
    f1s.append(f1_score)
    sens.append(sensitivity)
    spes.append(specificity)
    mccs.append(mcc)
    aucs.append(auc)

print()
print("acc", np.mean(accs))
print("auc", np.mean(aucs))
print("f1 score", np.mean(f1s))
print("sensitivity", np.mean(sens))
print("specificity", np.mean(spes))
print("mccs", np.mean(mccs))
