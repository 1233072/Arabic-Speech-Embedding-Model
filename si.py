from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Conv3D, MaxPool2D, Flatten, BatchNormalization, Activation, concatenate, Input, \
    ZeroPadding2D, AveragePooling2D, MaxPooling2D, Add, Dropout, LSTM, Reshape, Layer
from tensorflow.keras.initializers import glorot_uniform
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras import metrics
import matplotlib.pyplot as plt
from tensorflow.keras import metrics
from statistics import mean
from sklearn.model_selection import KFold
# import KNeighbors ClaSSifier from sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score, roc_curve, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB


visual_input = Input(shape=(120, 120, 3))
visual_model = Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation="relu")(visual_input)
visual_model = BatchNormalization()(visual_model)
visual_model = MaxPool2D(pool_size=(2, 2))(visual_model)
visual_model = Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu")(visual_model)
visual_model = BatchNormalization()(visual_model)
visual_model = MaxPool2D(pool_size=(2, 2))(visual_model)
visual_model = Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(visual_model)
visual_model = BatchNormalization()(visual_model)
visual_model = MaxPool2D(pool_size=(2, 2))(visual_model)
visual_model = Flatten()(visual_model)
dense1 = Dense(512, activation="relu")(visual_model)
dense1 = BatchNormalization()(dense1)
dense2 = Dense(256, activation="relu")(dense1)
dense2 = BatchNormalization()(dense2)
output = Dense(256)(dense2)

embedding = Model(visual_input, output, name="Embedding")



class DistanceLayer(Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)


anchor_input = Input(name="anchor", shape=(120, 120, 3))
positive_input = Input(name="positive", shape=(120, 120, 3))
negative_input = Input(name="negative", shape=(120, 120, 3))

distances = DistanceLayer()(
    embedding(preprocess_input(anchor_input)),
    embedding(preprocess_input(positive_input)),
    embedding(preprocess_input(negative_input)),
)

siamese_network = Model(
    inputs=[anchor_input, positive_input, negative_input], outputs=distances
)

siamese_network.load_weights('siamese_network_Arabic_speech_embedding.h5')

feature_embedding = siamese_network.get_layer('Embedding')

input = Input(shape=(120, 120, 3))
x = feature_embedding(input)
model = Model(input, x)

model.summary()

# ----- KNN

train_datagen_visual = ImageDataGenerator(rescale=1. / 255)

train = pd.read_csv('speakers.csv')
train_mel = pd.read_csv('si_mel_spec.csv')

# As we are going to divide dataset
df = train.copy()
df_mel = train_mel.copy()

# Creating X, Y for training
# train_y = df.Class
# train_x = df.drop(['Class'],axis=1)

# k-fold
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Variable for keeping count of split we are executing
j = 0

# Initializing Data Generators
train_datagen = ImageDataGenerator(rescale=1. / 255)

TRAIN_PATH_VISUAL = "si_mel_spec"



# instantiate the model
knn = KNeighborsClassifier(n_neighbors=1)
mlp = MLPClassifier()
adaboost = AdaBoostClassifier()
dtc = DecisionTreeClassifier(max_depth=50)
rf = RandomForestClassifier()
lr = LogisticRegression(solver='liblinear')
gb = GradientBoostingClassifier()
nv = GaussianNB()

inputt = Input(shape=(256,))
dense = Dense(1024, activation="relu")(inputt)
batchnorm1 = BatchNormalization()(dense)
dense = Dense(512, activation="relu")(batchnorm1)
batchnorm2 = BatchNormalization()(dense)
dense = Dense(256, activation="relu")(batchnorm2)
batchnorm3 = BatchNormalization()(dense)
classification = Dense(units=31, activation="softmax")(batchnorm3)

model2 = Model(inputt, classification)
ada_grad = Adam(lr=0.0001)
model2.compile(optimizer=ada_grad, loss='categorical_crossentropy', metrics=['accuracy'])
model2.summary()



# K-fold Train and test for each split
classifiers = [model2, knn, mlp, adaboost, dtc, rf, lr, gb, nv]

#PUT IT HERE
for clf in classifiers:
    acc = []
    f1 = []
    prec = []
    rec = []
    kappa = []
    for train_idx, val_idx in list(kfold.split(df, df)):
        train_idx = df.iloc[train_idx]
        val_idx = df.iloc[val_idx]
        
        x_train_visual_df = df_mel.loc[df_mel['segment'].isin(train_idx['segment'])]
        x_valid_visual_df = df_mel.loc[df_mel['segment'].isin(val_idx['segment'])]
        
        x_train_visual_df = x_train_visual_df.drop(['segment'], axis=1)
        x_valid_visual_df = x_valid_visual_df.drop(['segment'], axis=1)
        
        y_train = x_train_visual_df.Class
        y_test = x_valid_visual_df.Class
        
        j += 1
        
        training_set = train_datagen.flow_from_dataframe(dataframe=x_train_visual_df, directory=TRAIN_PATH_VISUAL,
                                                            x_col="mel_spec", y_col="Class",
                                                            class_mode=None,
                                                            target_size=(120, 120), batch_size=32, shuffle=False)
        
        validation_set = train_datagen.flow_from_dataframe(dataframe=x_valid_visual_df, directory=TRAIN_PATH_VISUAL,
                                                            x_col="mel_spec", y_col="Class",
                                                            class_mode=None,
                                                            target_size=(120, 120), batch_size=32, shuffle=False)
        
        print('Training embeddings:')
        training_embeddings = model.predict_generator(training_set, (x_train_visual_df.shape[0] // 32) + 1, verbose=1)
        print('Testing_embeddings:')
        testing_embeddings = model.predict_generator(validation_set, (x_valid_visual_df.shape[0] // 32) + 1, verbose=1)
        
        if clf == classifiers[0]:
        
            # fit the model to the training set
            clf.fit(training_embeddings, pd.get_dummies(y_train), steps_per_epoch=len(training_embeddings)//32+1, epochs=15)
        
            y_pred = clf.predict(testing_embeddings)

            revese = np.argmax(y_pred, axis=1)+1
            y_pred = np.where(revese > 9, revese + 2, revese)
        
        else:
            # fit the model to the training set
            clf.fit(training_embeddings, y_train)
        
            y_pred = clf.predict(testing_embeddings)
        
        acc.append(accuracy_score(y_test, y_pred))
        f1.append(f1_score(y_test, y_pred, average='macro'))
        prec.append(precision_score(y_test, y_pred, average='macro'))
        rec.append(recall_score(y_test, y_pred, average='macro'))
        kappa.append(cohen_kappa_score(y_test, y_pred))
            
    print('-------')
    print(clf)
    print('-------')
    print("Accuracy:", mean(acc))
    print("F1", mean(f1))
    print("Precision", mean(prec))
    print("Recall", mean(rec))
    print("Cohen's kappa:", mean(kappa))
    print('\n')



