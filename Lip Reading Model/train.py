import numpy as np
import matplotlib.pyplot as plt
import os

from keras.models import Sequential
from keras.layers import Dense, Convolution3D, ZeroPadding3D, Activation, MaxPooling3D, Flatten, Dropout, BatchNormalization
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.utils import to_categorical, plot_model

np.random.seed(7)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})


# dataset soure destination 
video_path = '<path to video mp4 files>'

# folder to save the preprocessed samples
NPY_FOLDER = '<path to store numpy array files of a video>'

# sizes of mouth region -> input shape
WIDTH = 24
HEIGHT = 32
DEPTH = 28

# print info when processing data
debug = False

# list all words
print(", ".join(os.listdir(video_path)))

# train_words subset of words for training
train_words = [name for name in os.listdir(video_path) if os.path.isdir(os.path.join(video_path, name))]
print(len(train_words))

classes = len(train_words)  # len(words)

# one hot encoding the labels
labels = train_words
num_labels = [i for i in range(0, len(labels))]
hot_labels = to_categorical(num_labels)

num_labels_dict = dict(zip(labels, num_labels))
hot_labels_dict = dict(zip(labels, hot_labels))

#custome data generator for keras training
def sample_generator(basedir, set_type, batch_size):
    """Replaces Keras' native ImageDataGenerator."""
    
    # Directory from which to load samples
    directory = os.path.join(basedir, set_type)
    
    # Create NumPy arrays to contain batch of features and labels
    batch_features = np.zeros((batch_size, DEPTH,WIDTH, HEIGHT, 1))
    batch_labels = np.zeros((batch_size, classes), dtype="uint8")
    
    file_list = []
    # Populate with file paths and labels
    for word_folder in labels:
        file_list.extend((word_folder, os.path.join(directory, word_folder, word_name))                          
        for word_name in os.listdir(os.path.join(directory, word_folder)))
    
    while True:
        for b in range(batch_size):
            
            i = np.random.choice(len(file_list), 1)[0]
            
            sample = np.load(file_list[i][1])  # get random sample
            
            # Normalize to [-1; 1]
            sample = (sample.astype("float16") - 128) / 128 
            sample = sample.reshape(sample.shape + (1,))
    
            batch_features[b] = sample
            batch_labels[b] = hot_labels_dict[file_list[i][0]]  # get hot_labels vector

        yield (batch_features, batch_labels)


# ## Shape and data type of generated samples 

a = sample_generator(NPY_FOLDER, "train", 2)
s = next(a)
print(s[0].dtype, s[1].dtype)
print(s[0].shape, s[1].shape)

# load all data at once into arrya - e.g when evaluating the test subset
def load_data(basedir, set_type):

    directory = os.path.join(basedir, set_type)
    file_list = []
    for word_folder in labels:
        file_list.extend((word_folder, os.path.join(directory, word_folder, word_name))                          for word_name in os.listdir(os.path.join(directory, word_folder)))
    
    #randomise data
    shuffle(file_list)
    
    X = []
    y = []
    
    for f in file_list:
        
        sample = np.load(f[1])
        sample = (sample.astype("float16") - 128) / 128  # normalize to 0 - 1
        X.append(sample)
        
        y.append(hot_labels_dict[f[0]])
    
    X = np.array(X)
    X = X.reshape(X.shape + (1,))

    return (X, np.array(y))


# ## Train the Lip Reading Model ##

models_dir = '<directory where trained model is saved>'


def get_model(dropout_rate):
    model = Sequential()
    model.add(Convolution3D(64, (5, 5, 5), padding='same', activation='relu', input_shape=(DEPTH, WIDTH, HEIGHT, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(3, 3, 3)))
    model.add(Dropout(dropout_rate))
    model.add(Convolution3D(128, (5, 5, 5), padding='same', activation='relu'))
    model.add(Convolution3D(128, (5, 5, 5), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(3, 3, 3)))
    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(classes, activation='softmax'))
        
    return model
    

opt = Adam(lr=1e-4)

model = get_model(0.4)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


# ## Save and plot the 3D-CNN architecture 

# save model architecture to json
model_json = model.to_json()
with open(models_dir + 'model_name' + ".json", "w") as json_file:
    json_file.write(model_json)

# plot the model architecture
model.summary()
plot_model(model, to_file='outputs/architecture_{}.pdf'.format('model_name'), show_shapes=True, show_layer_names=False)


# ## Initialize checkpoints for keras model training

tensorboard = TensorBoard(log_dir="logs/{}".format('model_name'),
                          write_graph=True, write_images=True)
    
filepath = models_dir + 'model_name' + ".h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1,
                             save_best_only=True, save_weights_only=True, mode='max')

earlyStopping = EarlyStopping(monitor='val_accuracy', patience=4, verbose=1, mode='max')

csv_logger = CSVLogger('outputs/log_{}.csv'.format('model_name'), append=True, separator=';')

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.0001)


# ## Run the training model
nb_epoch = 30
batch_size = 16

num_examples = classes * 900
num_val_examples = classes * 90

history = model.fit_generator(
          generator=sample_generator(NPY_FOLDER, "train", batch_size),
          epochs=nb_epoch,
          steps_per_epoch=num_examples // batch_size,
          validation_data=sample_generator(NPY_FOLDER, "val", batch_size),
          validation_steps=num_val_examples // batch_size,
          verbose=True,
          callbacks = [tensorboard, checkpoint, earlyStopping, csv_logger] # learning_rate_reduction
)


# ## Plot and Save the training results

def plot_and_save_training():
    plt.figure(1, figsize=(8,8))
    # summarize history for accuracy
    plt.subplot(211)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('no. of epoch')
    plt.legend(['train', 'val'], loc='lower right')
    plt.grid()

    # summarize history for loss
    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('no. of epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.grid()
    
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
    
    plt.savefig('outputs/train_{}.pdf'.format('model_name'))
    plt.show()

plot_and_save_training()



