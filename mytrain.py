import tensorflow as tf
import reader
import numpy as np

class_dim = 855
EPOCHS = 500
BATCH_SIZE = 32
init_model = "models/model_weights.h5"
log_path = 'F:/deeplearning/VoiceRecognition/VoicePrint'
train_dataset = reader.train_reader_tfrecord('dataset/train.tfrecord', EPOCHS, batch_size=BATCH_SIZE)
test_dataset = reader.test_reader_tfrecord('dataset/test.tfrecord', batch_size=BATCH_SIZE)

# sounds = train_dataset['data']
# labels = train_dataset['label']

def create_model(optimizer='rmsprop', init='glorot_uniform'):
    model = tf.keras.models.Sequential([
        tf.keras.applications.ResNet50V2(include_top=False, weights=None, input_shape=(128, None, 1)),
        tf.keras.layers.ActivityRegularization(l2=0.5),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.GlobalMaxPooling2D(),
        tf.keras.layers.Dense(units=class_dim, activation=tf.nn.softmax)
    ])

    return model




if __name__ == '__main__':
    model = create_model()
    # print("sounds:",sounds)
    # print("labels:",labels)
    print("train_dataset:",train_dataset)
    print("train_dataset:",train_dataset.shape())
    # call = tf.keras.callbacks.TensorBoard(log_dir=log_path)
    # history = model.fit(x, ylabels, epochs=200, batch_size=10, callbacks=[call])  # 不带回调函数的训练