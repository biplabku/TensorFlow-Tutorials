import tensorflow as tf
import pandas as pd

TRAIN_URL = "https://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']

def load_data(label_names ='Species') :

    train_path = tf.keras.utils.get_file(fname=TRAIN_URL.split('/')[-1], origin = TRAIN_URL)

    train  = pd.read_csv(filepath_or_buffer = train_path, names=CSV_COLUMN_NAMES, header = 0)

    train_features, train_label = train, train.pop(label_names)

    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)
    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header = 0)
    test_features, test_label = test, test.pop(label_names)
    # getting the features and label_names

    return (train_features, train_label), (test_features, test_label)

(train_features, train_label), (test_features, test_label) = load_data()
