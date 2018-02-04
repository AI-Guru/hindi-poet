import urllib.request
import os
import zipfile
import glob
import numpy as np
import random
from keras import models
from keras import layers
from keras import callbacks
from keras.utils import plot_model
import matplotlib.pyplot as plt
import progressbar
from indic_transliteration import sanscript


# Path where the corpus will end up.
corpus_path = "corpus"

# Hyperparameters.
transliteration = True # Transliterates the corpus.
input_length = 40 # Length of the input sequence.
output_length = 1 # Length of the output sequence.
data_set_size = 100000 # Size of the data-set to train on.
num_epochs = 250 # Number of epochs to train.
batch_size = 512 # Batch size during training.
hidden_size = 350 # Size of the hidden layer.
generation_length = 160 # Size of the strings that are generated.


def main():
    """ The main-method. """

    # Makes sure that the corpus is on the hard-drive.
    ensure_corpus()

    # Load the data.
    (train_input, train_output) = load_data()
    print("train_input", train_input.shape)
    print("train_output", train_output.shape)

    # Create the model.
    global model
    model = create_model()

    # This callback is invoked at the end of each epoch. In special
    # circumstances a prediction is done.
    generate_callback = callbacks.LambdaCallback(on_epoch_end=on_epoch_end)

    # Trains the model.
    history = model.fit(
        train_input, train_output,
        epochs=num_epochs, batch_size=batch_size,
        callbacks=[generate_callback]
    )

    # Plot the history.
    plot_history(history)


def ensure_corpus():
    """ Makes sure that the corpus is on the hard-drive."""

    # Do nothing if the filder already exists-
    if os.path.exists("corpus") == False:
        # Download the whole git-repository as a zip.
        print("Downloading corpus...")
        corpus_url = "https://github.com/cltk/hindi_text_ltrc/archive/master.zip"
        corpus_zip_path = "master.zip"
        urllib.request.urlretrieve(corpus_url, corpus_zip_path)

        # Unzip the whole git-repository to the corpus-path.
        print("Unzipping corpus...")
        zip_file = zipfile.ZipFile(corpus_zip_path, 'r')
        zip_file.extractall(corpus_path)
        zip_file.close()

        # Remove the zip-file.
        os.remove(corpus_zip_path)


def load_data():
    """ Loads the data from the corpus. """

    # Get paths to all files.
    glob_path = os.path.join(corpus_path, "**/*.txt")
    paths = glob.glob(glob_path, recursive=True)

    # Load all files to memory.
    print("Loading all files...")
    file_contents = []
    for path in paths:
        file_content = open(path, "r").read()
        if transliteration == True:
            file_content = sanscript.transliterate(file_content, sanscript.DEVANAGARI, sanscript.IAST)
        file_content = clean_text(file_content)
        file_contents.append(file_content)

    # Getting character set.
    print("Getting character set...")
    global full_text
    full_text = " ".join(file_contents)
    global character_set
    character_set = get_character_set(full_text)
    print("Character set:", character_set, len(character_set))

    # Process the data.
    data_input = []
    data_output = []
    current_size = 0
    print("Generating data set...")
    bar = progressbar.ProgressBar(max_value=data_set_size)
    while current_size < data_set_size:
        random_file_content = random.choice(file_contents)

        random_string = random_substring_of_length(random_file_content, input_length + output_length)

        random_string_encoded = encode_string(random_string)

        input_sequence = random_string_encoded[:input_length]
        output_sequence = random_string_encoded[input_length:]

        data_input.append(input_sequence)
        data_output.append(output_sequence)

        current_size += 1
        bar.update(current_size)
    bar.finish()

    # Done.
    train_input = np.array(data_input)
    train_output = np.array(data_output)
    return (train_input, train_output)


def clean_text(text):
    """ Cleans a text. """

    text = text.replace("\t", " ")
    text = text.replace("\n", " ")
    text = text.replace("।", " ")
    text = text.replace("0", " ")
    text = text.replace("1", " ")
    text = text.replace("2", " ")
    text = text.replace("3", " ")
    text = text.replace("4", " ")
    text = text.replace("5", " ")
    text = text.replace("6", " ")
    text = text.replace("7", " ")
    text = text.replace("8", " ")
    text = text.replace("9", " ")
    text = " ".join(text.split())
    return text


def get_character_set(string):
    """ Retrieves the unique set of characters. """

    return sorted(list(set(string)))


def create_model():
    """ Creates the model. """

    input_shape = (input_length, len(character_set))

    model = models.Sequential()
    #model.add(layers.LSTM(hidden_size, input_shape=input_shape, activation="relu"))
    #model.add(layers.SimpleRNN(hidden_size, input_shape=input_shape, activation="relu"))
    model.add(layers.GRU(hidden_size, input_shape=input_shape, activation="relu"))
    model.add(layers.Dense(output_length * len(character_set), activation="relu"))
    model.add(layers.Reshape((output_length, len(character_set))))
    model.add(layers.TimeDistributed(layers.Dense(len(character_set), activation="softmax")))
    model.summary()

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    return model


def plot_history(history):
    """ Plots the history. """

    # Render the accuracy.
    plt.plot(history.history['acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("accuracy.png")
    plt.clf()

    # Render the loss.
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("loss.png")


def on_epoch_end(epoch, logs):
    """ This callback is invoked at the end of each epoch. """

    # Do some magic every ten epochs, but not in the first one.
    if epoch % 10 == 0 and epoch != 0:
        print("")

        # Try different epochs.
        for temperature in [0.0, 0.25, 0.5, 0.75, 1.0]:
            print("Temperature:", temperature)
            global full_text
            random_string = random_substring_of_length(full_text, input_length)
            result_string = random_string
            print("Seed string:  ", random_string)
            input_sequence = encode_string(random_string)

            # Generate a string.
            while len(result_string) < generation_length:
                output_sequence = model.predict(np.expand_dims(input_sequence, axis=0))
                output_sequence = output_sequence[0]
                decoded_string = decode_sequence(output_sequence, temperature)
                result_string += decoded_string
                input_sequence = input_sequence[output_length:]
                input_sequence = np.concatenate((input_sequence, output_sequence), axis=0)

            print("Result string:", result_string, len(result_string))


def random_substring_of_length(string, length):
    """ Retrieves a random substring of a fixed length from a string. """

    start_index = random.randint(0, len(string) - length)
    return string[start_index:start_index + length]


def encode_string(string):
    """ Encodes a string in order to use it in the Neural Network context. """

    encoded_string = []
    for character in string:
        encoded_character = np.zeros((len(character_set),))
        one_hot_index = character_set.index(character)
        encoded_character[one_hot_index] = 1.0
        encoded_string.append(encoded_character)
    return np.array(encoded_string)


def decode_sequence(sequence, temperature=0.0):
    """ Decodes a predicted sequence into a string. Uses temperature. """

    result_string = ""
    for element in sequence:
        index = get_index_from_prediction(element)
        character = character_set[index]
        result_string += character
    return result_string


def get_index_from_prediction(prediction, temperature=0.0):
    """ Gets an index from a prediction. """

    # Zero temperature - use the argmax.
    if temperature == 0.0:
        return np.argmax(prediction)

    # Non-zero temperature - do some random magic.
    else:
        prediction = np.asarray(prediction).astype('float64')
        prediction = np.log(prediction) / temperature
        exp_prediction= np.exp(prediction)
        prediction = exp_prediction / np.sum(exp_prediction)
        probabilities = np.random.multinomial(1, prediction, 1)
        return np.argmax(probabilities)


def browser_download():
    """ Download files in the browser. """

    from google.colab import files

    files.download("accuracy.png")
    files.download("loss.png")


if __name__ == "__main__":
    main()
