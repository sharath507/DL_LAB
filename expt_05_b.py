


import tensorflow as tf
import numpy as np

path = tf.keras.utils.get_file("shakespeare.txt", "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt")
text = open(path, "rb").read().decode(encoding="utf-8")

vocab = sorted(set(text))
char_to_index = {char: index for index, char in enumerate(vocab)}
index_to_char = np.array(vocab)

text_as_int = np.array([char_to_index[char] for char in text])

seq_length = 100
batch_size = 64
buffer_size = 10000

# Corrected tf.data.Dataset method usage
dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
dataset = dataset.batch(seq_length + 1, drop_remainder=True)

def split_input_target(chunk):
    return chunk[:-1], chunk[1:]  # Fixed the return format

# Corrected map function usage
dataset = dataset.map(split_input_target)
dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)

# Corrected Sequential capitalization
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(vocab), 256),
    tf.keras.layers.LSTM(1024, return_sequences=True),
    tf.keras.layers.Dense(len(vocab))
])

# Fixed SparseCategoricalCrossentropy parameters
model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
model.fit(dataset, epochs=3)

def generate_text(model, start_string, num_generate=300):
    input_eval = [char_to_index[c] for c in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []

    for _ in range(num_generate):  # Fixed the loop keyword 'for_' to 'for'
        predictions = model(input_eval)  # Fixed spelling of 'predictions'
        predicted_id = tf.random.categorical(predictions[0][-1:], num_samples=1).numpy()[0, 0]  # Fixed slicing of predictions
        input_eval = tf.expand_dims([predicted_id], 0)  # Fixed expand_dims usage
        text_generated.append(index_to_char[predicted_id])  # Fixed append arguments

    return start_string + ''.join(text_generated)

print(generate_text(model, "once upon a time"))




