import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM

# data
data = [
    ("hello", "ಹಲೋ"),
    ("how are you", "ನೀವು ಹೇಗಿದ್ದೀರಾ"),
    ("i am fine", "ನಾನು ಚೆನ್ನಾಗಿದ್ದೇನೆ"),
    ("thank you", "ಧನ್ಯವಾದಗಳು"),
    ("good night", "ಶುಭ ರಾತ್ರಿ")
]

input_text = []
target_text = []
input_chars = set()
target_chars = set()

for inp, target in data:
    target_text.append('\t' + target + '\n')
    input_text.append(inp)
    input_chars.update(list(inp))
    target_chars.update(list(target))

# Add the special start and end tokens explicitly
target_chars.add('\t')
target_chars.add('\n')

input_chars = sorted(list(input_chars))
target_chars = sorted(list(target_chars))

num_encoder_tokens = len(input_chars)
num_decoder_tokens = len(target_chars)
max_encoder_seq_length = max([len(txt) for txt in input_text])
max_decoder_seq_length = max([len(txt) for txt in target_text])

input_token_index = dict([(char, i) for i, char in enumerate(input_chars)])
target_token_index = dict([(char, i) for i, char in enumerate(target_chars)])

reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

# vectorize the data
encoder_input_data = np.zeros(
    (len(input_text), max_encoder_seq_length, num_encoder_tokens), dtype="float32")
decoder_input_data = np.zeros(
    (len(input_text), max_decoder_seq_length, num_decoder_tokens), dtype="float32")
decoder_target_data = np.zeros(
    (len(input_text), max_decoder_seq_length, num_decoder_tokens), dtype="float32")

for i, (inp, targ) in enumerate(zip(input_text, target_text)):
    for t, char in enumerate(inp):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    for t, char in enumerate(targ):
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.

# build the encoder and the decoder model
latent_dim = 256

encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder_lstm = LSTM(latent_dim, return_state=True)
_, state_h, state_c = encoder_lstm(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# train the model
model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=2,
    epochs=500,
    verbose=0
)
print("training the model")

# inference models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states
)

# translate function
def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_token_index["\t"]] = 1.0
    decoded_sentence = ""
    stop_condition = False
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        if sampled_char == "\n" or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True

        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.
        states_value = [h, c]
    return decoded_sentence.strip()

# test translation
for seq_index in range(len(input_text)):
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print("-")
    print("Input sentence:", input_text[seq_index])
    print("Decoded sentence:", decoded_sentence)
