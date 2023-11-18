from keras.datasets import imdb
from keras.preprocessing import sequence
import tensorflow as tf
import os
import numpy as np

VOCAB_SIZE = 88584
MAXLEN = 250
BATCH_SIZE = 254

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=VOCAB_SIZE)
#train_data[8]

#have to pass same length data in NN, if review > 250 words, trim off extra words
#if review < 250 add 0s to make em = 250
train_data = sequence.pad_sequences(train_data, MAXLEN)
test_data = sequence.pad_sequences(test_data, MAXLEN)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, 32),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

model.compile(
    loss='binary_crossentropy',
    optimizer='rmsprop',
    metrics=['acc']
)

history = model.fit(train_data, train_labels, epochs=10, validation_split=0.2)

word_index = imdb.get_word_index()

def encode_text(text):
    tokens = tf.keras.preprocessing.text.text_to_word_sequence(text) #tokenization
    tokens = [word_index[word] if word in word_index else 0 for word in tokens] 
    #if tokenized word is in imdb vocab,replace it with the integer that represents it,
    # otherwise zero to show we dunno
    return sequence.pad_sequences([tokens],MAXLEN)[0]

text = 'that movie was so good, just super amazing!'
encoded = encode_text(text)
print(encoded)

reverse_word_index = {value: key for (key,value) in word_index.items()}
for key, value in reverse_word_index.items():
    print(key, ":", value)

def decode_integers(integers):
    pad = 0
    text = ''
    for num in integers:
        if num != pad:
            text += reverse_word_index[num] + ' '
    return text[:-1]

print(decode_integers(encoded))

def predict(text):
    encoded_text = encode_text(text)
    pred = np.zeros((1,250))
    pred[0] = encoded_text
    result = model.predict(pred)
    if result[0] < 0.50:
        print("This is a negative review!")
    else:
        print("This is a positive review!")
        

while True:
    rev = input("Enter movie review")
    predict(rev) 