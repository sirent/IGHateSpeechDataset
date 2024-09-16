import pandas as pd
from gensim.models import Word2Vec
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# 1. Load the data from the Excel file
df = pd.read_excel('./Preprocessed data/Labelled/ManuallyLabeled.xlsx')

# 2. Preprocess the text (Tokenization)
sentences = df['Comment'].apply(lambda x: x.split()).tolist()

# 3. Train the Word2Vec model on the tokenized sentences
word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# 4. Function to convert text to vectors using the trained Word2Vec model
def text_to_vector(sentence, word2vec_model):
    return np.array([word2vec_model.wv[word] if word in word2vec_model.wv else np.zeros(100) for word in sentence])

# 5. Convert all comments to vectors
data_vectors = [text_to_vector(sentence, word2vec_model) for sentence in df['Comment'].apply(lambda x: x.split()).tolist()]

# 6. Pad the sequences so they have the same length
max_length = max(len(vec) for vec in data_vectors)
data_vectors_padded = pad_sequences(data_vectors, maxlen=max_length, dtype='float32', padding='post')

# 7. Extract labels
labels = df['Label'].values

# 8. Define the CNN model architecture
model = Sequential()

# Add Conv1D layer (Convolutional layer)
model.add(Conv1D(filters=128, kernel_size=5, activation='relu', input_shape=(max_length, 100)))

# Add GlobalMaxPooling1D layer to reduce the dimensions
model.add(GlobalMaxPooling1D())

# Add a Dense layer with 10 units and ReLU activation
model.add(Dense(10, activation='relu'))

# Add a Dropout layer to prevent overfitting
model.add(Dropout(0.5))

# Add the output layer (binary classification)
model.add(Dense(1, activation='sigmoid'))

# Compile the model with Adam optimizer
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# 9. Train the CNN model
model.fit(data_vectors_padded, labels, epochs=10, batch_size=32)

# 10. Save the trained model and Word2Vec model
model.save('./Model/ML/cnn_text_classification_model.h5')
word2vec_model.save('./Model/Words/word2vec_model.model')
