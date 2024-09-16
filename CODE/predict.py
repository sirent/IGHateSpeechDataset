import numpy as np
from gensim.models import Word2Vec
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Load the trained CNN model and Word2Vec model
cnn_model = load_model('./Model/ML/cnn_text_classification_model.h5')
word2vec_model = Word2Vec.load('./Model/Words/word2vec_model.model')

# 2. Preprocess the new text (tokenization and converting to vectors)
def preprocess_text(text, word2vec_model, max_length):
    # Tokenize the input text
    sentence = text.split()
    
    # Convert each word to its corresponding Word2Vec vector
    vectorized_text = np.array([word2vec_model.wv[word] if word in word2vec_model.wv else np.zeros(100) for word in sentence])
    
    # Pad the sequence so it's the same length as the training data
    padded_text = pad_sequences([vectorized_text], maxlen=max_length, dtype='float32', padding='post')
    
    return padded_text

# 3. Function to predict the label for a new text
def predict_text(text, cnn_model, word2vec_model, max_length):
    # Preprocess the text
    padded_text = preprocess_text(text, word2vec_model, max_length)
    
    # Predict the label (output is between 0 and 1, so apply a threshold for classification)
    prediction = cnn_model.predict(padded_text)[0][0]
    
    # Classify based on a threshold of 0.5
    if prediction >= 0.5:
        return 1  # Positive class (or label 1)
    else:
        return 0  # Negative class (or label 0)

# 4. Example usage
new_text = "saya bukan cebong"
max_length = 100  # Ensure this matches the maximum length used during training

predicted_label = predict_text(new_text, cnn_model, word2vec_model, max_length)
print(f'The predicted label for the text is: {predicted_label}')
