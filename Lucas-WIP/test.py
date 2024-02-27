import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.metrics import Precision
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import numpy as np

# Load dataset

df['implicit_or_explicit'] = df['implicit_or_explicit'].fillna('none')
label_encoder_imp_exp = LabelEncoder()
df['implicit_or_explicit'] = label_encoder_imp_exp.fit_transform(df['implicit_or_explicit'])

df_hate = df[df['hate_or_not_hate'] == 'hate']

print(df_hate)