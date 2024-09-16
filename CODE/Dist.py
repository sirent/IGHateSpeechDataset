import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import chardet

# # Mendeteksi encoding file
# with open('./Preprocessed data/Labelled/ManuallyLabeled.xlsx', 'rb') as f:
#     result = chardet.detect(f.read())

# print(result['encoding'])

# Membaca file CSV
data = pd.read_csv('./Preprocessed data/Labelled/ManuallyLabeled.xlsx', encoding='utf-8', errors='ignore')

# Menampilkan distribusi label
label_counts = data['Label'].value_counts()
print(label_counts)

# Visualisasi distribusi label
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='Label')
plt.title('Distribusi Label')
plt.show()
