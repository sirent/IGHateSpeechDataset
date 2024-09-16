import pandas as pd

nama_file = 'Baseline Dataset'

# Membaca file CSV (ganti 'nama_file.csv' dengan nama file Anda)
df = pd.read_csv('./Preprocessed data/Raw Data/' + nama_file + '.csv')

# Mengubah semua teks dalam kolom 'comment' menjadi huruf kecil
df['Comment'] = df['Comment'].astype(str).str.lower()

# Menyimpan perubahan ke file CSV baru (opsional)
df.to_csv('./Preprocessed data/Baseline Dataset CSV/' + nama_file + '_lowercase.csv', index=False)