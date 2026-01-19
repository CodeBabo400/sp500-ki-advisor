import pandas as pd
import numpy as np
import os

input_file = "Abgaben/Final_Project/data/S&P 500 Historische Daten 1.csv"
output_file = "Abgaben/Final_Project/data/SP500_Cleaned.csv"


df = pd.read_csv(input_file)
df_clean = df.copy()

# Spalten umbenennen
df_clean = df_clean.rename(columns={
    'Datum': 'Date', 
    'Zuletzt': 'Close', 
    'Eröffn.': 'Open',
    'Hoch': 'High', 
    'Tief': 'Low', 
    'Vol.': 'Volume', 
    '+/- %': 'Change_Percent'
})

# Zahlenformate korrigieren: 1.000,00 -> 1000.00
cols_to_fix = ['Close', 'Open', 'High', 'Low', 'Change_Percent']

for col in cols_to_fix:
    # Zeichen ersetzen
    df_clean[col] = (df_clean[col].astype(str)
                     .str.replace('.', '', regex=False)
                     .str.replace(',', '.', regex=False)
                     .str.replace('%', '', regex=False))
    
    # In float umwandeln
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

# Datum konvertieren und als Index setzen
df_clean['Date'] = pd.to_datetime(df_clean['Date'], format='%d.%m.%Y')
df_clean = df_clean.set_index('Date').sort_index()


# Volumen entfernen 
if 'Volume' in df_clean.columns:
    df_clean = df_clean.drop(columns=['Volume'])

# Duplikate im Datum entfernen
df_clean = df_clean[~df_clean.index.duplicated(keep='first')]


# Leere Zeilen entfernen (falls durch Konvertierung Fehler entstanden sind)
df_final = df_clean.dropna()

print("\n--- Fertiger Datensatz ---")
print(df_final.head())


df_final.to_csv(output_file)