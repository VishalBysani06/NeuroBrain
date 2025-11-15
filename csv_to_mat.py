import pandas as pd
from scipy.io import savemat

# === Step 1: Load CSV ===
csv_file = r"C:\Users\abhib\Desktop\Neuro Brain\emg_output.csv"   # <-- your CSV file path
data = pd.read_csv(csv_file)

# === Step 2: Convert to MAT ===
mat_file = r"C:\Users\abhib\Desktop\Neuro Brain\emg_data.mat"

savemat(mat_file, {"emg_data": data.values})

print("Conversion complete!")
print("Saved to:", mat_file)
