import joblib
import json
import os

# Input .pkl file
file_path = r"C:\Users\Abhinav S  Bhat\OneDrive\Desktop\Neuro Brain\neurobrain_motor_imagery_random_forest.pkl"

# Output folder
output_folder = r"C:\Users\Abhinav S  Bhat\OneDrive\Desktop\Neuro Brain\models saved"

# Make sure the folder exists
os.makedirs(output_folder, exist_ok=True)

# Load with joblib
data = joblib.load(file_path)

# Convert objects (like RandomForestClassifier, StandardScaler) to strings for readability
clean_data = {k: str(v) for k, v in data.items()}

# Pretty print in terminal
print(json.dumps(clean_data, indent=4))

# Save as JSON in the given folder
json_path = os.path.join(output_folder, "neurobrain_motor_imagery_random_forest_formatted.json")
with open(json_path, "w") as jf:
    json.dump(clean_data, jf, indent=4)

print(f"✅ Saved formatted version to: {json_path}")
