import scipy.io as sio
import numpy as np

filepath = r"C:\Users\Abhinav S  Bhat\OneDrive\Desktop\Neuro Brain\Dataset\A02T.mat"

mat = sio.loadmat(filepath)
data_struct = mat['data'][0,0]

print("="*70)
print("EEG DATASET STRUCTURE ANALYSIS")
print("="*70)

# 1. X DATA (EEG Signals) - Main focus
print("\n" + "="*50)
print("1. EEG SIGNALS (X DATA)")
print("="*50)

X_data = None
if 'X' in data_struct.dtype.names:
    X_data = data_struct['X'][0]  # Extract the first element which contains the array
    print(f"Number of trials/chunks: {len(X_data)}")
    
    for i, trial_data in enumerate(X_data):
        print(f"\nTrial {i+1}:")
        print(f"  Shape: {trial_data.shape}")  # (time_points, channels)
        print(f"  Data type: {trial_data.dtype}")
        print(f"  Time points: {trial_data.shape[0]}")
        print(f"  Channels: {trial_data.shape[1]}")
        
        # Show sample data
        print(f"  First 3 time points, first 5 channels:")
        print("  " + "-" * 40)
        for j in range(min(3, trial_data.shape[0])):
            channels_sample = trial_data[j, :5]  # First 5 channels
            print(f"  Time {j+1}: {channels_sample}")
        
        print(f"  Data range: {np.min(trial_data):.2f} to {np.max(trial_data):.2f}")
        print(f"  Mean: {np.mean(trial_data):.2f}, Std: {np.std(trial_data):.2f}")

# 2. CLASSES Information
print("\n" + "="*50)
print("2. CLASS LABELS INFORMATION")
print("="*50)

if 'classes' in data_struct.dtype.names:
    classes_data = data_struct['classes'][0,0]  # Extract the nested array
    print("Available classes:")
    for i, class_name in enumerate(classes_data):
        print(f"  Class {i+1}: {class_name[0]}")
    
    # Since 'y' is empty, we need to understand how trials map to classes
    print(f"\nTotal classes: {len(classes_data)}")
    
    # Additional class information
    print(f"Class names: {[class_name[0] for class_name in classes_data]}")

# 3. SAMPLING RATE - FIXED VERSION
print("\n" + "="*50)
print("3. SAMPLING INFORMATION")
print("="*50)

if 'fs' in data_struct.dtype.names:
    fs_data = data_struct['fs'][0,0]  # Sampling frequency
    
    # Extract the actual numeric value from the array
    if isinstance(fs_data, np.ndarray):
        if fs_data.size > 0:
            fs_value = fs_data.item() if fs_data.size == 1 else fs_data[0,0]
        else:
            fs_value = 250  # Default value if empty
    else:
        fs_value = fs_data
    
    print(f"Sampling rate: {fs_value} Hz")
    
    if X_data is not None and len(X_data) > 0:
        # Ensure we have a numeric value for calculation
        trial_duration = X_data[0].shape[0] / float(fs_value)
        print(f"Trial duration: {trial_duration:.2f} seconds")
        print(f"Time points per trial: {X_data[0].shape[0]}")
        print(f"Channels per trial: {X_data[0].shape[1]}")
        
        # Calculate total recording time
        total_time_points = sum(trial.shape[0] for trial in X_data)
        total_recording_time = total_time_points / float(fs_value)
        print(f"Total recording time: {total_recording_time:.2f} seconds ({total_recording_time/60:.2f} minutes)")

# 4. TRIAL INFORMATION
print("\n" + "="*50)
print("4. TRIAL INFORMATION")
print("="*50)

if 'trial' in data_struct.dtype.names:
    trial_info = data_struct['trial'][0]
    if trial_info.size > 0:
        print(f"Trial info: {trial_info}")
        print(f"Number of trials: {len(trial_info)}")
    else:
        print("Trial information: Empty (may be implied by X data structure)")
        if X_data is not None:
            print(f"Number of trials inferred from X data: {len(X_data)}")
            
            # Create artificial trial numbers since the field is empty
            artificial_trials = list(range(1, len(X_data) + 1))
            print(f"Artificial trial numbering: {artificial_trials}")

# 5. SUBJECT INFORMATION
print("\n" + "="*50)
print("5. SUBJECT METADATA")
print("="*50)

if 'gender' in data_struct.dtype.names:
    gender_data = data_struct['gender'][0]
    if gender_data.size > 0:
        # Extract string from array
        if isinstance(gender_data[0], np.ndarray):
            gender = gender_data[0][0] if gender_data[0].size > 0 else "Not specified"
        else:
            gender = gender_data[0]
        print(f"Gender: {gender}")
    else:
        print("Gender: Not specified")
else:
    print("Gender: Field not available")

if 'age' in data_struct.dtype.names:
    age_data = data_struct['age'][0,0]
    # Extract numeric value from array
    if isinstance(age_data, np.ndarray):
        age = age_data.item() if age_data.size == 1 else age_data[0,0]
    else:
        age = age_data
    print(f"Age: {age} years")
else:
    print("Age: Field not available")

# 6. ARTIFACTS (if any)
print("\n" + "="*50)
print("6. ARTIFACT INFORMATION")
print("="*50)

if 'artifacts' in data_struct.dtype.names:
    artifacts = data_struct['artifacts'][0]
    if artifacts.size > 0:
        print(f"Artifacts detected: {artifacts}")
        print(f"Number of artifacts: {len(artifacts)}")
        print(f"Artifact indices: {artifacts.flatten()}")
    else:
        print("Artifacts: None detected or empty")
        print("✅ Data appears to be clean of marked artifacts")
else:
    print("Artifacts: Field not available in dataset")

# 7. LABELS INFORMATION (y field)
print("\n" + "="*50)
print("7. LABELS (y FIELD) INFORMATION")
print("="*50)

if 'y' in data_struct.dtype.names:
    y_data = data_struct['y'][0]
    if y_data.size > 0:
        print(f"Labels shape: {y_data.shape}")
        print(f"Labels content: {y_data}")
        print(f"Unique labels: {np.unique(y_data)}")
    else:
        print("Labels field (y) is empty")
        print("ℹ️  Labels might be encoded in the trial structure or separate file")
        print("ℹ️  Using class information for reference")
else:
    print("Labels field (y) not found in dataset")

# 8. DATASET SUMMARY
print("\n" + "="*50)
print("8. DATASET SUMMARY")
print("="*50)

print("📊 COMPREHENSIVE DATASET SUMMARY:")
print("-" * 40)

# Calculate overall statistics
if X_data is not None and len(X_data) > 0:
    all_trials_combined = np.vstack(X_data)  # Combine all trials
    
    # Get sampling rate properly
    fs_value = 250  # default
    if 'fs' in data_struct.dtype.names:
        fs_data = data_struct['fs'][0,0]
        if isinstance(fs_data, np.ndarray) and fs_data.size > 0:
            fs_value = fs_data.item() if fs_data.size == 1 else fs_data[0,0]
    
    print(f"• Total trials: {len(X_data)}")
    print(f"• Total time points: {all_trials_combined.shape[0]:,}")
    print(f"• Number of channels: {all_trials_combined.shape[1]}")
    print(f"• Total recording time: {all_trials_combined.shape[0]/float(fs_value):.2f} seconds")
    print(f"• Data shape (all trials combined): {all_trials_combined.shape}")
    print(f"• Global data range: {np.min(all_trials_combined):.2f} to {np.max(all_trials_combined):.2f} μV")
    print(f"• Global mean: {np.mean(all_trials_combined):.4f} μV")
    print(f"• Global standard deviation: {np.std(all_trials_combined):.4f} μV")
    
    # Channel statistics
    print(f"\n📈 CHANNEL STATISTICS:")
    print("-" * 25)
    for i in range(min(5, all_trials_combined.shape[1])):  # Show first 5 channels
        channel_data = all_trials_combined[:, i]
        print(f"Channel {i+1}: Mean={np.mean(channel_data):7.2f}, "
              f"Std={np.std(channel_data):6.2f}, "
              f"Range={np.ptp(channel_data):7.2f}")

# 9. DATA QUALITY CHECK
print("\n" + "="*50)
print("9. DATA QUALITY ASSESSMENT")
print("="*50)

if X_data is not None and len(X_data) > 0:
    print("✅ DATA QUALITY CHECKS:")
    print("-" * 25)
    
    # Check for NaN values
    has_nan = any(np.isnan(trial).any() for trial in X_data)
    print(f"• NaN values present: {'❌ YES' if has_nan else '✅ NO'}")
    
    # Check for infinite values
    has_inf = any(np.isinf(trial).any() for trial in X_data)
    print(f"• Infinite values present: {'❌ YES' if has_inf else '✅ NO'}")
    
    # Check data consistency across trials
    trial_shapes = [trial.shape for trial in X_data]
    consistent_shapes = all(shape == trial_shapes[0] for shape in trial_shapes)
    print(f"• Consistent trial shapes: {'✅ YES' if consistent_shapes else '❌ NO'}")
    
    if consistent_shapes:
        print(f"  All trials have shape: {trial_shapes[0]}")
    else:
        print(f"  Varied trial shapes: {set(trial_shapes)}")
    
    # Check signal amplitude ranges
    reasonable_range = all(np.max(np.abs(trial)) < 1000 for trial in X_data)  # Assuming μV scale
    print(f"• Reasonable amplitude range: {'✅ YES' if reasonable_range else '❌ NO'}")

# 10. RECOMMENDATIONS FOR ANALYSIS
print("\n" + "="*50)
print("10. ANALYSIS RECOMMENDATIONS")
print("="*50)

print("🎯 RECOMMENDED NEXT STEPS:")
print("-" * 30)
print("1. Preprocessing: Apply bandpass filter (e.g., 0.5-40 Hz)")
print("2. Feature Extraction: Time-domain, frequency-domain features")
print("3. Classification: Use class information for supervised learning")
print("4. Cross-validation: Stratified k-fold due to potential class imbalance")
print("5. Visualization: Time-series plots, spectrograms, topomaps")

print("\n⚠️  IMPORTANT NOTES:")
print("-" * 20)
print("• Labels (y field) are empty - need alternative labeling strategy")
print("• Use 'classes' field as reference for motor imagery tasks")
print("• Consider trial structure for event-related potential analysis")
print("• Verify channel locations for spatial analysis")

print("\n" + "="*70)
print("ANALYSIS COMPLETE - DATASET READY FOR PROCESSING")
print("="*70)