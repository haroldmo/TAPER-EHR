import random
import pickle

# Function to balance the data based on gender
import random


def balance_data(data_samples):
    # Count the number of samples for each gender
    male_samples = {key: value for key, value in data_samples.items() if value[0]['demographics'][1] == 0}
    female_samples = {key: value for key, value in data_samples.items() if value[0]['demographics'][1] == 1}
    
    # Determine the minimum number of samples between the two genders
    min_samples = min(len(male_samples), len(female_samples))
    
    # Randomly select samples from each gender to match the minimum number of samples
    male_keys = random.sample(list(male_samples.keys()), min_samples)
    female_keys = random.sample(list(female_samples.keys()), min_samples)
    
    balanced_data = {}
    balanced_data.update({key: male_samples[key] for key in male_keys})
    balanced_data.update({key: female_samples[key] for key in female_keys})
    
    # Shuffle the balanced data
    balanced_data_shuffled = dict(random.sample(balanced_data.items(), len(balanced_data)))
    
    return balanced_data

# Load the data from the Pickle file
with open('/home/haroldmo/MIMIC_III/gen_df_text_outputs/data.pkl', 'rb') as f:
    data = pickle.load(f)

# Get the 'data' samples
data_samples = data['data']

# Balance the data samples
balanced_data = balance_data(data_samples)
# print(balanced_data)

# Print the types
print("Type of data_samples:", type(data_samples))
print("Type of balanced_data:", type(balanced_data))

male_count, female_count = 0, 0
for sample_id in balanced_data.keys():
    sample_data = balanced_data[sample_id]
    # print("Sample ID:", sample_id)
    # print("Sample data:", sample_data[0]['demographics'])
    if sample_data[0]['demographics'][1] == 0:
        male_count += 1
    elif sample_data[0]['demographics'][1] == 1:
        female_count += 1
    else:
        print("ERROR")
        break

# print(male_count, female_count)

# Access the 'data' key
data_samples = data['data']

# Print the number of samples
print("Number of samples:", len(data_samples))

# Inspect the structure of a sample
sample_id = next(iter(data_samples.keys()))  # Get the ID of the first sample
sample_data = data_samples[sample_id]
print("Sample ID:", sample_id)
print("Sample data:", sample_data)

# Explore more attributes or features of the sample as needed

data['data'] = balanced_data


# Access specific attributes or values
info = data['info']
data_data = data['data']

# Inspect the structure of a sample
sample_id = next(iter(data_samples.keys()))  # Get the ID of the first sample
sample_data = data_samples[sample_id]
print("Sample ID after:", sample_id)
print("Sample data after:", sample_data)

# Print or visualize information about the data
print("Info:", info)
print("Number of samples:", len(data_data))


output_path = '/home/haroldmo/MIMIC_III/gen_df_text_outputs_balanced_gender/data.pkl'

with open(output_path, 'wb') as f:
    pickle.dump(data, f)
# # Load the data from the Pickle file
# with open('/home/haroldmo/MIMIC_III/gen_df_text_outputs/data.pkl', 'rb') as f:
#     data = pickle.load(f)

# # Get the 'data' samples
# data_samples = data['data']

# # Iterate through all sample IDs
# for sample_id in data_samples.keys():
#     sample_data = data_samples[sample_id]
#     print("Sample ID:", sample_id)
#     print("Sample data:", sample_data[0]['demographics'])

