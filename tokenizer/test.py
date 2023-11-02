import pickle

# Open the pickle file for reading
with open('../dataset/tokenized_files/toy_tokenized_data.pkl', 'rb') as file:
    # Load the data from the pickle file
    df = pickle.load(file)

# Now you can use the loaded_data in your code
print(df)
print(type(df))

print(df.iloc[1][1])
# Function to replace -1 with 32000
def replace_minus_one(val):
    return [32000 if x == -1 else x for x in val]

# Apply the function to the Tokenized_Data column
df['Tokenized_Data'] = df['Tokenized_Data'].apply(replace_minus_one)

print(df['Tokenized_Data'].iloc[2])
