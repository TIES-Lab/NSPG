import numpy as np
import pandas as pd

# Read the CSV file
df = pd.read_csv('train_dataset.csv')

# Convert data type to boolean for the first column
df['property'] = df['property'].astype(bool)

# Create a new dataframe
df_new = pd.DataFrame()

# Create a new column in the new dataframe
df_new['text'] = df['sentence']

# Loop through the rows of the  original dataframe
for index, row in df.iterrows():
    # Converting each sentence to a specific format for training the model
    if row['property'] is False:
        df_new.at[index, 'text'] = "You are an AI programming assistant, utilizing the falcon model, developed by TII, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer. \n ### Instruction: Determine whether the following sentence defines a property or non-property type sentence for a design documentation:\n" + row['sentence'] + "\n ### Response: \n This is a non-property type sentence for a design documentation. \n[EOT]"
    else: 
       df_new.at[index, 'text'] = "You are an AI programming assistant, utilizing the falcon model, developed by TII, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer. \n ### Instruction: Determine whether the following sentence defines a property or non-property type sentence for a design documentation:\n" + row['sentence'] + "\n ### Response: \n This is a property type sentence for a design documentation. \n[EOT]"

# Convert the new dataframe to a JSON file
df_new.to_json('train_dataset.json', orient='records', indent=4)