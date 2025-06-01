import pandas as pd
import os
import random

FILE_PATH = 'data/iris.csv' 

def generate_new_iris_rows(num_rows=130):
    """Generates a DataFrame with new synthetic Iris data rows (matching your CSV structure)."""
    new_data_list = []
    species_options = ['setosa-new', 'versicolor-new', 'virginica-new', 'hybrida-new']
    
    for _ in range(num_rows):
        row = {
            'sepal_length': round(random.uniform(4.0, 8.5), 1),
            'sepal_width': round(random.uniform(1.8, 4.8), 1),
            'petal_length': round(random.uniform(0.9, 7.5), 1),
            'petal_width': round(random.uniform(0.1, 2.8), 1),
            'species': random.choice(species_options)
        }
        new_data_list.append(row)
        
    # Define the column order based on your CSV (excluding the initial unnamed index column if it was there)
    column_order = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    return pd.DataFrame(new_data_list, columns=column_order)

def modify_iris_dataset_add_rows(path, num_rows_to_add=20):
    """Modifies the iris dataset by adding a specified number of new rows."""
    if not os.path.exists(path):
        print(f"ERROR: Data file {path} not found for modification!")
        return False

    df_original = pd.read_csv(path)
    print(f"Original dataset rows: {len(df_original)}")
    
    new_rows_df = generate_new_iris_rows(num_rows_to_add)

    # Concatenate; the original unnamed index column will have NaN for new rows if it exists
    df_modified = pd.concat([df_original, new_rows_df], ignore_index=True)

    # Save, index=False prevents pandas writing a new default index as a column
    df_modified.to_csv(path, index=False)
    
    print(f"Added {num_rows_to_add} new rows to {path}. New total rows: {len(df_modified)}")
    return True

if __name__ == "__main__":
    print(f"--- Modifying {FILE_PATH} ---")
    success = modify_iris_dataset_add_rows(FILE_PATH, num_rows_to_add=20)
    if success:
        print(f"--- Data Modification of {FILE_PATH} Complete ---")
    else:
        print(f"--- Data Modification of {FILE_PATH} Failed ---")