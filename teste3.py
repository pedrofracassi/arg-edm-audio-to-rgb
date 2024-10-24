import pandas as pd
import numpy as np


def scale_to_byte(values):
    """
    Scale an array of values to 0-255 range.

    Args:
        values (array-like): Input values to scale

    Returns:
        np.array: Scaled values as integers between 0-255
    """
    min_val = np.min(values)
    max_val = np.max(values)

    # Handle edge case where all values are the same
    if min_val == max_val:
        return np.full_like(values, 128, dtype=np.uint8)

    # Scale to 0-255 range
    scaled = ((values - min_val) / (max_val - min_val) * 255).round().astype(np.uint8)
    return scaled


def process_frequency_csv(input_path, output_path):
    """
    Read frequency CSV and create new CSV with scaled values.

    Args:
        input_path (str): Path to input CSV file
        output_path (str): Path to save output CSV file
    """
    # Read the CSV
    df = pd.read_csv(input_path)

    # Create a new dataframe for scaled values
    scaled_df = pd.DataFrame()
    scaled_df["Time (s)"] = df["Time (s)"]

    # Scale each frequency column
    for col in df.columns:
        if "Frequency" in col:
            scaled_values = scale_to_byte(df[col].values)
            new_col_name = col.replace("Hz", "scaled")
            scaled_df[new_col_name] = scaled_values

            # Print scaling information
            print(f"\nScaling info for {col}:")
            print(f"Original range: {df[col].min():.1f} to {df[col].max():.1f}")
            print(f"Scaled range: {scaled_values.min()} to {scaled_values.max()}")

    # Save to new CSV
    scaled_df.to_csv(output_path, index=False)
    print(f"\nScaled data saved to: {output_path}")

    return scaled_df


# Example usage
if __name__ == "__main__":
    input_file = "frequency_results.csv"
    output_file = "frequency_data_scaled.csv"

    scaled_data = process_frequency_csv(input_file, output_file)
    print("\nFirst few rows of scaled data:")
    print(scaled_data)
