import pandas as pd
import numpy as np
from PIL import Image
import math
import os


def create_square_image(scaled_data, square_size, output_path):
    """
    Create a square image of specified size from frequency data.

    Args:
        scaled_data (pd.DataFrame): DataFrame with scaled frequency data
        square_size (int): Size of square to generate
        output_path (str): Path to save the output image
    """
    # Create image array
    img_array = np.zeros((square_size, square_size, 3), dtype=np.uint8)

    # Get the scaled frequency columns
    freq_columns = [col for col in scaled_data.columns if "scaled" in col]
    num_points = min(len(scaled_data), square_size * square_size)

    # Fill the image array
    for i in range(num_points):
        row = i // square_size
        col = i % square_size
        rgb_values = scaled_data.iloc[i][freq_columns].values
        img_array[row, col] = rgb_values

    # Create and save image
    img = Image.fromarray(img_array)
    # Scale up for better visibility (50x)
    img = img.resize((square_size * 50, square_size * 50), Image.Resampling.NEAREST)
    img.save(output_path)

    return img_array


def generate_all_squares(input_csv, output_dir="square_images"):
    """
    Generate all possible square images from 1x1 up to the optimal size.

    Args:
        input_csv (str): Path to input CSV with frequency data
        output_dir (str): Directory to save output images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read and scale the data
    df = pd.read_csv(input_csv)
    num_points = len(df)
    max_square_size = math.ceil(math.sqrt(num_points))

    print(f"Number of data points: {num_points}")
    print(f"Maximum square size: {max_square_size}x{max_square_size}")

    # Scale the frequency data
    scaled_df = pd.DataFrame()
    scaled_df["Time (s)"] = df["Time (s)"]

    for col in df.columns:
        if "Frequency" in col:
            min_val = df[col].min()
            max_val = df[col].max()
            if min_val == max_val:
                scaled_values = np.full_like(df[col], 128, dtype=np.uint8)
            else:
                scaled_values = (
                    ((df[col] - min_val) / (max_val - min_val) * 255)
                    .round()
                    .astype(np.uint8)
                )
            new_col_name = col.replace("Hz", "scaled")
            scaled_df[new_col_name] = scaled_values

    # Generate squares of all sizes
    results = []
    for size in range(1, max_square_size + 1):
        output_path = os.path.join(output_dir, f"square_{size}x{size}.png")
        img_array = create_square_image(scaled_df, size, output_path)

        # Calculate how much of the data is represented
        points_used = min(num_points, size * size)
        data_coverage = (points_used / num_points) * 100

        result = {
            "size": size,
            "points_used": points_used,
            "total_points": num_points,
            "data_coverage": data_coverage,
            "image_path": output_path,
        }
        results.append(result)

        print(f"\nGenerated {size}x{size} square:")
        print(f"Points used: {points_used}/{num_points}")
        print(f"Data coverage: {data_coverage:.1f}%")
        print(f"Saved to: {output_path}")

    return results, scaled_df


# Example usage
if __name__ == "__main__":
    input_file = "frequency_data_scaled.csv"
    output_directory = "square_images"

    results, scaled_data = generate_all_squares(input_file, output_directory)

    print("\nScaled frequency data:")
    print(scaled_data)

    print("\nSummary of generated images:")
    for result in results:
        print(f"\n{result['size']}x{result['size']} square:")
        print(f"Data coverage: {result['data_coverage']:.1f}%")
