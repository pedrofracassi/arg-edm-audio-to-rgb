import pandas as pd
import numpy as np
from PIL import Image
import math
import os


def create_image(scaled_data, width, height, output_path):
    """
    Create an image with specified dimensions from frequency data.

    Args:
        scaled_data (pd.DataFrame): DataFrame with scaled frequency data
        width (int): Width of the image
        height (int): Height of the image
        output_path (str): Path to save the output image
    """
    # Create image array
    img_array = np.zeros((height, width, 3), dtype=np.uint8)

    # Get the scaled frequency columns
    freq_columns = [col for col in scaled_data.columns if "scaled" in col]
    num_points = min(len(scaled_data), width * height)

    # Fill the image array
    for i in range(num_points):
        row = i // width
        col = i % width
        rgb_values = scaled_data.iloc[i][freq_columns].values
        img_array[row, col] = rgb_values

    # Create and save image
    img = Image.fromarray(img_array)
    # Scale up for better visibility (50x)
    img = img.resize((width * 50, height * 50), Image.Resampling.NEAREST)
    img.save(output_path)

    return img_array


def get_all_factors(n):
    """Get all factors of a number."""
    factors = []
    for i in range(1, n + 1):
        if n % i == 0:
            factors.append(i)
    return factors


def generate_all_resolutions(input_csv, output_dir="resolution_images"):
    """
    Generate images of all possible resolutions that can contain the data points.

    Args:
        input_csv (str): Path to input CSV with frequency data
        output_dir (str): Directory to save output images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read and scale the data
    df = pd.read_csv(input_csv)
    num_points = len(df)

    print(f"Number of data points: {num_points}")

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

    # Find all possible dimensions
    results = []

    # Generate all possible width-height combinations
    for total_pixels in range(num_points, num_points + 1):  # We only want exact fits
        factors = get_all_factors(total_pixels)
        for width in factors:
            height = total_pixels // width
            output_path = os.path.join(output_dir, f"resolution_{width}x{height}.png")

            # Generate image
            img_array = create_image(scaled_df, width, height, output_path)

            # Calculate data coverage
            points_used = min(num_points, width * height)
            data_coverage = (points_used / num_points) * 100

            aspect_ratio = width / height

            result = {
                "width": width,
                "height": height,
                "points_used": points_used,
                "total_points": num_points,
                "data_coverage": data_coverage,
                "aspect_ratio": aspect_ratio,
                "image_path": output_path,
            }
            results.append(result)

            print(f"\nGenerated {width}x{height} image:")
            print(f"Aspect ratio: {aspect_ratio:.2f}")
            print(f"Points used: {points_used}/{num_points}")
            print(f"Data coverage: {data_coverage:.1f}%")
            print(f"Saved to: {output_path}")

            # Also generate the transposed version (height x width)
            if width != height:  # Skip if it's a square to avoid duplicates
                output_path = os.path.join(
                    output_dir, f"resolution_{height}x{width}.png"
                )
                img_array = create_image(scaled_df, height, width, output_path)

                result = {
                    "width": height,
                    "height": width,
                    "points_used": points_used,
                    "total_points": num_points,
                    "data_coverage": data_coverage,
                    "aspect_ratio": height / width,
                    "image_path": output_path,
                }
                results.append(result)

                print(f"\nGenerated {height}x{width} image:")
                print(f"Aspect ratio: {(height / width):.2f}")
                print(f"Points used: {points_used}/{num_points}")
                print(f"Data coverage: {data_coverage:.1f}%")
                print(f"Saved to: {output_path}")

    # Sort results by aspect ratio
    results.sort(key=lambda x: x["aspect_ratio"])

    # Print summary
    print("\nSummary of all generated resolutions:")
    print("Resolution | Aspect Ratio | Data Coverage")
    print("-" * 45)
    for result in results:
        print(
            f"{result['width']}x{result['height']} | {result['aspect_ratio']:.2f} | {result['data_coverage']:.1f}%"
        )

    return results, scaled_df


# Example usage
if __name__ == "__main__":
    input_file = "frequency_data_scaled.csv"
    output_directory = "resolution_images"

    results, scaled_data = generate_all_resolutions(input_file, output_directory)
