import pandas as pd
import numpy as np
from PIL import Image
import itertools
import os


def create_multi_channel_images(
    input_csv, width, height, output_dir="channel_arrangements"
):
    """
    Create images with all possible RGB channel arrangements from frequency data.

    Args:
        input_csv (str): Path to input CSV with frequency data
        width (int): Desired width of the image
        height (int): Desired height of the image
        output_dir (str): Directory to save output images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read and scale the data
    df = pd.read_csv(input_csv)
    num_points = len(df)
    required_pixels = width * height

    print(f"Input data points: {num_points}")
    print(f"Required pixels: {required_pixels}")
    print(f"Resolution: {width}x{height}")

    if num_points < required_pixels:
        print(f"Warning: Not enough data points to fill image")
        print(f"Missing pixels will be black")
    elif num_points > required_pixels:
        print(f"Warning: Only first {required_pixels} data points will be used")

    # Scale the frequency data
    scaled_values = []
    freq_columns = [col for col in df.columns if "Frequency" in col]

    for col in freq_columns:
        values = df[col].values
        min_val = values.min()
        max_val = values.max()
        if min_val == max_val:
            scaled = np.full_like(values, 128, dtype=np.uint8)
        else:
            scaled = (
                ((values - min_val) / (max_val - min_val) * 255)
                .round()
                .astype(np.uint8)
            )
        scaled_values.append(scaled)

    # Get all possible channel arrangements
    channels = ["R", "G", "B"]
    arrangements = list(itertools.permutations(range(len(scaled_values))))

    # Create base image array
    points_to_use = min(num_points, width * height)

    # Generate an image for each arrangement
    results = []
    for arrangement in arrangements:
        # Create descriptive name for this arrangement
        channel_names = [channels[i] for i in arrangement]
        arrangement_name = "".join(channel_names)

        # Create image array
        img_array = np.zeros((height, width, 3), dtype=np.uint8)

        # Fill the image array
        for i in range(points_to_use):
            row = i // width
            col = i % width

            # Arrange the channels according to current permutation
            pixel_values = [scaled_values[channel][i] for channel in arrangement]
            img_array[row, col] = pixel_values

        # Create and save image
        img = Image.fromarray(img_array)

        # Scale up for better visibility
        max_dimension = 1000
        scale_factor = min(50, max_dimension // max(width, height))
        scaled_size = (width * scale_factor, height * scale_factor)
        img = img.resize(scaled_size, Image.Resampling.NEAREST)

        # Save image
        output_path = os.path.join(
            output_dir, f"frequency_image_{width}x{height}_{arrangement_name}.png"
        )
        img.save(output_path)

        result = {
            "arrangement": arrangement_name,
            "mapping": {
                channels[i]: freq_columns[orig_idx]
                for i, orig_idx in enumerate(arrangement)
            },
            "path": output_path,
        }
        results.append(result)

        print(f"\nGenerated {arrangement_name} arrangement:")
        print(f"Channel mapping:")
        for channel, freq in result["mapping"].items():
            print(f"  {channel}: {freq}")
        print(f"Saved to: {output_path}")

    # Print summary
    print("\nGenerated all possible channel arrangements:")
    print(f"Total arrangements: {len(results)}")
    print("\nSummary of mappings:")
    for result in results:
        print(f"\n{result['arrangement']}:")
        for channel, freq in result["mapping"].items():
            print(f"  {channel} <- {freq}")

    return results


# Example usage
if __name__ == "__main__":
    input_file = "frequency_data_scaled.csv"
    width = 184
    height = 165
    output_directory = "channel_arrangements"

    results = create_multi_channel_images(
        input_csv=input_file, width=width, height=height, output_dir=output_directory
    )
