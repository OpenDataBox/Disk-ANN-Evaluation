import numpy as np
import struct
import os

def convert_bin_to_fbin_with_offset(original_bin_file, output_fbin_file, num_vectors, dimensions, header_offset_bytes):
    """
    Reads a raw binary file, skipping an existing header, prepends a new standard header,
    and saves as a new file.
    New Header format: [num_vectors (int32), dimensions (int32)]
    """
    if not os.path.exists(original_bin_file):
        print(f"Error: Input file not found at {original_bin_file}")
        return

    # Read the vector data, skipping the existing header by using the 'offset' parameter
    print(f"Reading raw data from {original_bin_file}, skipping first {header_offset_bytes} bytes...")
    try:
        vectors = np.fromfile(original_bin_file, dtype=np.float32, offset=header_offset_bytes)
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return
        
    # Check if the amount of data read is correct
    expected_elements = num_vectors * dimensions
    if vectors.size != expected_elements:
        print(f"Error: Data size mismatch after skipping header.")
        print(f"Expected {expected_elements} float elements, but found {vectors.size}.")
        print("Please verify the HEADER_OFFSET_BYTES, NUM_VECTORS, and DIMENSIONS parameters.")
        return

    # Reshape to ensure it matches expected dimensions
    vectors = vectors.reshape(num_vectors, dimensions)
    
    print(f"Data reshaped successfully to: {vectors.shape}")

    with open(output_fbin_file, 'wb') as f:
        # Write the NEW, standardized header
        print("Writing new standardized header...")
        # Write header: number of vectors (as a 4-byte integer)
        f.write(struct.pack('i', num_vectors))
        # Write header: dimensions (as a 4-byte integer)
        f.write(struct.pack('i', dimensions))
        # Write the vector data
        vectors.tofile(f)
        
    print(f"Successfully created {output_fbin_file} with a new, correct header.")

# --- 您需要在这里修改参数 ---
# GIST 1M 数据集的参数
NUM_VECTORS = 1000
DIMENSIONS = 960

# !!! 关键参数：假设文件头是2个4字节整数（向量数+维度），总共8字节
HEADER_OFFSET_BYTES = 8 

# 输入和输出文件名
INPUT_FILE = '/storage/vector_data/gist/gist_query.bin'
OUTPUT_FILE = '/home/chenxiaoyu/Gorgeous/tests_data/gist_query.fbin' # 新文件名

# 执行转换
convert_bin_to_fbin_with_offset(INPUT_FILE, OUTPUT_FILE, NUM_VECTORS, DIMENSIONS, HEADER_OFFSET_BYTES)
    
