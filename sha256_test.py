import hashlib
import time

def calculate_sha256_with_library(data):
    """
    Calculates the SHA-256 hash of the given data using Python's hashlib library.

    Args:
        data (bytes or str): The data to hash. If it's a string, it will be encoded to UTF-8.

    Returns:
        str: The hexadecimal representation of the SHA-256 hash.
    """
    try:
        # Create a SHA-256 hash object
        sha256_hash = hashlib.sha256()

        # Update the hash object with the data
        if isinstance(data, str):
            sha256_hash.update(data.encode('utf-8'))
        elif isinstance(data, bytes):
            sha256_hash.update(data)
        else:
            raise TypeError("Input data must be a string or bytes.")

        # Get the hexadecimal representation of the hash digest
        hex_digest = sha256_hash.hexdigest()
        return hex_digest

    except TypeError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

target_hash_prefix = "0" * 7

start_time = time.time()  # Record start time
found = False  # Flag to track if the hash is found

for i in range(62000000000, 63000000000):
    hashed_value = calculate_sha256_with_library("GeorgeW" + str(i))
    if hashed_value.startswith(target_hash_prefix):
        end_time = time.time()  # Record end time
        print(f"Found match: 'GeorgeW{i}' hashes to {hashed_value}")
        print(f"Time taken: {end_time - start_time:.4f} seconds")
        found = True
        break  # Exit the loop once a match is found

if not found:
    print("No matching hash found within the specified range.")