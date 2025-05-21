import hashlib

def calculate_sha256_with_library(data, nonce):
    """
    Calculate a SHA-256 hash by combining input data and a nonce.
    Includes print statements to show the nonce bytes and their conversion back to int.
    
    Args:
        data (str or bytes): The input data
        nonce (int): The nonce value
        
    Returns:
        str: The hexadecimal hash digest
    """
    # Convert data to bytes if necessary
    if isinstance(data, str):
        data_bytes = data.encode('utf-8')
    elif isinstance(data, bytes):
        data_bytes = data
    else:
        raise TypeError("Input must be string or bytes")
    
    # Append nonce bytes (big-endian)
    nonce_bytes = nonce.to_bytes(4, byteorder='big')
    message = data_bytes + nonce_bytes

    # --- New print statements for demonstration ---
    print(f"Original nonce (int): {nonce}")
    print(f"Nonce as bytes: {nonce_bytes}")

    # Convert nonce_bytes back to an integer using int.from_bytes()
    # It's crucial to specify the correct byteorder ('big' in this case)
    recovered_nonce = int.from_bytes(nonce_bytes, byteorder='big')
    print(f"Recovered nonce from bytes (int): {recovered_nonce}")
    # --- End new print statements ---
    
    # Calculate hash using hashlib
    sha256 = hashlib.sha256()
    sha256.update(message)
    return sha256.hexdigest()

        
print("\n--- Running the hash calculation ---")
print(calculate_sha256_with_library(data="GeorgeW", nonce=3136365110))