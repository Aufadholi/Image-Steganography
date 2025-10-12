"""
Utility functions for loading/saving images, payloads, and trace matrix operations
"""

import cv2
import numpy as np
import json
import pickle
from PIL import Image
import os
from pathlib import Path


def load_image(image_path):
    """
    Load an image from file
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Loaded image as numpy array
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Try loading with OpenCV first
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        # Fallback to PIL
        try:
            pil_image = Image.open(image_path)
            image = np.array(pil_image)
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Convert RGB to BGR for OpenCV compatibility
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise ValueError(f"Could not load image {image_path}: {str(e)}")
    
    return image


def save_image(image, output_path):
    """
    Save an image to file
    
    Args:
        image: Image as numpy array
        output_path: Path where to save the image
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save using OpenCV
    success = cv2.imwrite(output_path, image)
    if not success:
        raise ValueError(f"Could not save image to {output_path}")


def load_text_payload(payload_path):
    """
    Load text payload from file
    
    Args:
        payload_path: Path to the text file
    
    Returns:
        Text content as string
    """
    if not os.path.exists(payload_path):
        raise FileNotFoundError(f"Payload file not found: {payload_path}")
    
    with open(payload_path, 'r', encoding='utf-8') as file:
        return file.read()


def save_text_payload(text, output_path):
    """
    Save text payload to file
    
    Args:
        text: Text content to save
        output_path: Path where to save the text
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(text)


def load_image_payload(payload_path):
    """
    Load image payload from file
    
    Args:
        payload_path: Path to the image file
    
    Returns:
        Image as numpy array
    """
    return load_image(payload_path)


def save_image_payload(image, output_path):
    """
    Save image payload to file
    
    Args:
        image: Image as numpy array
        output_path: Path where to save the image
    """
    save_image(image, output_path)


def text_to_binary(text):
    """
    Convert text to binary representation
    
    Args:
        text: Input text string
    
    Returns:
        Binary string representation
    """
    binary = ''.join(format(ord(char), '08b') for char in text)
    return binary


def binary_to_text(binary):
    """
    Convert binary representation back to text
    
    Args:
        binary: Binary string
    
    Returns:
        Text string
    """
    # Ensure binary length is multiple of 8
    if len(binary) % 8 != 0:
        binary = binary[:-(len(binary) % 8)]
    
    text = ''
    for i in range(0, len(binary), 8):
        byte = binary[i:i+8]
        if len(byte) == 8:
            text += chr(int(byte, 2))
    
    return text


def image_to_binary(image):
    """
    Convert image to binary representation
    
    Args:
        image: Image as numpy array
    
    Returns:
        Binary string representation and image metadata
    """
    # Store image metadata
    metadata = {
        'shape': image.shape,
        'dtype': str(image.dtype)
    }
    
    # Flatten image and convert to binary
    flat_image = image.flatten()
    binary = ''.join(format(pixel, '08b') for pixel in flat_image)
    
    return binary, metadata


def binary_to_image(binary, metadata):
    """
    Convert binary representation back to image
    
    Args:
        binary: Binary string
        metadata: Image metadata dictionary
    
    Returns:
        Reconstructed image as numpy array
    """
    # Calculate expected binary length
    shape = metadata['shape']
    total_pixels = np.prod(shape)
    expected_length = total_pixels * 8
    
    # Truncate or pad binary if necessary
    if len(binary) > expected_length:
        binary = binary[:expected_length]
    elif len(binary) < expected_length:
        binary = binary.ljust(expected_length, '0')
    
    # Convert binary to pixel values
    pixels = []
    for i in range(0, len(binary), 8):
        byte = binary[i:i+8]
        if len(byte) == 8:
            pixels.append(int(byte, 2))
    
    # Reshape to original image shape
    image = np.array(pixels, dtype=metadata['dtype']).reshape(shape)
    
    return image


def save_trace_matrix(trace_matrix, output_path):
    """
    Save trace matrix to file for reversibility
    
    Args:
        trace_matrix: Dictionary containing trace information
        output_path: Path where to save the trace matrix
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as file:
        pickle.dump(trace_matrix, file)


def load_trace_matrix(trace_path):
    """
    Load trace matrix from file
    
    Args:
        trace_path: Path to the trace matrix file
    
    Returns:
        Trace matrix dictionary
    """
    if not os.path.exists(trace_path):
        raise FileNotFoundError(f"Trace matrix file not found: {trace_path}")
    
    with open(trace_path, 'rb') as file:
        return pickle.load(file)


def create_trace_matrix(image_shape, embedding_coords, original_lsbs):
    """
    Create trace matrix for reversibility
    
    Args:
        image_shape: Shape of the original image
        embedding_coords: List of (y, x) coordinates where embedding occurred
        original_lsbs: List of original LSB values
    
    Returns:
        Trace matrix dictionary
    """
    trace_matrix = {
        'image_shape': image_shape,
        'embedding_coords': embedding_coords,
        'original_lsbs': original_lsbs,
        'num_embedded_pixels': len(embedding_coords)
    }
    
    return trace_matrix


def prepare_payload(payload_path, payload_type='auto'):
    """
    Prepare payload for embedding (detect type and convert to binary)
    
    Args:
        payload_path: Path to payload file
        payload_type: Type of payload ('text', 'image', 'auto')
    
    Returns:
        Binary representation of payload and metadata
    """
    if payload_type == 'auto':
        # Auto-detect payload type based on file extension
        extension = Path(payload_path).suffix.lower()
        if extension in ['.txt', '.md', '.json']:
            payload_type = 'text'
        elif extension in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            payload_type = 'image'
        else:
            raise ValueError(f"Cannot auto-detect payload type for extension: {extension}")
    
    metadata = {'type': payload_type}
    
    if payload_type == 'text':
        text = load_text_payload(payload_path)
        binary = text_to_binary(text)
        metadata['length'] = len(text)
    elif payload_type == 'image':
        image = load_image_payload(payload_path)
        binary, img_metadata = image_to_binary(image)
        metadata.update(img_metadata)
    else:
        raise ValueError(f"Unsupported payload type: {payload_type}")
    
    return binary, metadata


def extract_payload_from_binary(binary, metadata, output_path):
    """
    Extract payload from binary and save to file
    
    Args:
        binary: Binary representation of payload
        metadata: Payload metadata
        output_path: Path where to save the extracted payload
    """
    payload_type = metadata['type']
    
    if payload_type == 'text':
        text = binary_to_text(binary)
        save_text_payload(text, output_path)
    elif payload_type == 'image':
        image = binary_to_image(binary, metadata)
        save_image_payload(image, output_path)
    else:
        raise ValueError(f"Unsupported payload type: {payload_type}")


def add_payload_metadata_header(binary, metadata):
    """
    Add metadata header to payload binary for self-describing payloads
    
    Args:
        binary: Payload binary
        metadata: Payload metadata
    
    Returns:
        Binary with metadata header
    """
    # Convert metadata to JSON and then to binary
    metadata_json = json.dumps(metadata)
    metadata_binary = text_to_binary(metadata_json)
    
    # Add header length (32 bits for metadata length)
    header_length = format(len(metadata_binary), '032b')
    
    # Combine: header_length + metadata + payload
    full_binary = header_length + metadata_binary + binary
    
    return full_binary


def extract_payload_metadata_header(binary):
    """
    Extract metadata header from payload binary
    
    Args:
        binary: Binary with metadata header
    
    Returns:
        Payload binary (without header) and metadata
    """
    # Extract header length (first 32 bits)
    header_length_binary = binary[:32]
    header_length = int(header_length_binary, 2)
    
    # Extract metadata
    metadata_binary = binary[32:32 + header_length]
    metadata_json = binary_to_text(metadata_binary)
    metadata = json.loads(metadata_json)
    
    # Extract payload binary
    payload_binary_full = binary[32 + header_length:]
    
    # Use the actual payload length from metadata if available
    if 'length' in metadata and metadata['type'] == 'text':
        # For text, calculate the correct number of bits needed
        expected_bits = metadata['length'] * 8  # 8 bits per character
        payload_binary = payload_binary_full[:expected_bits]
    else:
        # For other types or when length is not available, use full binary
        payload_binary = payload_binary_full
    
    return payload_binary, metadata


def calculate_embedding_capacity(image_shape, embedding_coords):
    """
    Calculate embedding capacity based on selected pixels
    
    Args:
        image_shape: Shape of the cover image
        embedding_coords: List of embedding coordinates
    
    Returns:
        Capacity statistics dictionary
    """
    total_pixels = image_shape[0] * image_shape[1]
    embedding_pixels = len(embedding_coords)
    
    capacity = {
        'total_pixels': total_pixels,
        'embedding_pixels': embedding_pixels,
        'capacity_ratio': embedding_pixels / total_pixels,
        'max_bits': embedding_pixels,  # 1 bit per pixel for LSB
        'max_bytes': embedding_pixels // 8
    }
    
    return capacity


def validate_payload_size(payload_binary, capacity):
    """
    Validate if payload can fit in the available capacity
    
    Args:
        payload_binary: Binary representation of payload
        capacity: Capacity dictionary from calculate_embedding_capacity
    
    Returns:
        True if payload fits, False otherwise
    """
    payload_bits = len(payload_binary)
    return payload_bits <= capacity['max_bits']


def get_file_info(file_path):
    """
    Get file information including size and type
    
    Args:
        file_path: Path to the file
    
    Returns:
        File information dictionary
    """
    if not os.path.exists(file_path):
        return None
    
    stat = os.stat(file_path)
    extension = Path(file_path).suffix.lower()
    
    info = {
        'path': file_path,
        'size_bytes': stat.st_size,
        'extension': extension,
        'modified_time': stat.st_mtime
    }
    
    return info


def create_sample_payloads(output_dir):
    """
    Create sample payload files for testing
    
    Args:
        output_dir: Directory where to create sample files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create sample text file
    sample_text = "hello"
    text_path = os.path.join(output_dir, "sample_text.txt")
    save_text_payload(sample_text, text_path)
    
    # Create sample small image (16x16 checkerboard for smaller payload)
    sample_image = np.zeros((16, 16), dtype=np.uint8)
    for i in range(16):
        for j in range(16):
            if (i // 4 + j // 4) % 2 == 0:
                sample_image[i, j] = 255
    
    image_path = os.path.join(output_dir, "sample_image.png")
    save_image(sample_image, image_path)
    
    return text_path, image_path