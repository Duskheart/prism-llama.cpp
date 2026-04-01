#!/usr/bin/env python3
"""
Patches GGUF file to remap Q1_0 type IDs after upstream added NVFP4.

No external dependencies required - uses only Python built-in modules.

Old prism code:  Q1_0 = 40, Q1_0_g128 = 41
New merged code: NVFP4 = 40, Q1_0 = 41, Q1_0_g128 = 42

This script remaps: 40 → 41, 41 → 42

Uses low-level GGUF parsing to avoid the block size mismatch issue.
"""

import struct
import shutil

# Type ID mapping (old -> new)
TYPE_REMAP = {
    40: 41,  # Q1_0: 40 -> 41
    41: 42,  # Q1_0_g128: 41 -> 42
}

input_file = "Bonsai-8B.gguf"
output_file = "Bonsai-8B_patched.gguf"

def read_string(f):
    """Read a GGUF string (length-prefixed)"""
    length = struct.unpack('<Q', f.read(8))[0]
    return f.read(length).decode('utf-8')

def patch_gguf(filename):
    """Patch tensor type IDs in a GGUF file"""
    
    with open(filename, 'r+b') as f:
        # Read header
        magic = f.read(4)
        if magic != b'GGUF':
            raise ValueError(f"Not a GGUF file: {magic}")
        
        version = struct.unpack('<I', f.read(4))[0]
        tensor_count = struct.unpack('<Q', f.read(8))[0]
        metadata_kv_count = struct.unpack('<Q', f.read(8))[0]
        
        print(f"GGUF version: {version}")
        print(f"Tensor count: {tensor_count}")
        print(f"Metadata KV count: {metadata_kv_count}")
        
        # Skip metadata key-value pairs
        for _ in range(metadata_kv_count):
            # Read key (string)
            key_len = struct.unpack('<Q', f.read(8))[0]
            f.read(key_len)  # skip key
            
            # Read value type
            value_type = struct.unpack('<I', f.read(4))[0]
            
            # Skip value based on type
            if value_type == 0:  # UINT8
                f.read(1)
            elif value_type == 1:  # INT8
                f.read(1)
            elif value_type == 2:  # UINT16
                f.read(2)
            elif value_type == 3:  # INT16
                f.read(2)
            elif value_type == 4:  # UINT32
                f.read(4)
            elif value_type == 5:  # INT32
                f.read(4)
            elif value_type == 6:  # FLOAT32
                f.read(4)
            elif value_type == 7:  # BOOL
                f.read(1)
            elif value_type == 8:  # STRING
                str_len = struct.unpack('<Q', f.read(8))[0]
                f.read(str_len)
            elif value_type == 9:  # ARRAY
                arr_type = struct.unpack('<I', f.read(4))[0]
                arr_len = struct.unpack('<Q', f.read(8))[0]
                # Skip array elements
                if arr_type == 0:  # UINT8
                    f.read(arr_len)
                elif arr_type == 1:  # INT8
                    f.read(arr_len)
                elif arr_type == 2:  # UINT16
                    f.read(arr_len * 2)
                elif arr_type == 3:  # INT16
                    f.read(arr_len * 2)
                elif arr_type == 4:  # UINT32
                    f.read(arr_len * 4)
                elif arr_type == 5:  # INT32
                    f.read(arr_len * 4)
                elif arr_type == 6:  # FLOAT32
                    f.read(arr_len * 4)
                elif arr_type == 7:  # BOOL
                    f.read(arr_len)
                elif arr_type == 8:  # STRING array
                    for _ in range(arr_len):
                        s_len = struct.unpack('<Q', f.read(8))[0]
                        f.read(s_len)
                else:
                    raise ValueError(f"Unknown array type: {arr_type}")
            elif value_type == 10:  # UINT64
                f.read(8)
            elif value_type == 11:  # INT64
                f.read(8)
            elif value_type == 12:  # FLOAT64
                f.read(8)
            else:
                raise ValueError(f"Unknown value type: {value_type}")
        
        print(f"\nTensor info starts at offset: {f.tell()}")
        
        # Now read tensor infos and patch types
        patched = 0
        for i in range(tensor_count):
            # Tensor name (string)
            name_len = struct.unpack('<Q', f.read(8))[0]
            name = f.read(name_len).decode('utf-8')
            
            # Number of dimensions
            n_dims = struct.unpack('<I', f.read(4))[0]
            
            # Dimensions (uint64 array)
            dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
            
            # Type (uint32) - THIS IS WHAT WE PATCH
            type_offset = f.tell()
            tensor_type = struct.unpack('<I', f.read(4))[0]
            
            # Offset (uint64)
            data_offset = struct.unpack('<Q', f.read(8))[0]
            
            # Check if we need to patch
            if tensor_type in TYPE_REMAP:
                new_type = TYPE_REMAP[tensor_type]
                print(f"  {name}: type {tensor_type} -> {new_type} (offset {type_offset})")
                
                # Seek back and write new type
                current_pos = f.tell()
                f.seek(type_offset)
                f.write(struct.pack('<I', new_type))
                f.seek(current_pos)
                patched += 1
            else:
                print(f"  {name}: type {tensor_type} (unchanged)")
        
        return patched

# Copy input to output
print(f"Copying {input_file} -> {output_file}")
shutil.copy(input_file, output_file)

# Patch the copy
print(f"\nPatching {output_file}...")
patched = patch_gguf(output_file)

print(f"\nDone! Patched {patched} tensor type IDs.")
print(f"Output: {output_file}")
