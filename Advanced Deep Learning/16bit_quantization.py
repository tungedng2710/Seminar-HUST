import struct
import math

def float32_to_fp16(f):
    """
    Convert a Python float (64-bit) to 16-bit float by:
    1) packing to float32,
    2) using the built-in half-precision conversion via struct or a library approach.
    """
    # Pack float into 32-bit binary
    b = struct.pack('>f', f)  # big-endian 32-bit
    # Unpack as integer to inspect bits
    bits = struct.unpack('>I', b)[0]
    
    # Extract sign, exponent, fraction
    sign   = (bits >> 31) & 0x1
    exp    = (bits >> 23) & 0xFF
    frac   = bits & 0x7FFFFF  # 23 bits
    
    # FP32 bias = 127, FP16 bias = 15
    # Re-bias exponent
    new_exp = exp - 127 + 15
    
    # Handle special cases
    if exp == 255:  
        # Infinity or NaN
        new_exp = 31
        new_frac = 0 if frac == 0 else (1 << 9)  # NaN's MSB fraction bit set
    elif exp == 0:
        # Zero / subnormal
        # For simplicity, clamp to zero here (proper subnormal handling is more involved)
        new_exp = 0
        new_frac = 0
    else:
        # Normalized
        # Round the fraction from 23 bits to 10 bits
        # The fraction is top 10 bits of frac, with rounding
        round_bit_mask = (1 << (23 - 10)) - 1   # bits we are cutting off
        round_bits = frac & round_bit_mask
        frac_10 = frac >> (23 - 10)
        
        # Apply simple rounding (round to nearest, ties to even)
        half_way = 1 << ((23 - 10) - 1)
        if round_bits > half_way or (round_bits == half_way and (frac_10 & 1) == 1):
            frac_10 += 1
        
        new_frac = frac_10 & 0x3FF  # 10 bits
    
    # Now pack into 16 bits
    fp16_bits = (sign << 15) | ((new_exp & 0x1F) << 10) | (new_frac & 0x3FF)
    return fp16_bits

def float32_to_bf16(f):
    """
    Convert a Python float (64-bit) to bfloat16 by:
    1) packing to float32,
    2) extracting top 16 bits (with rounding).
    """
    # Pack float into 32-bit binary
    b = struct.pack('>f', f)
    bits = struct.unpack('>I', b)[0]
    
    # Extract sign, exponent, fraction
    sign = (bits >> 31) & 0x1
    exp  = (bits >> 23) & 0xFF
    frac = bits & 0x7FFFFF  # 23 bits
    
    # We only keep 7 fraction bits in BF16 (instead of 23).
    # So we drop (23 - 7) = 16 bits from the fraction,
    # but we need to round them properly.
    to_drop = 23 - 7  # 16
    round_mask = (1 << to_drop) - 1
    dropped_bits = frac & round_mask
    frac_7 = frac >> to_drop
    
    # Round to nearest, ties to even
    half_way = 1 << (to_drop - 1)
    if dropped_bits > half_way or (dropped_bits == half_way and (frac_7 & 1) == 1):
        frac_7 += 1
    
    # Now form the BF16 bits: sign(1) + exponent(8) + fraction(7)
    bf16_bits = (sign << 15) | ((exp & 0xFF) << 7) | (frac_7 & 0x7F)
    return bf16_bits

# EXAMPLE USAGE
my_vals = [3.14, 0.0001, 12345.678, -0.75, 1.0, float('inf'), float('nan')]

for v in my_vals:
    hbits = float32_to_fp16(v)
    bbits = float32_to_bf16(v)
    print(f"Value: {v}")
    print(f"  FP16 bits = {hbits:04X} (hex)")
    print(f"  BF16 bits = {bbits:04X} (hex)\n")