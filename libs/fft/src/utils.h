#ifndef _UTILS
#define _UTILS

// Compute base^exp without the floating-point precision
// errors of the built-in pow function.
static inline size_t ipow(size_t base, size_t exp)
{
    if (exp == 1)
        return base;

    if (exp == 0)
        return 1;

    size_t result = 1;
    while (1)
    {
        if (exp & 1)
            result *= base;
        exp >>= 1;
        if (!exp)
            break;
        base *= base;
    }

    return result;
}

// Multiplications of 16 GF(4) elements.
// 16 elements are paced into an element of uint32_t.
static inline uint32_t multiply_32(const uint32_t a, const uint32_t b) {

    const uint32_t pattern = 0xaaaaaaaa;
    uint32_t mask_h = pattern;     // 0b101010101010101001010
    uint32_t mask_l = mask_h >> 1; // 0b010101010101010100101

    uint32_t a_h, a_l, b_h, b_l;
    // multiplication over F4
    a_h = (a & mask_h);
    a_l = (a & mask_l);
    b_h = (b & mask_h);
    b_l = (b & mask_l);

    uint32_t tmp = (a_h & b_h);
    uint32_t rlt = tmp ^ (a_h & (b_l << 1));
    rlt ^= ((a_l << 1) & b_h);
    rlt |= a_l & b_l ^ (tmp >> 1);
    return rlt;
}

static inline uint64_t multiply_64(const uint64_t a, const uint64_t b)
{
    const uint64_t pattern = 0xaaaaaaaaaaaaaaaa;
    uint64_t mask_h = pattern;     // 0b101010101010101001010
    uint64_t mask_l = mask_h >> 1; // 0b010101010101010100101

    uint64_t a_h, a_l, b_h, b_l;
    // multiplication over F4
    a_h = (a & mask_h);
    a_l = (a & mask_l);
    b_h = (b & mask_h);
    b_l = (b & mask_l);

    uint64_t tmp = (a_h & b_h);
    uint64_t rlt = tmp ^ (a_h & (b_l << 1));
    rlt ^= ((a_l << 1) & b_h);
    rlt |= a_l & b_l ^ (tmp >> 1);
    return rlt;
}

#endif