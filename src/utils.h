#ifndef _UTILS
#define _UTILS

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <openssl/rand.h>

typedef __int128 int128_t;
typedef unsigned __int128 uint128_t;


static void printBytes(void *p, int num)
{
    unsigned char *c = (unsigned char *)p;
    for (int i = 0; i < num; i++)
    {
        printf("%02x", c[i]);
    }
    printf("\n");
}

// Samples a uniformly random value between 0 and max via rejection sampling.
static uint64_t random_index(uint64_t max)
{
    if (max == 0)
        return 0;

    unsigned char rand_bytes[8];

    while (1)
    {
        RAND_bytes(rand_bytes, 8);

        // Construct a random value from the random bytes
        uint64_t rand_value = 0;
        for (int i = 0; i < 8; ++i)
            rand_value |= ((uint64_t)rand_bytes[i] << (8 * i));

        // Use rejection sampling to ensure uniformity
        if (rand_value <= (UINT64_MAX - (UINT64_MAX % (max + 1))))
            return rand_value % (max + 1);
    }
}

// Samples a random trit (0,1,2) via rejection sampling
static uint8_t rand_trit()
{
    uint8_t t;
    unsigned char rand_byte;

    while (1)
    {
        RAND_bytes(&rand_byte, 1);
        t = (uint8_t)rand_byte;
        if (t <= 170) // Rejecting values greater than 170
            return t % 3;
    }
}

// Reverses the order of elements in an array of uint8_t values
static void reverse_uint8_array(uint8_t *trits, size_t size)
{
    size_t i = 0;
    size_t j = size - 1;

    while (i < j)
    {
        // Swap elements at positions i and j
        uint8_t temp = trits[i];
        trits[i] = trits[j];
        trits[j] = temp;

        // Move towards the center of the array
        i++;
        j--;
    }
}

// Converts an array of trits (not packed) into their integer representation.
static size_t trits_to_int(uint8_t *trits, size_t size)
{
    reverse_uint8_array(trits, size);
    size_t result = 0;
    for (size_t i = 0; i < size; i++)
        result = result * 3 + (size_t)trits[i];

    return result;
}

// Converts an integer into ternary representation (each trit = 0,1,2)
static void int_to_trits(size_t n, uint8_t *trits, size_t size)
{
    for (size_t i = 0; i < size; i++)
        trits[i] = 0;

    size_t index = 0;
    while (n > 0 && index < size)
    {
        trits[index] = (uint8_t)(n % 3);
        n = n / 3;
        index++;
    }
}

// Computes the log of `a` base `base`
static double log_base(double a, double base)
{
    return log2(a) / log2(base);
}

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


#endif