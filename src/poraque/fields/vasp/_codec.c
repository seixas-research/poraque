/*
 * -*- coding: utf-8 -*-
 * file: _codec.c
 *
 * This code is part of Poraquê.
 * MIT License
 *
 * Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>
 *
 * The text codec of VASP volumetric grid blocks, for the cache builder.
 *
 * Reading and writing the E17.11 text of a CHGCAR were 86 % of the cache
 * build per material, both pure Python: a split() and a float() per value on
 * the way in, a fortran_exponential() per value on the way out. This file
 * does both in C, loaded through ctypes by poraque/fields/vasp/codec.py, and
 * must agree with the Python path exactly:
 *
 *   - reading uses strtod, which rounds correctly, so every value is the
 *     double Python's float() would produce;
 *   - writing reproduces fortran_exponential byte for byte: the mantissa in
 *     [0.1, 1), the optional leading zero dropped when the sign would
 *     overflow the field, a three-digit exponent written without its letter.
 *
 * Every entry point runs under the "C" locale (uselocale), so a process that
 * set a decimal comma cannot make strtod read 0.5 as 0 or snprintf write
 * 0,5. The reader never relies on a NUL terminator: each token is copied into
 * a bounded local buffer before it is converted, so the caller may hand in a
 * window of a larger buffer, as the streaming reader does.
 */

#define _GNU_SOURCE
#include <locale.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef __APPLE__
#include <xlocale.h>
#endif

#define PORAQUE_CODEC_ABI 3

/* Longest token converted; a grid value is 17-18 characters. */
#define TOKEN_MAX 64

int poraque_codec_abi(void) { return PORAQUE_CODEC_ABI; }

static locale_t c_locale(void)
{
    static locale_t cached = (locale_t)0;
    if (cached == (locale_t)0)
        cached = newlocale(LC_ALL_MASK, "C", (locale_t)0);
    return cached;
}

static int is_space(char c)
{
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f'
        || c == '\v';
}

/*
 * Convert one token, `p[0:length]`. Returns 1 and sets `*value` on success,
 * 0 when the token is not a number. Accepts Fortran's letterless three-digit
 * exponent (0.50000000000-299), which strtod stops short of.
 */
static int convert(const char *p, int64_t length, double *value)
{
    char token[TOKEN_MAX + 2];
    char *rest = NULL;

    if (length <= 0 || length > TOKEN_MAX)
        return 0;
    memcpy(token, p, (size_t)length);
    token[length] = '\0';
    *value = strtod(token, &rest);
    if (rest == token)
        return 0;
    if (*rest == '\0')
        return 1;
    /* A sign right after the mantissa, followed only by digits, and no
     * exponent letter consumed: insert the letter and convert again. */
    if ((*rest == '+' || *rest == '-') && rest[1] >= '0' && rest[1] <= '9'
        && strpbrk(token, "eEdD") == NULL) {
        const char *digit = rest + 1;
        while (*digit >= '0' && *digit <= '9')
            digit++;
        if (*digit != '\0')
            return 0;
        char letter[TOKEN_MAX + 3];
        int64_t mantissa = (int64_t)(rest - token);
        memcpy(letter, token, (size_t)mantissa);
        letter[mantissa] = 'E';
        strcpy(letter + mantissa + 1, rest);
        *value = strtod(letter, &rest);
        return *rest == '\0';
    }
    return 0;
}

/*
 * Parse up to `n` whitespace-separated numbers from `buf[0:len]` into `out`.
 *
 * Returns the number parsed. `*end` is the offset just past the last token
 * converted -- not past its line: the streaming caller may stop mid-line at a
 * window boundary and resume. `*bad` is set to 1 when parsing stopped at a
 * token that is not a number (a `****` overflow field, say), 0 when it ran out
 * of tokens or reached `n`.
 */
int64_t poraque_parse_doubles(const char *buf, int64_t len, double *out,
                              int64_t n, int64_t *end, int *bad)
{
    locale_t previous = uselocale(c_locale());
    const char *p = buf;
    const char *stop = buf + len;
    int64_t count = 0;

    *bad = 0;
    *end = 0;
    while (count < n) {
        while (p < stop && is_space(*p))
            p++;
        if (p >= stop)
            break;
        const char *start = p;
        while (p < stop && !is_space(*p))
            p++;
        if (!convert(start, (int64_t)(p - start), &out[count])) {
            *bad = 1;
            *end = (int64_t)(start - buf);
            uselocale(previous);
            return count;
        }
        count++;
        *end = (int64_t)(p - buf);
    }
    uselocale(previous);
    return count;
}

/*
 * Skip `n` whitespace-separated tokens of `buf[0:len]` without converting
 * them: the grid block between a CHGCAR's header and the augmentation records
 * that follow it. Returns the number skipped; `*end` is the offset just past
 * the last one.
 */
int64_t poraque_skip_tokens(const char *buf, int64_t len, int64_t n,
                            int64_t *end)
{
    const char *p = buf;
    const char *stop = buf + len;
    int64_t count = 0;

    *end = 0;
    while (count < n) {
        while (p < stop && is_space(*p))
            p++;
        if (p >= stop)
            break;
        while (p < stop && !is_space(*p))
            p++;
        count++;
        *end = (int64_t)(p - buf);
    }
    return count;
}

/* One value as Fortran's Ew.d: exactly `width` characters into `out`. */
static void fortran_e(double value, int decimals, int width, char *out)
{
    double mantissa;
    int exponent;
    char body[160];
    char rounded[160];

    if (!isfinite(value) || value == 0.0) {
        mantissa = 0.0;
        exponent = 0;
    } else {
        exponent = (int)floor(log10(fabs(value))) + 1;
        mantissa = value / pow(10.0, exponent);
        /* Rounding can carry the mantissa to 1.0, which is not a legal
         * Fortran mantissa; renormalize when it does. */
        snprintf(rounded, sizeof(rounded), "%.*f", decimals, fabs(mantissa));
        if (rounded[0] != '0') {
            mantissa /= 10.0;
            exponent += 1;
        }
    }

    int length = snprintf(body, sizeof(body), "%.*f", decimals, mantissa);
    if (exponent >= -99 && exponent <= 99)
        length += snprintf(body + length, sizeof(body) - (size_t)length,
                           "E%+03d", exponent);
    else
        length += snprintf(body + length, sizeof(body) - (size_t)length,
                           "%+04d", exponent);

    char *start = body;
    if (length > width) {
        if (strncmp(start, "-0.", 3) == 0) {
            start[1] = '-';
            start += 1;
            length -= 1;
        } else if (strncmp(start, "0.", 2) == 0) {
            start += 1;
            length -= 1;
        }
    }
    if (length > width) {
        memset(out, '*', (size_t)width);
        return;
    }
    memset(out, ' ', (size_t)(width - length));
    memcpy(out + (width - length), start, (size_t)length);
}

/*
 * Format `n` values as a grid block: `columns` per line, each written as
 * " " + Ew.d, every line ended by a newline. `out` must hold
 * n * (width + 1) + ceil(n / columns) bytes; the count written is returned,
 * or -1 for arguments outside 1 <= columns, 1 <= width <= 64,
 * 0 <= decimals <= 40 (the Python side validates them first).
 */
int64_t poraque_format_block(const double *values, int64_t n, int columns,
                             int width, int decimals, char *out)
{
    if (columns < 1 || width < 1 || width > 64 || decimals < 0
        || decimals > 40)
        return -1;
    locale_t previous = uselocale(c_locale());
    char *p = out;
    for (int64_t i = 0; i < n; i++) {
        *p++ = ' ';
        fortran_e(values[i], decimals, width, p);
        p += width;
        if ((i + 1) % columns == 0 || i + 1 == n)
            *p++ = '\n';
    }
    uselocale(previous);
    return (int64_t)(p - out);
}
