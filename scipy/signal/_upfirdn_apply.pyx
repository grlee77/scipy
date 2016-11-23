# -*- coding: utf-8 -*-

# Code adapted from "upfirdn" python library with permission:
#
# Copyright (c) 2009, Motorola, Inc
#
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# * Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# * Neither the name of Motorola nor the names of its contributors may be
# used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cimport cython
cimport numpy as np
import numpy as np
from cython import bint  # boolean integer type
from libc.stdlib cimport malloc, free


ctypedef double complex double_complex
ctypedef float complex float_complex

ctypedef fused DTYPE_t:
    # Eventually we could add "object", too, but then we'd lose the "nogil"
    # on the _apply_impl function.
    float
    float_complex
    double
    double_complex

cdef struct ArrayInfo:
    Py_ssize_t * shape
    Py_ssize_t * strides
    Py_ssize_t ndim


def _output_len(Py_ssize_t len_h,
                Py_ssize_t in_len,
                Py_ssize_t up,
                Py_ssize_t down):
    """The output length that results from a given input"""
    cdef Py_ssize_t np
    cdef Py_ssize_t in_len_copy
    in_len_copy = in_len + (len_h + (-len_h % up)) // up - 1
    np = in_len_copy * up
    cdef Py_ssize_t need = np // down
    if np % down > 0:
        need += 1
    return need


cpdef _apply_axis(np.ndarray data, np.ndarray h_trans_flip, np.ndarray out,
                  Py_ssize_t up, Py_ssize_t down, Py_ssize_t axis):
    cdef ArrayInfo data_info, output_info

    data_info.ndim = data.ndim
    data_info.strides = <Py_ssize_t *> data.strides
    data_info.shape = <Py_ssize_t *> data.shape

    output_info.ndim = out.ndim
    output_info.strides = <Py_ssize_t *> out.strides
    output_info.shape = <Py_ssize_t *> out.shape

    if data.dtype == np.float64:
        with nogil:
            _apply_axis_inner(<double *> data.data, input_info,
                              <double *> h_trans_flip.data,
                              <double *> output.data, output_info,
                              up, down, axis)
        if retval:
            raise RuntimeError("_apply_axis_inner failed")
    elif data.dtype == np.float32:
        with nogil:
            _apply_axis_inner(<float *> data.data, input_info,
                              <float *> h_trans_flip.data,
                              <float *> output.data, output_info,
                              up, down, axis)
        if retval:
            raise RuntimeError("_apply_axis_inner failed")
    elif data.dtype == np.complex128:
        with nogil:
            _apply_axis_inner(<double_complex *> data.data, input_info,
                              <double_complex *> h_trans_flip.data,
                              <double_complex *> output.data, output_info,
                              up, down, axis)
        if retval:
            raise RuntimeError("_apply_axis_inner failed")
    elif data.dtype == np.complex64:
        with nogil:
            _apply_axis_inner(<float_complex *> data.data, input_info,
                              <float_complex *> h_trans_flip.data,
                              <float_complex *> output.data, output_info,
                              up, down, axis)
        if retval:
            raise RuntimeError("_apply_axis_inner failed")
    return output


cdef int _apply_axis_inner(DTYPE_t* input, ArrayInfo input_info,
                           DTYPE_t* h_trans_flip,
                           DTYPE_t* output, ArrayInfo output_info,
                           Py_ssize_t up, Py_ssize_t down,
                           Py_ssize_t axis):
    cdef Py_ssize_t i
    cdef num_loops = 1
    cdef bint make_temp_input, make_temp_output  # booleans
    DTYPE_t* temp_input = NULL
    DTYPE_t* temp_output = NULL

    if (input_info.ndim != output_info.ndim):
        return 1
    if (axis >= input_info.ndim):
        return 1

    make_temp_input = input_info.strides[axis] != sizeof(DTYPE_t);
    make_temp_output = output_info.strides[axis] != sizeof(DTYPE_t);
    if (make_temp_input):
        if ((temp_input = malloc(input_info.shape[axis] * sizeof(DTYPE_t))) == NULL):
            free(temp_input)
            free(temp_output)
            return 2
    if (make_temp_output):
        if ((temp_output = malloc(output_info.shape[axis] * sizeof(DTYPE_t))) == NULL):
            free(temp_input)
            free(temp_output)
            return 2

    for i in range(output_info.ndim):
        if i != axis:
            num_loops *= output_info.shape[i]

    for i in range(num_loops):
        cdef size_t j
        cdef Py_ssize_t input_offset = 0
        cdef Py_ssize_t output_offset = 0
        cdef const DTYPE_t* input_row
        cdef DTYPE_t* output_row

        # Calculate offset into linear buffer
        cdef Py_ssize_t reduced_idx = i
        for j in range(output_info.ndim):
            cdef Py_ssize_t j_rev = output_info.ndim - 1 - j
            if j_rev != axis:
                cdef Py_ssize_t axis_idx
                axis_idx = reduced_idx % output_info.shape[j_rev]
                reduced_idx /= output_info.shape[j_rev]
                input_offset += (axis_idx * input_info.strides[j_rev])
                output_offset += (axis_idx * output_info.strides[j_rev])

        # Copy to temporary input if necessary
        if make_temp_input:
            for j in range(input_info.shape[axis]):
                # Offsets are byte offsets, to need to cast to char and back
                temp_input[j] = *(<DTYPE_t *>((<char *> input) + input_offset +
                    j * input_info.strides[axis]))

        # Select temporary or direct output and input
        if make_temp_input:
            input_row = temp_input
        else:
            input_row = <const DTYPE_t *>(<const char *> input + input_offset)
        if make_temp_output:
            output_row = temp_output
        else:
            output_row = <DTYPE_t *>(<char *> output + output_offset)

        # call 1D upfirdn
        _apply_impl(input_row, h_trans_flip, output_row, up, down)

        # Copy from temporary output if necessary
        if make_temp_output:
            for j in range(output_info.shape[axis]):
                # Offsets are byte offsets, to need to cast to char and back
                *(<DTYPE_t *>((<char *> output) + output_offset +
                    j * output_info.strides[axis])) = output_row[j]
    # cleanup
    free(temp_input)
    free(temp_output)
    return 0


def _apply(DTYPE_t [:] x, DTYPE_t [:] h_trans_flip, DTYPE_t [:] out,
                 Py_ssize_t up, Py_ssize_t down):
    """legacy version without axis support"""
    _apply_impl(&x[0], &h_trans_flip[0], &out[0], up, down)


@cython.cdivision(True)  # faster modulo
@cython.boundscheck(False)  # designed to stay within bounds
@cython.wraparound(False)  # we don't use negative indexing
cdef void _apply_impl(DTYPE_t *x, DTYPE_t *h_trans_flip, DTYPE_t *out,
                      Py_ssize_t up, Py_ssize_t down) nogil:
    cdef Py_ssize_t len_x = x.shape[0]
    cdef Py_ssize_t h_per_phase = h_trans_flip.shape[0] / up
    cdef Py_ssize_t padded_len = len_x + h_per_phase - 1
    cdef Py_ssize_t x_idx = 0
    cdef Py_ssize_t y_idx = 0
    cdef Py_ssize_t h_idx = 0
    cdef Py_ssize_t t = 0
    cdef Py_ssize_t x_conv_idx = 0

    while x_idx < len_x:
        h_idx = t * h_per_phase
        x_conv_idx = x_idx - h_per_phase + 1
        if x_conv_idx < 0:
            h_idx -= x_conv_idx
            x_conv_idx = 0
        for x_conv_idx in range(x_conv_idx, x_idx + 1):
            out[y_idx] = out[y_idx] + x[x_conv_idx] * h_trans_flip[h_idx]
            h_idx += 1
        # store and increment
        y_idx += 1
        t += down
        x_idx += t / up  # integer div
        # which phase of the filter to use
        t = t % up

    # Use a second simplified loop to flush out the last bits
    while x_idx < padded_len:
        h_idx = t * h_per_phase
        x_conv_idx = x_idx - h_per_phase + 1
        for x_conv_idx in range(x_conv_idx, x_idx + 1):
            if x_conv_idx < len_x and x_conv_idx > 0:
                out[y_idx] = out[y_idx] + x[x_conv_idx] * h_trans_flip[h_idx]
            h_idx += 1
        y_idx += 1
        t += down
        x_idx += t / up  # integer div
        t = t % up
