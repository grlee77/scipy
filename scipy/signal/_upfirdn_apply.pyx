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
cimport numpy as cnp
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
    cdef Py_ssize_t len_h = h_trans_flip.size
    # TODO: is their a way to keep nogil below and not have to declare all
    #       possible pointer types up here?
    # (cannot use cnp.PyArray_DATA within nogil block)
    cdef double *d_dptr
    cdef double *d_hptr
    cdef double *d_optr
    cdef float *s_dptr
    cdef float *s_hptr
    cdef float *s_optr
    cdef float_complex *c_dptr
    cdef float_complex *c_hptr
    cdef float_complex *c_optr
    cdef double_complex *z_dptr
    cdef double_complex *z_hptr
    cdef double_complex *z_optr

    data_info.ndim = data.ndim
    data_info.strides = <Py_ssize_t *> data.strides
    data_info.shape = <Py_ssize_t *> data.shape

    output_info.ndim = out.ndim
    output_info.strides = <Py_ssize_t *> out.strides
    output_info.shape = <Py_ssize_t *> out.shape

    if data.dtype == np.float64:
        d_dptr = <double*> cnp.PyArray_DATA(data)
        d_hptr = <double*> cnp.PyArray_DATA(h_trans_flip)
        d_optr = <double*> cnp.PyArray_DATA(out)
        with nogil:
            retval = _apply_axis_inner(d_dptr, data_info,
                                       d_hptr, len_h,
                                       d_optr, output_info,
                                       up, down, axis)
        if retval:
            raise RuntimeError("_apply_axis_inner failed")
    elif data.dtype == np.float32:
        s_dptr = <float*> cnp.PyArray_DATA(data)
        s_hptr = <float*> cnp.PyArray_DATA(h_trans_flip)
        s_optr = <float*> cnp.PyArray_DATA(out)
        with nogil:
            retval = _apply_axis_inner(s_dptr, data_info,
                                       s_hptr, len_h,
                                       s_optr, output_info,
                                       up, down, axis)
    elif data.dtype == np.complex128:
        z_dptr = <double_complex*> cnp.PyArray_DATA(data)
        z_hptr = <double_complex*> cnp.PyArray_DATA(h_trans_flip)
        z_optr = <double_complex*> cnp.PyArray_DATA(out)
        with nogil:
            retval = _apply_axis_inner(z_dptr, data_info,
                                       z_hptr, len_h,
                                       z_optr, output_info,
                                       up, down, axis)
        if retval:
            raise RuntimeError("_apply_axis_inner failed")
    elif data.dtype == np.complex64:
        c_dptr = <float_complex*> cnp.PyArray_DATA(data)
        c_hptr = <float_complex*> cnp.PyArray_DATA(h_trans_flip)
        c_optr = <float_complex*> cnp.PyArray_DATA(out)
        with nogil:
            retval = _apply_axis_inner(c_dptr, data_info,
                                       c_hptr, len_h,
                                       c_optr, output_info,
                                       up, down, axis)
        if retval:
            raise RuntimeError("_apply_axis_inner failed")
    return out


cdef int _apply_axis_inner(DTYPE_t* input, ArrayInfo input_info,
                           DTYPE_t* h_trans_flip, Py_ssize_t len_h,
                           DTYPE_t* output, ArrayInfo output_info,
                           Py_ssize_t up, Py_ssize_t down,
                           Py_ssize_t axis) nogil:
    cdef Py_ssize_t i
    cdef Py_ssize_t num_loops = 1
    cdef bint make_temp_input, make_temp_output  # booleans
    cdef DTYPE_t* temp_input = NULL
    cdef DTYPE_t* temp_output = NULL

    if (input_info.ndim != output_info.ndim):
        return 1
    if (axis >= input_info.ndim):
        return 1

    make_temp_input = input_info.strides[axis] != sizeof(DTYPE_t);
    make_temp_output = output_info.strides[axis] != sizeof(DTYPE_t);
    if make_temp_input:
        temp_input = <DTYPE_t*>malloc(input_info.shape[axis] * sizeof(DTYPE_t))
        if not temp_input:
            free(temp_input)
            free(temp_output)
            return 2
    if make_temp_output:
        temp_output = <DTYPE_t*>malloc(output_info.shape[axis] * sizeof(DTYPE_t))
        if not temp_output:
            free(temp_input)
            free(temp_output)
            return 2

    for i in range(output_info.ndim):
        if i != axis:
            num_loops *= output_info.shape[i]

    cdef Py_ssize_t j
    cdef Py_ssize_t input_offset = 0
    cdef Py_ssize_t output_offset = 0
    cdef const DTYPE_t* input_row
    cdef DTYPE_t* output_row
    cdef Py_ssize_t reduced_idx
    cdef Py_ssize_t j_rev
    cdef Py_ssize_t axis_idx
    cdef DTYPE_t* tmp_ptr = NULL
    for i in range(num_loops):

        # Calculate offset into linear buffer
        reduced_idx = i
        for j in range(output_info.ndim):
            j_rev = output_info.ndim - 1 - j
            if j_rev != axis:
                axis_idx = reduced_idx % output_info.shape[j_rev]
                reduced_idx /= output_info.shape[j_rev]
                input_offset += (axis_idx * input_info.strides[j_rev])
                output_offset += (axis_idx * output_info.strides[j_rev])

        # Copy to temporary input if necessary
        if make_temp_input:
            for j in range(input_info.shape[axis]):
                # Offsets are byte offsets, to need to cast to char and back
                temp_input[j] = (<DTYPE_t *>((<char *> input) + input_offset +
                    j * input_info.strides[axis]))[0]

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
        _apply_impl(input_row, input_info.shape[axis],
                    h_trans_flip, len_h, output_row, up, down)

        # Copy from temporary output if necessary
        if make_temp_output:
            for j in range(output_info.shape[axis]):
                # Offsets are byte offsets, to need to cast to char and back
                tmp_ptr = (<DTYPE_t *>((<char *> output) + output_offset +
                    j * output_info.strides[axis]))
                tmp_ptr[0] = output_row[j]

    # cleanup
    free(temp_input)
    free(temp_output)
    return 0


def _apply(DTYPE_t [:] x, DTYPE_t [:] h_trans_flip, DTYPE_t [:] out,
                 Py_ssize_t up, Py_ssize_t down):
    """legacy version without axis support"""
    _apply_impl(&x[0], len(x), &h_trans_flip[0], len(h_trans_flip),
                &out[0], up, down)


@cython.cdivision(True)  # faster modulo
@cython.boundscheck(False)  # designed to stay within bounds
@cython.wraparound(False)  # we don't use negative indexing
cdef void _apply_impl(DTYPE_t *x, Py_ssize_t len_x, DTYPE_t *h_trans_flip,
                      Py_ssize_t len_h, DTYPE_t *out,
                      Py_ssize_t up, Py_ssize_t down) nogil:
    #cdef Py_ssize_t len_x = x.shape[0]
    cdef Py_ssize_t h_per_phase = len_h / up
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
