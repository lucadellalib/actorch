// ==============================================================================
// Copyright 2022 Luca Della Libera.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

#include <stdbool.h>
#include <stdint.h>
#include <Python.h>

#include <numpy/arrayobject.h>


// NumPy arrays are in general non-contiguous
#define IDX(i) *(long *)(idx_obj->data + (i) * idx_obj->strides[0])
#define TERMINALS(i) *(bool *)(terminals_obj->data + (i) * terminals_obj->strides[0])
#define PRIORITIES(i) *(float *)(priorities_obj->data + (i) * priorities_obj->strides[0])


PyDoc_STRVAR(
utils__doc__,
"Buffer utilities."
);


static void
cleanup(PyObject *capsule)
{
    void *memory = PyCapsule_GetPointer(capsule, NULL);
    free(memory);
}


PyDoc_STRVAR(
compute_trajectory_priorities__doc__,
"Compute total priorities of the trajectories stored in a buffer."   "\n"
                                                                     "\n"
"Parameters"                                                         "\n"
"----------"                                                         "\n"
"batch_size:"                                                        "\n"
"    The buffer batch size."                                         "\n"
"num_experiences:"                                                   "\n"
"    The number of experiences stored in the buffer."                "\n"
"idx:"                                                               "\n"
"    The buffer current index, shape: ``[batch_size]``."             "\n"
"terminals:"                                                         "\n"
"    The buffer terminals, shape: ``[num_experiences]``."            "\n"
"priorities"                                                         "\n"
"    The buffer priorities, shape: ``[num_experiences]``."           "\n"
                                                                     "\n"
"Returns"                                                            "\n"
"-------"                                                            "\n"
"    The trajectory total priorities, shape: ``[num_experiences]``." "\n"
                                                                     "\n"
);
static PyArrayObject *
compute_trajectory_priorities(PyObject *self, PyObject *args)
{
    long batch_size, num_experiences;
    PyArrayObject *idx_obj, *terminals_obj, *priorities_obj;

    if (!PyArg_ParseTuple(
        args,
        "llO!O!O!",
        &batch_size,
        &num_experiences,
        &PyArray_Type,
        &idx_obj,
        &PyArray_Type,
        &terminals_obj,
        &PyArray_Type,
        &priorities_obj
    )) return NULL;

    float *trajectory_priorities = (float *)malloc((long)(num_experiences * sizeof(float)));
    long *curr_idx = (long *)malloc((long)(batch_size * sizeof(long)));
    float *prev = (float *)malloc((long)(batch_size * sizeof(float)));

    for (long i = 0; i < batch_size; i++)
    {
        prev[i] = trajectory_priorities[IDX(i)] = PRIORITIES(IDX(i));
        curr_idx[i] = IDX(i) - batch_size;
        if (curr_idx[i] < 0) curr_idx[i] += num_experiences;
    }

    const bool is_wrapped_idx = IDX(0) > IDX(batch_size - 1);
    for (long i = 0; i < batch_size; i++)
    {
        while (is_wrapped_idx ?
               (curr_idx[i] < IDX(0) && curr_idx[i] > IDX(batch_size - 1)) :
               (curr_idx[i] < IDX(0) || curr_idx[i] > IDX(batch_size - 1))
              )
        {
            if (TERMINALS(curr_idx[i])) prev[i] = 0;
            prev[i] += PRIORITIES(IDX(i));
            trajectory_priorities[curr_idx[i]] = prev[i];
            curr_idx[i] -= batch_size;
            if (curr_idx[i] < 0) curr_idx[i] += num_experiences;
        }
    }

    const npy_intp dims[1] = {num_experiences};
    const PyObject *py_array = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, trajectory_priorities);
    const PyObject *result = Py_BuildValue("O", py_array);
    const PyObject *capsule = PyCapsule_New(trajectory_priorities, NULL, cleanup);
    PyArray_SetBaseObject((PyArrayObject *)result, capsule);
    free(curr_idx);
    free(prev);
    return result;
}


static PyMethodDef utils_methods[] = {
    {
        "compute_trajectory_priorities",
	    compute_trajectory_priorities,
	    METH_VARARGS,
        compute_trajectory_priorities__doc__
    },
	{NULL, NULL, 0, NULL}  // Sentinel
};


static struct PyModuleDef utils_module = {
    PyModuleDef_HEAD_INIT,
    "utils",
    utils__doc__,
    -1,
    utils_methods
};


PyMODINIT_FUNC
PyInit_utils(void)
{
    PyObject *module;
    module = PyModule_Create(&utils_module);
    if (module == NULL) return NULL;
    import_array();
    if (PyErr_Occurred()) return NULL;
    return module;
}
