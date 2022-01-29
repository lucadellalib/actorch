// ==============================================================================
// Copyright 2022 Luca Della Libera. All Rights Reserved.
// ==============================================================================

#include <stdbool.h>
#include <stdint.h>
#include <Python.h>

#include <numpy/arrayobject.h>


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
"idx:"                                                               "\n"
"    The buffer current index, shape: ``[batch_size]``."             "\n"
"terminals:"                                                         "\n"
"    The buffer terminals, shape: ``[num_experiences]``."            "\n"
"priorities"                                                         "\n"
"    The buffer priorities, shape: ``[num_experiences]``."           "\n"
"num_experiences:"                                                   "\n"
"    The number of experiences stored in the buffer."                "\n"
"batch_size:"                                                        "\n"
"    The buffer batch size."                                         "\n"
                                                                     "\n"
"Returns"                                                            "\n"
"-------"                                                            "\n"
"    The trajectory total priorities, shape: ``[num_experiences]``." "\n"
                                                                     "\n"
);
static PyArrayObject *
compute_trajectory_priorities(PyObject *self, PyObject *args)
{
    PyArrayObject *idx_obj, *terminals_obj, *priorities_obj;
    int64_t num_experiences, batch_size;

    if (!PyArg_ParseTuple(
        args,
        "O!O!O!ll",
        &PyArray_Type,
        &idx_obj,
        &PyArray_Type,
        &terminals_obj,
        &PyArray_Type,
        &priorities_obj,
        &num_experiences,
        &batch_size
    )) return NULL;

    const int64_t *idx = (int64_t *)idx_obj->data;
    const bool *terminals = (bool *)terminals_obj->data;
    const float *priorities = (float *)priorities_obj->data;

    float *trajectory_priorities = (float *)malloc(num_experiences * sizeof(float));
    int64_t curr_idx[batch_size], prev[batch_size];
    for (int64_t i = 0; i < batch_size; i++)
    {
        prev[i] = trajectory_priorities[idx[i]] = priorities[idx[i]];
        curr_idx[i] = idx[i] - batch_size;
        if (curr_idx[i] < 0) curr_idx[i] += num_experiences;
    }

    const bool is_wrapped_idx = idx[0] > idx[batch_size - 1];
    for (int64_t i = 0; i < batch_size; i++)
    {
        while (is_wrapped_idx ?
               (curr_idx[i] < idx[0] && curr_idx[i] > idx[batch_size - 1]) :
               (curr_idx[i] < idx[0] || curr_idx[i] > idx[batch_size - 1])
              )
        {
            if (terminals[curr_idx[i]]) prev[i] = 0;
            prev[i] += priorities[idx[i]];
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
