""" Test the FDTD type-checking """


## Imports
import fdtd

# dict to test the arrray class for the backends
class_for_backend = {
    "numpy": "numpy.ndarray",
    "torch.float32": "torch.Tensor",
    "torch.float64": "torch.Tensor",
    "torch.cuda.float32": "torch.Tensor",
    "torch.cuda.float64": "torch.Tensor",
}

# dict to test the dtype from the backends
dtype_for_backend = {
    "numpy": "float64",
    "torch.float32": "torch.float32",
    "torch.float64": "torch.float64",
    "torch.cuda.float32": "torch.float32",
    "torch.cuda.float64": "torch.float64",
}

## Tests
def test_class_of_saved_data_from_LineDetector_all_bends(backends):
    fdtd.set_backend(backends)
    grid = fdtd.Grid(shape=(3, 3, 1))
    grid[2, 2] = fdtd.detectors.LineDetector()
    grid.run(2)

    var_to_test = grid.detectors[0].E[0]

    assert str(type(var_to_test)) == "<class '" + class_for_backend[backends] + "'>"


def test_dtype_of_saved_data_from_LineDetector_all_bends(backends):
    fdtd.set_backend(backends)
    grid = fdtd.Grid(shape=(3, 3, 1))
    grid[2, 2] = fdtd.detectors.LineDetector()
    grid.run(2)

    var_to_test = grid.detectors[0].E[0]
    
    assert str(var_to_test.dtype) == dtype_for_backend[backends]
