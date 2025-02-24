import pathlib
import jax.lax as lax
import einops as ei
import jax
import matplotlib.collections as mcollections
import numpy as np

from datetime import timedelta
from typing import Any, Callable, Iterable, Sequence, TypeVar, Tuple, List, NamedTuple
from jax import numpy as jnp, tree_util as jtu
from jax._src.lib import xla_client as xc
from matplotlib.animation import FuncAnimation
from rich.progress import Progress, ProgressColumn
from rich.text import Text

from .typing import Array, Shape, BoolScalar


def merge01(x):
    return ei.rearrange(x, "n1 n2 ... -> (n1 n2) ...")


# _P = ParamSpec("_P")
# _R = TypeVar("_R")
# _Fn = Callable[_P, _R]

_PyTree = TypeVar("_PyTree")
_Arr = TypeVar("_Arr", np.ndarray, jnp.ndarray, bool)
_T = TypeVar("_T")
_U = TypeVar("_U")


def jax_vmap(fn, in_axes: int = 0, out_axes: Any = 0):
    return jax.vmap(fn, in_axes, out_axes)


def rep_vmap(fn, rep: int, in_axes: int, **kwargs):
    for ii in range(rep):
        fn = jax.vmap(fn, in_axes=in_axes, **kwargs)
    return fn


def concat_at_front(arr1: jnp.ndarray, arr2: jnp.ndarray, axis: int) -> jnp.ndarray:
    """
    :param arr1: (nx, )
    :param arr2: (T, nx)
    :param axis: Which axis for arr2 to concat under.
    :return: (T + 1, nx) with [arr1 arr2]
    """
    # The shapes of arr1 and arr2 should be the same without the dim at axis for arr1.
    arr2_shape = list(arr2.shape)
    del arr2_shape[axis]
    assert np.all(np.array(arr2_shape) == np.array(arr1.shape))

    if isinstance(arr1, np.ndarray):
        return np.concatenate([np.expand_dims(arr1, axis=axis), arr2], axis=axis)
    else:
        return jnp.concatenate([jnp.expand_dims(arr1, axis=axis), arr2], axis=axis)


def tree_concat_at_front(tree1: _PyTree, tree2: _PyTree, axis: int) -> _PyTree:
    def tree_concat_at_front_inner(arr1: jnp.ndarray, arr2: jnp.ndarray):
        return concat_at_front(arr1, arr2, axis=axis)

    return jtu.tree_map(tree_concat_at_front_inner, tree1, tree2)


def tree_index(tree: _PyTree, idx) -> _PyTree:
    return jtu.tree_map(lambda x: x[idx], tree)


def jax2np(pytree: _PyTree) -> _PyTree:
    return jtu.tree_map(np.array, pytree)


def np2jax(pytree: _PyTree) -> _PyTree:
    return jtu.tree_map(jnp.array, pytree)


def mask2index(mask: jnp.ndarray, n_true: int) -> jnp.ndarray:
    idx = lax.top_k(mask, n_true)[1]
    return idx


def jax_jit_np(
        fn,
        static_argnums=None,
        static_argnames=None,
        donate_argnums=(),
        device: xc.Device = None,
        *args,
        **kwargs,
):
    jit_fn = jax.jit(fn, static_argnums, static_argnames, donate_argnums, device, *args, **kwargs)

    def wrapper(*args, **kwargs):
        return jax2np(jit_fn(*args, **kwargs))

    return wrapper


def chunk_vmap(fn, chunks: int):
    fn_jit_vmap = jax_jit_np(jax.vmap(fn))

    def wrapper(*args):
        args = list(args)
        # 1: Get the batch size.
        batch_size = len(jtu.tree_leaves(args[0])[0])
        chunk_idxs = np.array_split(np.arange(batch_size), chunks)

        out = []
        for idxs in chunk_idxs:
            chunk_input = jtu.tree_map(lambda x: x[idxs], args)
            out.append(fn_jit_vmap(*chunk_input))
        
        # 2: Concatenate the output.
        out = tree_merge(out)
        return out

    return wrapper


class MutablePatchCollection(mcollections.PatchCollection):
    def __init__(self, patches, *args, **kwargs):
        self._paths = None
        self.patches = patches
        mcollections.PatchCollection.__init__(self, patches, *args, **kwargs)

    def get_paths(self):
        self.set_paths(self.patches)
        return self._paths


class CustomTimeElapsedColumn(ProgressColumn):
    """Renders time elapsed."""

    def render(self, task: "Task") -> Text:
        """Show time elapsed."""
        elapsed = task.finished_time if task.finished else task.elapsed
        if elapsed is None:
            return Text("-:--:--", style="progress.elapsed")
        delta = timedelta(seconds=elapsed)
        delta = timedelta(seconds=delta.seconds, milliseconds=round(delta.microseconds // 1000))
        delta_str = str(delta)
        return Text(delta_str, style="progress.elapsed")


def save_anim(ani: FuncAnimation, path: pathlib.Path):
    pbar = Progress(*Progress.get_default_columns(), CustomTimeElapsedColumn())
    pbar.start()
    if hasattr(ani, "save_count"):
        save_count = ani.save_count
    else:
        save_count = ani._save_count
    task = pbar.add_task("Animating", total=save_count)

    def progress_callback(curr_frame: int, total_frames: int):
        pbar.update(task, advance=1)

    ani.save(path, progress_callback=progress_callback)
    pbar.stop()


def tree_merge(data: List[NamedTuple]):
    def body(*x):
        x = list(x)
        if isinstance(x[0], np.ndarray):
            return np.concatenate(x, axis=0)
        else:
            return jnp.concatenate(x, axis=0)
    out = jtu.tree_map(body, *data)
    return out


def tree_stack(trees: list):
    def tree_stack_inner(*arrs):
        arrs = list(arrs)
        if isinstance(arrs[0], np.ndarray):
            return np.stack(arrs, axis=0)
        return np.stack(arrs, axis=0)

    return jtu.tree_map(tree_stack_inner, *trees)


def as_shape(shape) -> Shape:
    if isinstance(shape, int):
        shape = (shape,)
    if not isinstance(shape, tuple):
        raise ValueError(f"Expected shape {shape} to be a tuple!")
    return shape


def get_or(maybe, value: _U):
    return value if maybe is None else maybe


def assert_shape(arr: _Arr, shape, label = None) -> _Arr:
    shape = as_shape(shape)
    label = get_or(label, "array")
    if arr.shape != shape:
        raise AssertionError(f"Expected {label} of shape {shape}, but got shape {arr.shape} of type {type(arr)}!")
    return arr


def tree_where(cond, true_val: _PyTree, false_val: _PyTree) -> _PyTree:
    return jtu.tree_map(lambda x, y: jnp.where(cond, x, y), true_val, false_val)

