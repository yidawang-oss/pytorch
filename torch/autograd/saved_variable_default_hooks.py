import torch
from typing import Callable, Any

def set_saved_tensors_default_hooks(pack_hook: Callable[[torch.Tensor], Any], unpack_hook: Callable[[Any], torch.Tensor]):
    r"""Sets a pair of pack / unpack hooks for saved tensors.

    When default hooks are set, the ``pack_hook`` function will be called everytime
    an operation saves a tensor for backward. This includes intermediary results
    saved using :func:`~torch.autograd.function._ContextMethodMixin.save_for_backward` but also those needed
    by a Pytorch-defined operation.
    The ``unpack_hook`` is called when the saved tensor needs to be accessed, namely
    when executing :func:`torch.Tensor.backward()` or :func:`torch.autograd.grad()`.
    The hooks should have the following signatures:

        pack_hook(tensor: Tensor) -> Any

        unpack_hook(Any) -> Tensor

    where the return value of ``pack_hook`` is a valid input to ``unpack_hook``.

    Example::

        >>> def pack_hook(x):
        >>>     print("Packing", x)
        >>>     return x
        >>>
        >>> def unpack_hook(x):
        >>>     print("Unpacking", x)
        >>>     return x
        >>>
        >>> a = torch.ones(5, requires_grad=True)
        >>> b = torch.ones(5, requires_grad=True) * 2
        >>> torch.autograd.graph.set_saved_tensors_default_hooks(pack_hook, unpack_hook)
        >>> y = a * b
        Packing tensor([1., 1., 1., 1., 1.])
        Packing tensor([2., 2., 2., 2., 2.])
        >>> torch.autograd.graph.reset_saved_tensors_default_hooks()
        >>> y.sum().backward()
        Unpacking tensor([1., 1., 1., 1., 1.])
        Unpacking tensor([2., 2., 2., 2., 2.])

    .. warning ::
        Modifying the input to either hooks may lead to undefined behavior.

    .. warning ::
        Only one pair of hooks is allowed at a time. To register a different
        pair of hooks, call :func:`~torch.autograd.graph.reset_saved_tensors_default_hooks()`
        first.

    """
    torch._C._autograd._register_default_hooks(pack_hook, unpack_hook)

def reset_saved_tensors_default_hooks():
    r"""Removes the current pair of default pack / unpack hooks for saved tensors.

    Intermediary values that are saved after this function is called won't be
    packed / unpacked with the default hooks previously defined.

    This does not affect previously saved tensors: intermediary values that
    were saved before this function is called are still stored in their *packed* form
    and will be unpacked using the corresponding ``unpack_hook``.

    Also see :func:`~torch.autograd.graph.set_saved_tensors_default_hooks()`.
    """
    torch._C._autograd._reset_default_hooks()
