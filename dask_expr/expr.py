from __future__ import annotations

import functools
import os
from collections.abc import Generator

import pandas as pd
import toolz
from dask.base import normalize_token, tokenize
from dask.core import ishashable
from dask.dataframe.core import is_dataframe_like, is_index_like, is_series_like
from dask.utils import funcname, import_required, is_arraylike

replacement_rules = []

no_default = "__no_default__"


class Expr:
    """Primary class for all Expressions

    This mostly includes Dask protocols.
    """

    _parameters = []
    _defaults = {}

    def __init__(self, *args, **kwargs):
        operands = list(args)
        for parameter in type(self)._parameters[len(operands) :]:
            try:
                operands.append(kwargs.pop(parameter))
            except KeyError:
                operands.append(type(self)._defaults[parameter])
        assert not kwargs
        self.operands = operands

    def __getattr__(self, key):
        try:
            return object.__getattribute__(self, key)
        except AttributeError as err:
            # Allow operands to be accessed as attributes
            # as long as the keys are not already reserved
            # by existing methods/properties
            _parameters = type(self)._parameters
            if key in _parameters:
                idx = _parameters.index(key)
                return self.operands[idx]
            raise err

    def __str__(self):
        s = ", ".join(
            str(param) + "=" + str(operand)
            for param, operand in zip(self._parameters, self.operands)
            if operand != self._defaults.get(param)
        )
        return f"{type(self).__name__}({s})"

    def __repr__(self):
        return str(self)

    def _tree_repr_lines(self, indent=0, recursive=True):
        header = funcname(type(self)) + ":"
        lines = []
        for i, op in enumerate(self.operands):
            if isinstance(op, Expr):
                if recursive:
                    lines.extend(op._tree_repr_lines(2))
            else:
                try:
                    param = self._parameters[i]
                    default = self._defaults[param]
                except (IndexError, KeyError):
                    param = self._parameters[i] if i < len(self._parameters) else ""
                    default = "--no-default--"

                if isinstance(op, pd.core.base.PandasObject):
                    op = "<pandas>"
                elif is_dataframe_like(op):
                    op = "<dataframe>"
                elif is_index_like(op):
                    op = "<index>"
                elif is_series_like(op):
                    op = "<series>"
                elif is_arraylike(op):
                    op = "<array>"

                elif repr(op) != repr(default):
                    if param:
                        header += f" {param}={repr(op)}"
                    else:
                        header += repr(op)
        lines = [header] + lines
        lines = [" " * indent + line for line in lines]

        return lines

    def tree_repr(self):
        return os.linesep.join(self._tree_repr_lines())

    def pprint(self):
        for line in self._tree_repr_lines():
            print(line)

    def __hash__(self):
        return hash(self._name)

    def __reduce__(self):
        return type(self), tuple(self.operands)

    def _task(self, index: int):
        """The task for the i'th partition

        Parameters
        ----------
        index:
            The index of the partition

        Returns
        -------
        task:
            The Dask task to compute this partition

        See Also
        --------
        Expr._layer
        """
        raise NotImplementedError(
            "Expressions should define either _layer (full dictionary) or _task"
            " (single task).  This expression type defines neither"
        )

    def _layer(self) -> dict:
        """The graph layer added by this expression

        Returns
        -------
        layer: dict
            The Dask task graph added by this expression

        See Also
        --------
        Expr._task
        Expr.__dask_graph__
        """
        raise NotImplementedError(
            "Expressions should define either _layer (full dictionary) or _task"
            " (single task).  This expression type defines neither"
        )

    def _depth(self):
        """Depth of the expression tree

        Returns
        -------
        depth: int
        """
        if not self.dependencies():
            return 1
        else:
            return max(expr._depth() for expr in self.dependencies()) + 1

    def operand(self, key):
        # Access an operand unambiguously
        # (e.g. if the key is reserved by a method/property)
        return self.operands[type(self)._parameters.index(key)]

    def dependencies(self):
        # Dependencies are `Expr` operands only
        return [operand for operand in self.operands if isinstance(operand, Expr)]

    @property
    def npartitions(self):
        raise NotImplementedError

    def simplify(self):
        """Simplify expression

        This leverages the ``._simplify_down`` method defined on each class

        Returns
        -------
        expr:
            output expression
        changed:
            whether or not any change occured
        """
        expr = self

        while True:
            _continue = False

            # Simplify this node
            out = expr._simplify_down()
            if out is None:
                out = expr
            if not isinstance(out, Expr):
                return out
            if out._name != expr._name:
                expr = out
                continue

            # Allow children to simplify their parents
            for child in expr.dependencies():
                out = child._simplify_up(expr)
                if out is None:
                    out = expr
                if not isinstance(out, Expr):
                    return out
                if out is not expr and out._name != expr._name:
                    expr = out
                    _continue = True
                    break

            if _continue:
                continue

            # Simplify all of the children
            new_operands = []
            changed = False
            for operand in expr.operands:
                if isinstance(operand, Expr):
                    new = operand.simplify()
                    if new._name != operand._name:
                        changed = True
                else:
                    new = operand
                new_operands.append(new)

            if changed:
                expr = type(expr)(*new_operands)
                continue
            else:
                break

        return expr

    def _simplify_down(self):
        return

    def _simplify_up(self, parent):
        return

    def optimize(self, **kwargs):
        raise NotImplementedError()

    @functools.cached_property
    def _name(self):
        return funcname(type(self)).lower() + "-" + tokenize(*self.operands)

    def __dask_graph__(self):
        """Traverse expression tree, collect layers"""
        stack = [self]
        seen = set()
        layers = []
        while stack:
            expr = stack.pop()

            if expr._name in seen:
                continue
            seen.add(expr._name)

            layers.append(expr._layer())
            for operand in expr.operands:
                if isinstance(operand, Expr):
                    stack.append(operand)

        return toolz.merge(layers)

    def __dask_keys__(self):
        return [(self._name, i) for i in range(self.npartitions)]

    def substitute(self, substitutions: dict) -> Expr:
        """Substitute specific `Expr` instances within `self`

        Parameters
        ----------
        substitutions:
            mapping old terms to new terms. Note that using
            non-`Expr` keys may produce unexpected results,
            and substituting boolean values is not allowed.

        Examples
        --------
        >>> (df + 10).substitute({10: 20})
        df + 20
        """
        if not substitutions:
            return self

        if self in substitutions:
            return substitutions[self]

        new = []
        update = False
        for operand in self.operands:
            if (
                not isinstance(operand, bool)
                and ishashable(operand)
                and operand in substitutions
            ):
                new.append(substitutions[operand])
                update = True
            elif isinstance(operand, Expr):
                val = operand.substitute(substitutions)
                if operand._name != val._name:
                    update = True
                new.append(val)
            else:
                new.append(operand)

        if update:  # Only recreate if something changed
            return type(self)(*new)
        return self

    def _node_label_args(self):
        """Operands to include in the node label by `visualize`"""
        return self.dependencies()

    def _to_graphviz(
        self,
        rankdir="BT",
        graph_attr=None,
        node_attr=None,
        edge_attr=None,
        **kwargs,
    ):
        from dask.dot import label, name

        graphviz = import_required(
            "graphviz",
            "Drawing dask graphs with the graphviz visualization engine requires the `graphviz` "
            "python library and the `graphviz` system library.\n\n"
            "Please either conda or pip install as follows:\n\n"
            "  conda install python-graphviz     # either conda install\n"
            "  python -m pip install graphviz    # or pip install and follow installation instructions",
        )

        graph_attr = graph_attr or {}
        node_attr = node_attr or {}
        edge_attr = edge_attr or {}

        graph_attr["rankdir"] = rankdir
        node_attr["shape"] = "box"
        node_attr["fontname"] = "helvetica"

        graph_attr.update(kwargs)
        g = graphviz.Digraph(
            graph_attr=graph_attr,
            node_attr=node_attr,
            edge_attr=edge_attr,
        )

        stack = [self]
        seen = set()
        dependencies = {}
        while stack:
            expr = stack.pop()

            if expr._name in seen:
                continue
            seen.add(expr._name)

            dependencies[expr] = set(expr.dependencies())
            for dep in expr.dependencies():
                stack.append(dep)

        cache = {}
        for expr in dependencies:
            expr_name = name(expr)
            attrs = {}

            # Make node label
            deps = [
                funcname(type(dep)) if isinstance(dep, Expr) else str(dep)
                for dep in expr._node_label_args()
            ]
            _label = funcname(type(expr))
            if deps:
                _label = f"{_label}({', '.join(deps)})" if deps else _label
            node_label = label(_label, cache=cache)

            attrs.setdefault("label", str(node_label))
            attrs.setdefault("fontsize", "20")
            g.node(expr_name, **attrs)

        for expr, deps in dependencies.items():
            expr_name = name(expr)
            for dep in deps:
                dep_name = name(dep)
                g.edge(dep_name, expr_name)

        return g

    def visualize(self, filename="dask-expr.svg", format=None, **kwargs):
        """
        Visualize the expression graph.
        Requires ``graphviz`` to be installed.

        Parameters
        ----------
        filename : str or None, optional
            The name of the file to write to disk. If the provided `filename`
            doesn't include an extension, '.png' will be used by default.
            If `filename` is None, no file will be written, and the graph is
            rendered in the Jupyter notebook only.
        format : {'png', 'pdf', 'dot', 'svg', 'jpeg', 'jpg'}, optional
            Format in which to write output file. Default is 'svg'.
        **kwargs
           Additional keyword arguments to forward to ``to_graphviz``.
        """
        from dask.dot import graphviz_to_file

        g = self._to_graphviz(**kwargs)
        graphviz_to_file(g, filename, format)
        return g

    def find_operations(self, operation: type) -> Generator[Expr]:
        """Search the expression graph for a specific operation type

        Parameters
        ----------
        operation
            The operation type to search for.

        Returns
        -------
        nodes
            Generator of `operation` instances. Ordering corresponds
            to a depth-first search of the expression graph.
        """

        assert issubclass(operation, Expr), "`operation` must be `Expr` subclass"
        stack = [self]
        seen = set()
        while stack:
            node = stack.pop()
            if node._name in seen:
                continue
            seen.add(node._name)

            for dep in node.dependencies():
                stack.append(dep)

            if isinstance(node, operation):
                yield node


@normalize_token.register(Expr)
def normalize_expression(expr):
    return expr._name
