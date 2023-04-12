import functools
import numbers
import operator
from collections import defaultdict
from collections.abc import Iterator

import pandas as pd
import toolz
from dask.base import normalize_token, tokenize
from dask.core import ishashable
from dask.dataframe import methods
from dask.optimization import SubgraphCallable
from dask.utils import M, apply, funcname
from matchpy import (
    Arity,
    CustomConstraint,
    Operation,
    Pattern,
    ReplacementRule,
    Wildcard,
    replace_all,
)
from matchpy.expressions.expressions import _OperationMeta

replacement_rules = []


class _ExprMeta(_OperationMeta):
    """Metaclass to determine Operation behavior

    Matchpy overrides `__call__` so that `__init__` doesn't behave as expected.
    This is gross, but has some logic behind it.  We need to enforce that
    expressions can be easily replayed.  Eliminating `__init__` is one way to
    do this.  It forces us to compute things lazily rather than at
    initialization.

    We motidify Matchpy's implementation so that we can handle keywords and
    default values more cleanly.

    We also collect replacement rules here.
    """

    seen = set()

    def __call__(cls, *args, variable_name=None, **kwargs):
        # Collect optimization rules for new classes
        if cls not in _ExprMeta.seen:
            _ExprMeta.seen.add(cls)
            for rule in cls._replacement_rules():
                replacement_rules.append(rule)

        # Grab keywords and manage default values
        operands = list(args)
        for parameter in cls._parameters[len(operands) :]:
            operands.append(kwargs.pop(parameter, cls._defaults[parameter]))
        assert not kwargs

        # Defer up to matchpy
        return super().__call__(*operands, variable_name=None)


_defer_to_matchpy = False


class Expr(Operation, metaclass=_ExprMeta):
    """Primary class for all Expressions

    This mostly includes Dask protocols and various Pandas-like method
    definitions to make us look more like a DataFrame.
    """

    commutative = False
    associative = False
    _parameters = []
    _defaults = {}

    @functools.cached_property
    def ndim(self):
        meta = self._meta
        try:
            return meta.ndim
        except AttributeError:
            return 0

    @classmethod
    def _replacement_rules(cls) -> Iterator[ReplacementRule]:
        """Rules associated to this class that are useful for optimization

        See also:
            optimize
            _ExprMeta
        """
        yield from []

    def __str__(self):
        s = ", ".join(
            str(param) + "=" + str(operand)
            for param, operand in zip(self._parameters, self.operands)
            if operand != self._defaults.get(param)
        )
        return f"{type(self).__name__}({s})"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(self._name)

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

    def operand(self, key):
        # Access an operand unambiguously
        # (e.g. if the key is reserved by a method/property)
        return self.operands[type(self)._parameters.index(key)]

    def dependencies(self):
        # Dependencies are `Expr` operands only
        return [operand for operand in self.operands if isinstance(operand, Expr)]

    def simplify(self):
        return self

    @property
    def index(self):
        return ProjectIndex(self)

    @property
    def size(self):
        return Size(self)

    def __setattr__(self, key, value):
        object.__setattr__(self, key, value)

    def __getitem__(self, other):
        if isinstance(other, Expr):
            return Filter(self, other)  # df[df.x > 1]
        else:
            return Projection(self, other)  # df[["a", "b", "c"]]

    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(other, self)

    def __sub__(self, other):
        return Sub(self, other)

    def __rsub__(self, other):
        return Sub(other, self)

    def __mul__(self, other):
        return Mul(self, other)

    def __rmul__(self, other):
        return Mul(other, self)

    def __truediv__(self, other):
        return Div(self, other)

    def __rtruediv__(self, other):
        return Div(other, self)

    def __lt__(self, other):
        return LT(self, other)

    def __rlt__(self, other):
        return LT(other, self)

    def __gt__(self, other):
        return GT(self, other)

    def __rgt__(self, other):
        return GT(other, self)

    def __le__(self, other):
        return LE(self, other)

    def __rle__(self, other):
        return LE(other, self)

    def __ge__(self, other):
        return GE(self, other)

    def __rge__(self, other):
        return GE(other, self)

    def __eq__(self, other):
        if _defer_to_matchpy:  # Defer to matchpy when optimizing
            return Operation.__eq__(self, other)
        else:
            return EQ(other, self)

    def __ne__(self, other):
        if _defer_to_matchpy:  # Defer to matchpy when optimizing
            return Operation.__ne__(self, other)
        else:
            return NE(other, self)

    def sum(self, skipna=True, numeric_only=None, min_count=0):
        return Sum(self, skipna, numeric_only, min_count)

    def mean(self, skipna=True, numeric_only=None, min_count=0):
        return self.sum(skipna=skipna) / self.count()

    def max(self, skipna=True, numeric_only=None, min_count=0):
        return Max(self, skipna, numeric_only, min_count)

    def mode(self, dropna=True):
        return Mode(self, dropna=dropna)

    def min(self, skipna=True, numeric_only=None, min_count=0):
        return Min(self, skipna, numeric_only, min_count)

    def count(self, numeric_only=None):
        return Count(self, numeric_only)

    def astype(self, dtypes):
        return AsType(self, dtypes)

    def apply(self, function, *args, **kwargs):
        return Apply(self, function, args, kwargs)

    @functools.cached_property
    def divisions(self):
        return tuple(self._divisions())

    def _divisions(self):
        raise NotImplementedError()

    @property
    def known_divisions(self):
        """Whether divisions are already known"""
        return len(self.divisions) > 0 and self.divisions[0] is not None

    @property
    def npartitions(self):
        if "npartitions" in self._parameters:
            idx = self._parameters.index("npartitions")
            return self.operands[idx]
        else:
            return len(self.divisions) - 1

    @functools.cached_property
    def _name(self):
        return funcname(type(self)).lower() + "-" + tokenize(*self.operands)

    @property
    def columns(self):
        return self._meta.columns

    @property
    def dtypes(self):
        return self._meta.dtypes

    @property
    def _meta(self):
        raise NotImplementedError()

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

    def substitute(self, substitutions: dict) -> "Expr":
        """Substitute specific `Expr` instances within `self`

        Parameters
        ----------
        substitutions:
            mapping old terms to new terms

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
            if ishashable(operand) and operand in substitutions:
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


class Blockwise(Expr):
    """Super-class for block-wise operations

    This is fairly generic, and includes definitions for `_meta`, `divisions`,
    `_layer` that are often (but not always) correct.  Mostly this helps us
    avoid duplication in the future.
    """

    operation = None

    @functools.cached_property
    def _meta(self):
        return self.operation(
            *[arg._meta if isinstance(arg, Expr) else arg for arg in self.operands]
        )

    @property
    def _kwargs(self):
        return {}

    def _broadcast_dep(self, dep: Expr):
        # Checks if a dependency should be broadcasted to
        # all partitions of this `Blockwise` operation
        return (
            not isinstance(dep, BlockwiseArg)
            and dep.npartitions == 1
            and dep.ndim < self.ndim
        )

    def _divisions(self):
        # This is an issue.  In normal Dask we re-divide everything in a step
        # which combines divisions and graph.
        # We either have to create a new Align layer (ok) or combine divisions
        # and graph into a single operation.
        dependencies = self.dependencies()
        for arg in dependencies:
            if not self._broadcast_dep(arg):
                assert arg.divisions == dependencies[0].divisions
        return dependencies[0].divisions

    @functools.cached_property
    def _name(self):
        return funcname(self.operation) + "-" + tokenize(*self.operands)

    def _blockwise_layer(self):
        args = tuple(
            operand._name if isinstance(operand, Expr) else operand
            for operand in self.operands
        )
        if self._kwargs:
            return {self._name: (apply, self.operation, args, self._kwargs)}
        else:
            return {self._name: (self.operation,) + args}

    def _layer(self):
        # Use BlockwiseArg to broadcast dependencies (if necessary)
        dependencies = [
            BlockwiseArg([(dep._name, 0)] * self.npartitions, dep._name)
            if self._broadcast_dep(dep)
            else dep
            for dep in self.dependencies()
        ]

        # Create SubgraphCallable
        func = SubgraphCallable(
            self._blockwise_layer(),
            self._name,
            [dep._name for dep in dependencies],
            self._name,
        )

        # Tasks depend on external-dependency keys only
        return {
            (self._name, i): (
                func,
                *[
                    dep[i] if isinstance(dep, BlockwiseArg) else (dep._name, i)
                    for dep in dependencies
                ],
            )
            for i in range(self.npartitions)
        }


class BlockwiseArg(Expr):
    """Indexable Blockwise argument

    This class is used by IO expressions to map path-like
    arguments over output partitions in a fusion-compatible way.

    Parameters
    ----------
    lookup: Sequence
        Indexable sequence that should return a task argument
        for a given parition index (e.g. ``[0, npartitions]``).
        Note that ``Blockwise._layer`` will eagerly populate
        leteral partition-dependent task arguments at graph
        creation time using ``BlockwiseArg`` dependencies.
    name: str, optional
        Custom expression name. This operand should be specified
        if ``lookup`` does not produce a deterministic hash.
    """

    _parameters = ["lookup", "name"]
    _defaults = {"name": None}

    @property
    def _meta(self):
        return None

    @functools.cached_property
    def _name(self):
        return self.operand("name") or f"arg-{tokenize(self.lookup)}"

    def _divisions(self):
        return (None,) * (len(self.lookup) + 1)

    def __getitem__(self, index):
        return self.lookup[index]

    def _layer(self):
        # `BlockwiseArg` should never produce a graph.
        # The parent `Blockwise._layer` method should
        # index this object to populate its graph
        # with the values of `BlockwiseArg.lookup`
        return {}


class Elemwise(Blockwise):
    """
    This doesn't really do anything, but we anticipate that future
    optimizations, like `len` will care about which operations preserve length
    """

    pass


class AsType(Elemwise):
    """A good example of writing a trivial blockwise operation"""

    _parameters = ["frame", "dtypes"]
    operation = M.astype


class Apply(Elemwise):
    """A good example of writing a less-trivial blockwise operation"""

    _parameters = ["frame", "function", "args", "kwargs"]
    _defaults = {"args": (), "kwargs": {}}
    operation = M.apply

    @property
    def _meta(self):
        return self.frame._meta.apply(self.function, *self.args, **self.kwargs)

    def _blockwise_layer(self):
        return {
            self._name: (
                apply,
                M.apply,
                [self.frame._name, self.function] + list(self.args),
                self.kwargs,
            )
        }


class Assign(Elemwise):
    """Column Assignment"""

    _parameters = ["frame", "key", "value"]
    operation = methods.assign

    @property
    def _meta(self):
        return self.frame._meta.assign(**{self.key: self.value._meta})

    def _blockwise_layer(self):
        return {
            self._name: (
                methods.assign,
                self.frame._name,
                self.key,
                self.value._name,
            )
        }


class Filter(Blockwise):
    _parameters = ["frame", "predicate"]
    operation = operator.getitem

    @classmethod
    def _replacement_rules(self):
        df = Wildcard.dot("df")
        condition = Wildcard.dot("condition")
        columns = Wildcard.dot("columns")

        # Project columns down through dataframe
        # df[df.x > 1].y -> df.y[df.x > 1]
        yield ReplacementRule(
            Pattern(Filter(df, condition)[columns]),
            lambda df, condition, columns: df[columns][condition],
        )


class Projection(Elemwise):
    """Column Selection"""

    _parameters = ["frame", "columns"]
    operation = operator.getitem

    def _divisions(self):
        return self.frame.divisions

    @property
    def columns(self):
        if isinstance(self.operand("columns"), list):
            return pd.Index(self.operand("columns"))
        else:
            return self.operand("columns")

    @property
    def _meta(self):
        return self.frame._meta[self.columns]

    def __str__(self):
        base = str(self.frame)
        if " " in base:
            base = "(" + base + ")"
        return f"{base}[{repr(self.columns)}]"

    def simplify(self):
        if isinstance(self.frame, Projection):
            # df[a][b]
            a = self.frame.operand("columns")
            b = self.operand("columns")

            if not isinstance(a, list):
                assert a == b
            elif isinstance(b, list):
                assert all(bb in a for bb in b)
            else:
                assert b in a

            return self.frame.frame[b]

    @classmethod
    def _replacement_rules(self):
        df = Wildcard.dot("df")
        a = Wildcard.dot("a")
        b = Wildcard.dot("b")

        # Project columns down through dataframe
        # df[a][b] -> df[b]
        def projection_merge(df, a, b):
            if not isinstance(a, list):
                assert a == b
            elif isinstance(b, list):
                assert all(bb in a for bb in b)
            else:
                assert b in a

            return df[b]

        yield ReplacementRule(
            Pattern(Projection(Projection(df, a), b)), projection_merge
        )


class ProjectIndex(Elemwise):
    """Column Selection"""

    _parameters = ["frame"]
    operation = getattr

    def _divisions(self):
        return self.frame.divisions

    @property
    def _meta(self):
        return self.frame._meta.index

    def _blockwise_layer(self):
        return {self._name: (getattr, self.frame._name, "index")}


class Head(Expr):
    _parameters = ["frame", "n"]
    _defaults = {"n": 5}

    @property
    def _meta(self):
        return self.frame._meta

    def _divisions(self):
        return self.frame.divisions[:2]

    def _layer(self):
        return {
            (self._name, 0): (M.head, (self.frame._name, 0), self.n),
        }

    def simplify(self):
        if isinstance(self.frame, Elemwise):
            operands = [
                Head(op, self.n) if isinstance(op, Expr) else op
                for op in self.frame.operands
            ]
            return type(self.frame)(*operands)

        return self


class Binop(Elemwise):
    _parameters = ["left", "right"]
    arity = Arity.binary

    def __str__(self):
        return f"{self.left} {self._operator_repr} {self.right}"

    @classmethod
    def _replacement_rules(cls):
        left = Wildcard.dot("left")
        right = Wildcard.dot("right")
        columns = Wildcard.dot("columns")

        # Column Projection
        def transform(left, right, columns, cls=None):
            if isinstance(left, Expr):
                left = left[columns]  # TODO: filter just the correct columns

            if isinstance(right, Expr):
                right = right[columns]

            return cls(left, right)

        # (a + b)[c] -> a[c] + b[c]
        yield ReplacementRule(
            Pattern(cls(left, right)[columns]), functools.partial(transform, cls=cls)
        )


class Add(Binop):
    operation = operator.add
    _operator_repr = "+"

    @classmethod
    def _replacement_rules(cls):
        x = Wildcard.dot("x")
        yield ReplacementRule(
            Pattern(Add(x, x)),
            lambda x: Mul(2, x),
        )

        yield from super()._replacement_rules()


class Sub(Binop):
    operation = operator.sub
    _operator_repr = "-"


class Mul(Binop):
    operation = operator.mul
    _operator_repr = "*"

    @classmethod
    def _replacement_rules(cls):
        a, b, c = map(Wildcard.dot, "abc")
        yield ReplacementRule(
            Pattern(
                Mul(a, Mul(b, c)),
                CustomConstraint(
                    lambda a, b, c: isinstance(a, numbers.Number)
                    and isinstance(b, numbers.Number)
                ),
            ),
            lambda a, b, c: Mul(a * b, c),
        )

        yield from super()._replacement_rules()


class Div(Binop):
    operation = operator.truediv
    _operator_repr = "/"


class LT(Binop):
    operation = operator.lt
    _operator_repr = "<"


class LE(Binop):
    operation = operator.le
    _operator_repr = "<="


class GT(Binop):
    operation = operator.gt
    _operator_repr = ">"


class GE(Binop):
    operation = operator.ge
    _operator_repr = ">="


class EQ(Binop):
    operation = operator.eq
    _operator_repr = "=="


class NE(Binop):
    operation = operator.ne
    _operator_repr = "!="


@normalize_token.register(Expr)
def normalize_expression(expr):
    return expr._name


def optimize(expr, fuse=True):
    """High level query optimization

    Today we just use MatchPy's term rewriting system, leveraging the
    replacement rules found in the `replacement_rules` global list .  We continue
    rewriting until nothing changes.  The `replacement_rules` list can be added
    to by anyone, but is mostly populated by the various `_replacement_rules`
    methods on the Expr subclasses.

    Note: matchpy expects `__eq__` and `__ne__` to work in a certain way during
    matching.  This is a bit of a hack, but we turn off our implementations of
    `__eq__` and `__ne__` when this function is running using the
    `_defer_to_matchpy` global.  Please forgive us our sins, as we forgive
    those who sin against us.
    """
    last = None
    global _defer_to_matchpy

    expr, _ = simplify(expr)

    _defer_to_matchpy = True  # take over ==/!= when optimizing
    try:
        while str(expr) != str(last):
            last = expr
            expr = replace_all(expr, replacement_rules)
    finally:
        _defer_to_matchpy = False

    if fuse:
        expr = optimize_blockwise_fusion(expr)

    return expr


def simplify(expr: Expr) -> tuple[Expr, bool]:
    """Simplify expression

    This leverages the ``.simplify`` method defined on each class

    Parameters
    ----------
    expr:
        input expression

    Returns
    -------
    expr:
        output expression
    changed:
        whether or not any change occured
    """
    if not isinstance(expr, Expr):
        return expr, False

    changed_final = False

    while True:
        out = expr.simplify()
        if out is None:
            out = expr
        if out._name == expr._name:
            break
        else:
            changed_final = True
            expr = out

    changed_any = False
    new_operands = []
    for operand in expr.operands:
        new, changed_one = simplify(operand)
        new_operands.append(new)
        changed_any |= changed_one

    if changed_any:
        changed_final = True
        expr = type(expr)(*new_operands)
        expr, _ = simplify(expr)

    return expr, changed_final


## Utilites for Expr fusion


def optimize_blockwise_fusion(expr):
    """Traverse the expression graph and apply fusion"""

    def _fusion_pass(expr):
        # Full pass to find global dependencies
        seen = set()
        stack = [expr]
        dependents = defaultdict(set)
        dependencies = {}
        while stack:
            next = stack.pop()

            if next._name in seen:
                continue
            seen.add(next._name)

            if isinstance(next, Blockwise):
                dependencies[next] = set()
                if next not in dependents:
                    dependents[next] = set()

            for operand in next.operands:
                if isinstance(operand, Expr):
                    stack.append(operand)
                    if isinstance(operand, Blockwise):
                        if next in dependencies:
                            dependencies[next].add(operand)
                        dependents[operand].add(next)

        # Traverse each "root" until we find a fusable sub-group.
        # Here we use root to refer to a Blockwise Expr node that
        # has no Blockwise dependents
        roots = [
            k
            for k, v in dependents.items()
            if v == set() or all(not isinstance(_expr, Blockwise) for _expr in v)
        ]
        while roots:
            root = roots.pop()
            seen = set()
            stack = [root]
            group = []
            while stack:
                next = stack.pop()

                if next._name in seen:
                    continue
                seen.add(next._name)

                group.append(next)
                for dep in dependencies[next]:
                    if not (dependents[dep] - set(stack) - set(group)):
                        # All of deps dependents are contained
                        # in the local group (or the local stack
                        # of expr nodes that we know we will be
                        # adding to the local group)
                        stack.append(dep)
                    elif dep not in roots and dependencies[dep]:
                        # Couldn't fuse dep, but we may be able to
                        # use it as a new root on the next pass
                        roots.append(dep)

            # Replace fusable sub-group
            if len(group) > 1:
                group_deps = []
                local_names = [_expr._name for _expr in group]
                for _expr in group:
                    group_deps += [
                        operand
                        for operand in _expr.dependencies()
                        if operand._name not in local_names
                    ]
                to_replace = {group[0]: Fused(group, *group_deps)}
                return expr.substitute(to_replace), not roots

        # Return original expr if no fusable sub-groups were found
        return expr, True

    while True:
        original_name = expr._name
        expr, done = _fusion_pass(expr)
        if done or expr._name == original_name:
            break

    return expr


class Fused(Blockwise):
    """Fused ``Blockwise`` expression

    A ``Fused`` corresponds to the fusion of multiple
    ``Blockwise`` expressions into a single ``Expr`` object.
    Before graph-materialization time, the behavior of this
    object should be identical to that of the first element
    of ``Fused.exprs`` (i.e. the top-most expression in
    the fused group).

    Parameters
    ----------
    exprs : List[Expr]
        Group of original ``Expr`` objects being fused together.
    *dependencies:
        List of external and ``BlockwiseArg``-based ``Expr``
        dependencies. External-``Expr``dependencies correspond to
        any ``Expr`` operand that is not already included in
        ``exprs``. Note that these dependencies should be defined
        in the order of the ``Expr`` objects that require them
        (in ``exprs``). These dependencies do not include literal
        operands, because those arguments should already be
        captured in the fused subgraphs.
    """

    _parameters = ["exprs"]

    @functools.cached_property
    def _meta(self):
        return self.exprs[0]._meta

    def __str__(self):
        names = [expr._name.split("-")[0] for expr in self.exprs]
        if len(names) > 3:
            names = [names[0], f"{len(names) - 2}", names[-1]]
        descr = "-".join(names)
        return f"Fused-{descr}"

    @functools.cached_property
    def _name(self):
        return f"{str(self)}-{tokenize(self.exprs)}"

    def _divisions(self):
        return self.exprs[0]._divisions()

    def dependencies(self):
        return self.operands[1:]

    def _blockwise_layer(self):
        block = {self._name: self.exprs[0]._name}
        for _expr in self.exprs:
            for k, v in _expr._blockwise_layer().items():
                block[k] = v
        return block


from dask_match.reductions import Count, Max, Min, Mode, Size, Sum
