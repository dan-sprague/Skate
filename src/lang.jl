
import Base.@kwdef

# ═══════════════════════════════════════════════════════════════════════════════
# Known lpdf/lccdf function names (for typo detection)
# ═══════════════════════════════════════════════════════════════════════════════

const _KNOWN_LPDF_NAMES = Set{Symbol}([
    :normal_lpdf, :cauchy_lpdf, :exponential_lpdf, :gamma_lpdf, :beta_lpdf,
    :lognormal_lpdf, :student_t_lpdf, :uniform_lpdf, :laplace_lpdf, :logistic_lpdf,
    :weibull_lpdf, :weibull_logsigma_lpdf, :weibull_lccdf, :weibull_logsigma_lccdf,
    :poisson_lpdf, :binomial_lpdf, :bernoulli_logit_lpdf, :binomial_logit_lpdf,
    :neg_binomial_2_lpdf, :beta_binomial_lpdf, :categorical_logit_lpdf,
    :dirichlet_lpdf, :multi_normal_diag_lpdf, :multi_normal_cholesky_lpdf,
    :multi_normal_cholesky_scaled_lpdf, :lkj_corr_cholesky_lpdf,
    :correlated_topic_lpdf,
])

"""Simple edit distance for typo suggestions (Levenshtein)."""
function _edit_distance(a::AbstractString, b::AbstractString)
    m, n = length(a), length(b)
    d = zeros(Int, m + 1, n + 1)
    for i in 0:m; d[i+1, 1] = i; end
    for j in 0:n; d[1, j+1] = j; end
    for i in 1:m, j in 1:n
        cost = a[i] == b[j] ? 0 : 1
        d[i+1, j+1] = min(d[i, j+1] + 1, d[i+1, j] + 1, d[i, j] + 1 - (1 - cost))
    end
    d[m+1, n+1]
end

"""Find the closest known lpdf name to `name`, or nothing if no close match."""
function _suggest_lpdf(name::Symbol)
    s = string(name)
    best_name = nothing
    best_dist = typemax(Int)
    for known in _KNOWN_LPDF_NAMES
        d = _edit_distance(s, string(known))
        if d < best_dist
            best_dist = d
            best_name = known
        end
    end
    # Only suggest if within ~30% of the name length
    best_dist <= max(2, length(s) ÷ 3) ? best_name : nothing
end

# ═══════════════════════════════════════════════════════════════════════════════
# @logjoint body validation helpers
# ═══════════════════════════════════════════════════════════════════════════════

"""Collect all free symbols referenced in an expression (excluding function call names)."""
function _collect_refs(ex, refs::Set{Symbol})
    if ex isa Symbol
        push!(refs, ex)
    elseif ex isa Expr
        if ex.head == :call
            # skip function name (args[1]), recurse into arguments only
            for a in ex.args[2:end]
                _collect_refs(a, refs)
            end
        elseif ex.head == :. && length(ex.args) == 2 && ex.args[2] isa Expr && ex.args[2].head == :tuple
            # dot-call like f.(args...) — skip function name
            for a in ex.args[2].args
                _collect_refs(a, refs)
            end
        elseif ex.head == :macrocall
            # skip macro name, recurse into rest
            for a in ex.args[2:end]
                _collect_refs(a, refs)
            end
        else
            for a in ex.args
                _collect_refs(a, refs)
            end
        end
    end
end

"""Collect all symbols assigned (LHS of =) in a statement list."""
function _collect_assigns(stmts, assigned::Set{Symbol})
    for s in stmts
        s isa Expr || continue
        if s.head == :(=) && s.args[1] isa Symbol
            push!(assigned, s.args[1])
        elseif s.head == :block
            _collect_assigns(s.args, assigned)
        elseif s.head == :for && length(s.args) >= 2
            # for loop — collect iteration variable and body assigns
            iter = s.args[1]
            if iter isa Expr && iter.head == :(=) && iter.args[1] isa Symbol
                push!(assigned, iter.args[1])
            end
            _collect_assigns(s.args[2:end], assigned)
        end
    end
end

"""Check if an expression is a bare lpdf/lccdf call (not wrapped in target +=)."""
function _is_bare_lpdf_call(ex)
    ex isa Expr || return false
    if ex.head == :call && ex.args[1] isa Symbol
        name = string(ex.args[1])
        return endswith(name, "_lpdf") || endswith(name, "_lccdf") || endswith(name, "_lcdf")
    end
    false
end

"""Collect all function call names in an expression."""
function _collect_call_names(ex, names::Set{Symbol})
    ex isa Expr || return
    if ex.head == :call && ex.args[1] isa Symbol
        push!(names, ex.args[1])
    elseif ex.head == :. && length(ex.args) == 2 && ex.args[2] isa Expr && ex.args[2].head == :tuple
        ex.args[1] isa Symbol && push!(names, ex.args[1])
    end
    for a in ex.args
        _collect_call_names(a, names)
    end
end

"""Replace `var` with `replacement` throughout an expression."""
function _subst_var(ex, var::Symbol, replacement)
    ex isa Symbol && ex == var && return replacement
    ex isa Expr || return ex
    return Expr(ex.head, [_subst_var(a, var, replacement) for a in ex.args]...)
end

"""
Extract (weights, params, body) from a `log_mix` call, or `nothing` if not recognized.
Supports two forms:
  - `log_mix(weights) do j; body; end`   → Expr(:do, ...)
  - `log_mix(weights, j -> body)`        → Expr(:call, ..., Expr(:->,...))
"""
function _extract_log_mix(ex)
    # Form 1: do-block
    if ex.head == :do && length(ex.args) == 2
        call = ex.args[1]
        lambda = ex.args[2]
        if call isa Expr && call.head == :call && call.args[1] == :log_mix &&
           lambda isa Expr && lambda.head == :->
            return call.args[2], lambda.args[1], lambda.args[2]
        end
    end
    # Form 2: arrow argument
    if ex.head == :call && length(ex.args) == 3 && ex.args[1] == :log_mix
        arrow = ex.args[3]
        if arrow isa Expr && arrow.head == :->
            return ex.args[2], arrow.args[1], arrow.args[2]
        end
    end
    return nothing, nothing, nothing
end

"""Check if an expression contains any closure (`->` or `do`) nodes."""
function _has_closure(ex)
    ex isa Expr || return false
    (ex.head == :-> || ex.head == :do) && return true
    return any(_has_closure, ex.args)
end

"""Check if an index expression represents a slice (`:` or a range like `1:n`)."""
_is_slice_index(ex) = ex === :(:) ||
    (ex isa Expr && ex.head == :call && !isempty(ex.args) && ex.args[1] == :(:))

"""
Automatically wrap matrix/array slices (e.g. `x[i, :]`) with `view(...)`.
Existing explicit `@view(...)` calls are preserved as-is.
"""
function _auto_view(ex)
    ex isa Expr || return ex
    # Preserve explicit @view — don't recurse into it
    if ex.head == :macrocall && !isempty(ex.args) && ex.args[1] == Symbol("@view")
        return ex
    end
    # Recurse first
    ex = Expr(ex.head, [_auto_view(a) for a in ex.args]...)
    # Wrap sliced indexing with view()
    if ex.head == :ref && length(ex.args) >= 2 && any(_is_slice_index, ex.args[2:end])
        return Expr(:call, :view, ex.args...)
    end
    return ex
end

"""Rewrite bare data-name symbols in an expression to `data.name`."""
function _rewrite_data_refs(ex, data_names::Set{Symbol}, param_names::Set{Symbol})
    ex isa Symbol && ex ∈ data_names && ex ∉ param_names && return :(data.$ex)
    ex isa Expr || return ex
    # Don't rewrite the LHS of simple assignments (bare symbol),
    # but DO recurse into complex LHS (e.g., ref indexing) so that
    # data dims like K in `name[k in 1:K]` get rewritten.
    if ex.head == :(=)
        lhs = ex.args[1]
        rhs = _rewrite_data_refs(ex.args[2], data_names, param_names)
        if lhs isa Symbol
            return Expr(:(=), lhs, rhs)
        else
            return Expr(:(=), _rewrite_data_refs(lhs, data_names, param_names), rhs)
        end
    end
    return Expr(ex.head, [_rewrite_data_refs(a, data_names, param_names) for a in ex.args]...)
end

function _resolve_size(arg, dn::Set{Symbol})
    arg isa Integer && return arg
    arg isa Symbol  || error("@params: size argument must be an integer literal or a @data name, got: $arg")
    arg ∈ dn        || error("@params: size argument ':$arg' is not declared in @data")
    return :(data.$arg)
end
_lines(b::Expr) = filter(x -> !(x isa LineNumberNode), b.args)

const _SUPPORTED_CONSTANT_TYPES = Set{Symbol}([:Int, :Float64])
const _SUPPORTED_CONSTANT_CONTAINER_TYPES = Set{Symbol}([:Vector, :Matrix])

function _parse_constants(block::Expr)
    fields = Expr[]
    names = Symbol[]
    for line in _lines(block)
        line isa Expr || continue

        if line.head == :(::)
            var = line.args[1]
            typespec = line.args[2]
            var isa Symbol || error(
                "@constants: expected 'name::Type', got '$(line)'. " *
                "Each entry must be a simple name with a type annotation.")
            # Validate the type
            if typespec isa Symbol && typespec ∉ _SUPPORTED_CONSTANT_TYPES
                error("@constants: unsupported type '$typespec' for '$var'. " *
                      "Supported types: Int, Float64, Vector{Float64}, Vector{Int}, Matrix{Float64}")
            end
            push!(fields, :($var::$(_dsl_to_julia_type(typespec))))
            push!(names, var)

        elseif line.head == :(=)
            # Common mistake: trying to assign a value
            error("@constants: got '$line' — use 'name::Type' to declare data types. " *
                  "Assign values when constructing the Data struct, e.g. ModelData($(line.args[1])=...)")
        elseif line.head == :call || (line.head == :ref)
            error("@constants: unexpected expression '$(line)'. " *
                  "Each entry must be 'name::Type' (e.g. N::Int, y::Vector{Float64})")
        else
            # Bare symbol without type annotation
            if line isa Symbol
                error("@constants: '$line' is missing a type annotation. " *
                      "Use '$line::Type' (e.g. $line::Int or $line::Vector{Float64})")
            end
            error("@constants: unexpected expression '$(line)'. " *
                  "Each entry must be 'name::Type' (e.g. N::Int, y::Vector{Float64})")
        end
    end
    fields, names
end

function _dsl_to_julia_type(spec)
    spec isa Symbol && return spec
    spec isa Expr && spec.head == :curly && return spec
    spec isa Expr && spec.head == :call && return spec.args[1]
    return spec
end

function _make_constraint_expr(lo, hi)
    isnothing(lo) && isnothing(hi)  && return :(IdentityConstraint())
    !isnothing(lo) && isnothing(hi) && return :(LowerBounded($lo))
    isnothing(lo) && !isnothing(hi) && return :(UpperBounded($hi))
    return :(Bounded($lo, $hi))
end

struct _ParamSpec
    name::Symbol
    constraint_expr::Expr   # e.g. :(LowerBounded(0.0))
    container::Symbol       # :scalar | :vector | :simplex | :ordered | :matrix
    sizes::Vector{Any}
    ordered_dim::Int        # matrix only: 0 = none, N>0 = column N ordered, -1 = row simplex, -2 = row simplex + ordered col 1
end

function _parse_params(block::Expr, data_names::Set{Symbol})
    specs = _ParamSpec[]
    seen_names = Set{Symbol}()
    for line in _lines(block)
        line isa Expr || continue

        if line.head == :(::)
            var = line.args[1]
            T = line.args[2]
            var isa Symbol || error(
                "@params: '$var::$T' — bare annotations only support Float64. " *
                "For vectors and matrices use param(Vector{Float64}, n, ...)")
            var ∈ seen_names && error(
                "@params: duplicate parameter name '$var'. Each parameter must have a unique name.")
            var ∈ data_names && error(
                "@params: parameter '$var' shadows a @constants name. " *
                "Use a different name to avoid ambiguity.")
            push!(seen_names, var)
            push!(specs, _ParamSpec(var, :(IdentityConstraint()), :scalar, [], 0))

        elseif line.head == :(=)
            var = line.args[1]
            rhs = line.args[2]
            rhs isa Expr && rhs.head == :call && rhs.args[1] == :param || error(
                "@params: '$var = $rhs' — expected a call to param(...)")
            var isa Symbol && var ∈ seen_names && error(
                "@params: duplicate parameter name '$var'. Each parameter must have a unique name.")
            var isa Symbol && var ∈ data_names && error(
                "@params: parameter '$var' shadows a @constants name. " *
                "Use a different name to avoid ambiguity.")
            var isa Symbol && push!(seen_names, var)
            push!(specs, _param_to_spec(var, rhs.args[2:end], data_names))
        end
    end
    specs
end

"""Convert `param(T, sizes...; lower=…, upper=…)` to _ParamSpec."""
function _param_to_spec(name::Symbol, args, dn::Set{Symbol})
    isempty(args) && error("@params: param() requires a type as its first argument")

    kw_args = Expr[]
    positional = Any[]
    for a in args
        if a isa Expr && a.head == :parameters
            append!(kw_args, a.args)
        elseif a isa Expr && a.head == :kw
            push!(kw_args, a)
        else
            push!(positional, a)
        end
    end

    isempty(positional) && error("@params: param() requires a type as its first argument")
    T       = positional[1]
    sz_args = positional[2:end]

    lo = hi = nothing
    is_simplex = false
    is_ordered = false
    for a in kw_args
        a.args[1] == :lower   && (lo = a.args[2])
        a.args[1] == :upper   && (hi = a.args[2])
        a.args[1] == :simplex && (is_simplex = a.args[2])
        a.args[1] == :ordered && (is_ordered = a.args[2])
    end

    constraint = _make_constraint_expr(lo, hi)

    if T == :Float64
        is_simplex && error("@params: simplex not supported for scalars")
        isempty(sz_args) || error("@params: param(Float64) takes no positional size arguments")
        return _ParamSpec(name, constraint, :scalar, [], 0)
    end

    if T isa Expr && T.head == :curly
        base = T.args[1]
        elem = T.args[2]
        if base == :Vector && elem == :Float64
            length(sz_args) == 1 || error("@params: param(Vector{Float64}, n) takes one size argument")
            n = _resolve_size(sz_args[1], dn)
            if is_simplex
                (lo !== nothing || hi !== nothing) &&
                    error("@params: simplex params cannot have bounds")
                return _ParamSpec(name, :(SimplexConstraint()), :simplex, [n], 0)
            end
            if is_ordered
                (lo !== nothing || hi !== nothing) &&
                    error("@params: ordered params cannot have bounds")
                return _ParamSpec(name, :(OrderedConstraint()), :ordered, [n], 0)
            end
            return _ParamSpec(name, constraint, :vector, [n], 0)
        elseif base == :Matrix && elem == :Float64
            length(sz_args) == 2 || error("@params: param(Matrix{Float64}, n, m) takes two size arguments")
            sz1 = _resolve_size(sz_args[1], dn)
            sz2 = _resolve_size(sz_args[2], dn)
            if is_simplex
                (lo !== nothing || hi !== nothing) &&
                    error("@params: simplex matrix params cannot have bounds")
                if is_ordered !== false
                    # simplex rows + ordered column 1 (only column 1 supported)
                    ordered_col = is_ordered === true ? 1 : Int(is_ordered)
                    ordered_col == 1 ||
                        error("@params: simplex + ordered only supports column 1 (stick-breaking monotonicity)")
                    return _ParamSpec(name, :(SimplexConstraint()), :matrix, [sz1, sz2], -2)
                end
                return _ParamSpec(name, :(SimplexConstraint()), :matrix, [sz1, sz2], -1)
            end
            ordered_col = if is_ordered === false
                0
            elseif is_ordered === true
                1
            elseif is_ordered isa Integer
                Int(is_ordered)
            else
                error("@params: ordered must be true or a column index (integer)")
            end
            if ordered_col > 0 && (lo !== nothing || hi !== nothing)
                error("@params: ordered matrix columns cannot have bounds")
            end
            if ordered_col == 0
                constraint != :(IdentityConstraint()) && error("@params: bounded matrices not supported")
            end
            return _ParamSpec(name, constraint, :matrix, [sz1, sz2], ordered_col)
        end
    end

    if T == :CholCorr
        (lo !== nothing || hi !== nothing) &&
            error("@params: CholCorr params cannot have bounds")
        is_simplex && error("@params: simplex not supported for CholCorr")
        is_ordered !== false && error("@params: ordered not supported for CholCorr")

        if length(sz_args) == 1
            D = _resolve_size(sz_args[1], dn)
            return _ParamSpec(name, :(IdentityConstraint()), :chol_corr, [D], 0)
        elseif length(sz_args) == 2
            K = _resolve_size(sz_args[1], dn)
            D = _resolve_size(sz_args[2], dn)
            return _ParamSpec(name, :(IdentityConstraint()), :chol_corr_batch, [K, D], 0)
        else
            error("@params: param(CholCorr, ...) takes 1 (D) or 2 (K, D) size arguments")
        end
    end

    error("@params: unsupported type in param(): $T")
end

# ═══════════════════════════════════════════════════════════════════════════════
# @for broadcast-to-loop unrolling (CPU only — XLA prefers broadcasts)
# ═══════════════════════════════════════════════════════════════════════════════

@enum _ShapeKind _shape_scalar _shape_vector _shape_matrix _shape_unknown

struct _ShapeInfo
    kind::_ShapeKind
    len::Any        # vector length expr, or nothing
    ncols::Any      # matrix col count, or nothing
end

_scalar_shape() = _ShapeInfo(_shape_scalar, nothing, nothing)
_vector_shape(len) = _ShapeInfo(_shape_vector, len, nothing)
_matrix_shape(nrows, ncols) = _ShapeInfo(_shape_matrix, nrows, ncols)
_unknown_shape() = _ShapeInfo(_shape_unknown, nothing, nothing)

"""Build initial shape environment from @params and @constants declarations."""
function _build_shape_env(param_specs, data_fields)
    env = Dict{Any, _ShapeInfo}()
    for s in param_specs
        if s.container == :scalar
            env[s.name] = _scalar_shape()
        elseif s.container in (:vector, :simplex, :ordered)
            env[s.name] = _vector_shape(s.sizes[1])
        elseif s.container == :matrix
            env[s.name] = _matrix_shape(s.sizes[1], s.sizes[2])
        end
    end
    for f in data_fields
        f isa Expr && f.head == :(::) || continue
        var = f.args[1]
        T = f.args[2]
        dkey = :(data.$var)
        if T isa Expr && T.head == :curly
            base = T.args[1]
            if base == :Vector
                env[dkey] = _vector_shape(:(length(data.$var)))
            elseif base == :Matrix
                env[dkey] = _matrix_shape(:(size(data.$var, 1)), :(size(data.$var, 2)))
            else
                env[dkey] = _scalar_shape()
            end
        else
            env[dkey] = _scalar_shape()
        end
    end
    env
end

const _DOT_OPS = Set(Symbol[Symbol(".+"), Symbol(".-"), Symbol(".*"), Symbol("./")])

_is_dot_op(s) = s isa Symbol && s in _DOT_OPS

function _undot(op::Symbol)
    op == Symbol(".+") && return :+
    op == Symbol(".-") && return :-
    op == Symbol(".*") && return :*
    op == Symbol("./") && return :/
    error("Unknown dot operator: $op")
end

"""Check if ex is a data.X expression."""
function _is_data_ref(ex)
    ex isa Expr && ex.head == :. && length(ex.args) == 2 &&
        ex.args[1] == :data && ex.args[2] isa QuoteNode
end

"""Get the :(data.X) key for env lookup from a data reference expression."""
function _data_key(ex)
    ex isa Expr && ex.head == :. && ex.args[1] == :data && ex.args[2] isa QuoteNode &&
        return Expr(:., :data, ex.args[2])
    return ex
end

"""Check if an indexing expression is a matrix column slice like X[:, 1:k]."""
function _is_mat_col_slice(ex, env)
    ex isa Expr && ex.head == :ref && length(ex.args) == 3 || return false
    base_shape = _infer_shape(ex.args[1], env)
    base_shape.kind == _shape_matrix || return false
    _is_slice_index(ex.args[2]) || return false
    return true
end

"""Extract (base, col_start, col_stop) from M[:, start:stop]."""
function _mat_slice_parts(ex)
    base = ex.args[1]
    col_idx = ex.args[3]
    if col_idx isa Expr && col_idx.head == :call && col_idx.args[1] == :(:)
        return base, col_idx.args[2], col_idx.args[3]
    end
    return base, col_idx, col_idx
end

"""Recursive shape inference for expressions.

Returns `_unknown_shape()` when the shape cannot be determined (e.g. undefined symbol).
Returns `_scalar_shape()` only when the expression is *known* to be scalar.
This distinction allows `@for` to error on ambiguous shapes rather than silently
treating unknown expressions as scalars.
"""
function _infer_shape(ex, env::Dict{Any, _ShapeInfo})
    ex isa Number && return _scalar_shape()

    if ex isa Symbol
        haskey(env, ex) && return env[ex]
        # Unknown symbol — could be a local variable, Julia builtin, etc.
        return _unknown_shape()
    end

    ex isa Expr || return _unknown_shape()

    if _is_data_ref(ex)
        key = _data_key(ex)
        haskey(env, key) && return env[key]
        return _unknown_shape()
    end

    # Dot-call: f.(args...) — always broadcasts to vector if any arg is vector
    if ex.head == :. && length(ex.args) == 2 &&
       ex.args[2] isa Expr && ex.args[2].head == :tuple
        for a in ex.args[2].args
            s = _infer_shape(a, env)
            s.kind == _shape_vector && return s
        end
        # All args are scalar or unknown — if all known scalar, return scalar
        all_known = all(ex.args[2].args) do a
            s = _infer_shape(a, env)
            s.kind == _shape_scalar
        end
        return all_known ? _scalar_shape() : _unknown_shape()
    end

    if ex.head == :call
        op = ex.args[1]
        operands = ex.args[2:end]

        if _is_dot_op(op)
            for a in operands
                s = _infer_shape(a, env)
                s.kind == _shape_vector && return s
            end
            all_known = all(operands) do a
                s = _infer_shape(a, env)
                s.kind == _shape_scalar
            end
            return all_known ? _scalar_shape() : _unknown_shape()
        end

        if op == :* && length(operands) == 2
            s1 = _infer_shape(operands[1], env)
            s2 = _infer_shape(operands[2], env)
            if s1.kind == _shape_matrix && s2.kind == _shape_vector
                return _vector_shape(s1.len)
            end
            if _is_mat_col_slice(operands[1], env) && s2.kind == _shape_vector
                base = operands[1].args[1]
                base_shape = _infer_shape(base, env)
                return _vector_shape(base_shape.len)
            end
            if s1.kind == _shape_vector return s1 end
            if s2.kind == _shape_vector return s2 end
            # Both scalar or both unknown — known scalar only if both known scalar
            if s1.kind == _shape_scalar && s2.kind == _shape_scalar
                return _scalar_shape()
            end
            return _unknown_shape()
        end

        # sum() always returns scalar
        op == :sum && return _scalar_shape()

        # Other function calls: if any arg is vector, result is vector (broadcast semantics)
        for a in operands
            s = _infer_shape(a, env)
            s.kind == _shape_vector && return s
        end
        # Known scalar only if all operands are known scalar
        all_known = all(operands) do a
            s = _infer_shape(a, env)
            s.kind == _shape_scalar
        end
        return all_known ? _scalar_shape() : _unknown_shape()
    end

    if ex.head == :ref
        base = ex.args[1]
        base_shape = _infer_shape(base, env)
        indices = ex.args[2:end]

        if base_shape.kind == _shape_vector && length(indices) == 1
            idx_shape = _infer_shape(indices[1], env)
            if idx_shape.kind == _shape_vector
                return _vector_shape(idx_shape.len)
            end
            return _scalar_shape()  # scalar index into known vector = scalar
        end

        if base_shape.kind == _shape_matrix && length(indices) == 2
            if _is_slice_index(indices[1])
                return _matrix_shape(base_shape.len, nothing)
            end
            # Single element of a matrix is scalar
            return _scalar_shape()
        end
    end

    return _unknown_shape()
end

"""Like _infer_shape but treats unknown as scalar (for non-@for contexts).

Outside of @for, unknown shapes are safe to treat as scalar because
they are just regular Julia expressions. This is used by _expand_for_block
to decide which statements need loop expansion vs. which are scalar."""
function _infer_shape_permissive(ex, env::Dict{Any, _ShapeInfo})
    s = _infer_shape(ex, env)
    s.kind == _shape_unknown ? _scalar_shape() : s
end

"""Find symbols in an expression whose shape is unknown (not in env)."""
function _find_unknown_symbols(ex, env::Dict{Any, _ShapeInfo})
    unknowns = Symbol[]
    _find_unknown_symbols!(ex, env, unknowns)
    unique(unknowns)
end

function _find_unknown_symbols!(ex, env, unknowns::Vector{Symbol})
    if ex isa Symbol
        if !haskey(env, ex) && ex ∉ (:+, :-, :*, :/, :^, :log, :exp, :sqrt, :abs,
                                      :log1p, :max, :min, :clamp, :sum, :length, :eps)
            push!(unknowns, ex)
        end
        return
    end
    if ex isa Expr
        if _is_data_ref(ex)
            key = _data_key(ex)
            if !haskey(env, key)
                # Extract the variable name from data.X
                push!(unknowns, ex.args[2].value)
            end
            return
        end
        if ex.head == :call
            # Skip function name, recurse args
            for a in ex.args[2:end]
                _find_unknown_symbols!(a, env, unknowns)
            end
        elseif ex.head == :. && length(ex.args) == 2 && ex.args[2] isa Expr && ex.args[2].head == :tuple
            # Dot-call: skip function name
            for a in ex.args[2].args
                _find_unknown_symbols!(a, env, unknowns)
            end
        else
            for a in ex.args
                _find_unknown_symbols!(a, env, unknowns)
            end
        end
    end
end

"""Core: scalarize a broadcast expression for loop index `idx`."""
function _scalarize(ex, idx::Symbol, env::Dict{Any, _ShapeInfo}, preamble::Vector{Expr})
    shape = _infer_shape(ex, env)

    if shape.kind == _shape_scalar || shape.kind == _shape_unknown
        # Unknown shapes are treated as scalar pass-through in scalarize.
        # The validation happens at the @for expansion level, not here.
        return ex
    end

    if ex isa Symbol && shape.kind == _shape_vector
        return :($ex[$idx])
    end

    ex isa Expr || return ex

    if _is_data_ref(ex) && shape.kind == _shape_vector
        return :($ex[$idx])
    end

    if ex.head == :call && _is_dot_op(ex.args[1])
        scalar_op = _undot(ex.args[1])
        s_args = [_scalarize(a, idx, env, preamble) for a in ex.args[2:end]]
        return Expr(:call, scalar_op, s_args...)
    end

    if ex.head == :. && length(ex.args) == 2 &&
       ex.args[2] isa Expr && ex.args[2].head == :tuple
        f = ex.args[1]
        s_args = [_scalarize(a, idx, env, preamble) for a in ex.args[2].args]
        return Expr(:call, f, s_args...)
    end

    if ex.head == :call && ex.args[1] == :* && length(ex.args) == 3
        mat_ex, vec_ex = ex.args[2], ex.args[3]

        if _is_mat_col_slice(mat_ex, env)
            mat_base, col_start, col_stop = _mat_slice_parts(mat_ex)
            dot_var = gensym(:dot)
            j_var = gensym(:j)
            push!(preamble, quote
                $dot_var = 0.0
                for $j_var in $col_start:$col_stop
                    $dot_var += $mat_base[$idx, $j_var] * $vec_ex[$j_var]
                end
            end)
            return dot_var
        end

        mat_shape = _infer_shape(mat_ex, env)
        vec_shape = _infer_shape(vec_ex, env)
        if mat_shape.kind == _shape_matrix && vec_shape.kind == _shape_vector
            dot_var = gensym(:dot)
            j_var = gensym(:j)
            ncols = mat_shape.ncols
            push!(preamble, quote
                $dot_var = 0.0
                for $j_var in 1:$ncols
                    $dot_var += $mat_ex[$idx, $j_var] * $vec_ex[$j_var]
                end
            end)
            return dot_var
        end
    end

    if ex.head == :ref && length(ex.args) == 2
        base = ex.args[1]
        index = ex.args[2]
        base_shape = _infer_shape(base, env)
        idx_shape = _infer_shape(index, env)
        if base_shape.kind == _shape_vector && idx_shape.kind == _shape_vector
            return :($base[$index[$idx]])
        end
    end

    new_args = [_scalarize(a, idx, env, preamble) for a in ex.args]
    return Expr(ex.head, new_args...)
end

"""
    _hoist_matvec(ex, env, hoisted) → new_ex

Walk `ex` and replace any `Matrix * Vector` subexpression with a fresh symbol.
Pushes `(sym, mat_expr, vec_expr)` tuples onto `hoisted` for each replacement.
The caller emits `sym = mat_expr * vec_expr` before the loop (BLAS path).
"""
function _hoist_matvec(ex, env, hoisted::Vector{Tuple{Symbol, Any, Any}})
    ex isa Expr || return ex
    # Match  *(A, b)  where A is matrix, b is vector
    if ex.head == :call && ex.args[1] == :* && length(ex.args) == 3
        mat_ex, vec_ex = ex.args[2], ex.args[3]
        mat_shape = _infer_shape(mat_ex, env)
        vec_shape = _infer_shape(vec_ex, env)
        if mat_shape.kind == _shape_matrix && vec_shape.kind == _shape_vector
            sym = gensym(:mv)
            push!(hoisted, (sym, mat_ex, vec_ex))
            return sym  # replaced — caller will register sym as vector in env
        end
        # Also check if it's a column-slice mat * vec
        if _is_mat_col_slice(mat_ex, env)
            sym = gensym(:mv)
            push!(hoisted, (sym, mat_ex, vec_ex))
            return sym
        end
    end
    new_args = [_hoist_matvec(a, env, hoisted) for a in ex.args]
    return Expr(ex.head, new_args...)
end

"""Expand a single @for assignment: `@for y = broadcast_expr`."""
function _expand_for_assign(stmt, env)
    lhs = stmt.args[1]
    rhs = stmt.args[2]
    shape = _infer_shape(rhs, env)

    if shape.kind == _shape_unknown
        # Collect which subexpressions have unknown shape
        unknowns = _find_unknown_symbols(rhs, env)
        error("@for: cannot determine shape of '$lhs = $rhs'. " *
              "Unknown variables: $(join(unknowns, ", ")). " *
              "All variables in @for must be declared in @constants, @params, " *
              "or a preceding @for/@let block.")
    end
    if shape.kind == _shape_scalar
        error("@for: RHS of '$lhs = $rhs' inferred as scalar, not a vector. " *
              "@for only expands broadcast (vectorized) expressions. " *
              "Use a plain assignment instead.")
    end
    len_expr = shape.len

    # Hoist matrix-vector products out of the loop (use BLAS instead of scalar dot)
    hoisted = Tuple{Symbol, Any, Any}[]
    rhs = _hoist_matvec(rhs, env, hoisted)
    pre_loop = Expr[]
    for (sym, mat_ex, vec_ex) in hoisted
        push!(pre_loop, :($sym = $mat_ex * $vec_ex))
        env[sym] = _vector_shape(len_expr)
    end

    idx = gensym(:i)
    preamble = Expr[]
    body = _scalarize(rhs, idx, env, preamble)

    env[lhs] = _vector_shape(len_expr)

    return quote
        $(pre_loop...)
        $lhs = Vector{Float64}(undef, $len_expr)
        @inbounds @simd for $idx in 1:$len_expr
            $(preamble...)
            $lhs[$idx] = $body
        end
    end
end

"""Expand a fused @for block: multiple assignments in one loop."""
function _expand_for_block(block, env)
    stmts = _lines(block)
    isempty(stmts) && return block

    # First pass: infer shapes incrementally (each LHS feeds into subsequent RHS)
    # Use a temporary env copy so we can add LHS shapes as we go.
    tmp_env = copy(env)
    len_expr = nothing
    unknown_stmts = Tuple{Any,Any,Vector{Symbol}}[]
    for s in stmts
        s isa Expr && s.head == :(=) || continue
        shape = _infer_shape(s.args[2], tmp_env)
        if shape.kind == _shape_vector && shape.len !== nothing
            len_expr === nothing && (len_expr = shape.len)
            # Register this LHS as a vector so subsequent RHS expressions can see it
            tmp_env[s.args[1]] = _vector_shape(shape.len)
        elseif shape.kind == _shape_unknown
            unknowns = _find_unknown_symbols(s.args[2], tmp_env)
            push!(unknown_stmts, (s.args[1], s.args[2], unknowns))
        end
    end

    if !isempty(unknown_stmts)
        msgs = ["  $(u[1]) = $(u[2])  (unknown: $(join(u[3], ", ")))" for u in unknown_stmts]
        error("@for block: cannot determine shape of some expressions:\n" *
              join(msgs, "\n") * "\n" *
              "All variables in @for must be declared in @constants, @params, " *
              "or a preceding @for/@let block.")
    end

    if len_expr === nothing
        lhs_names = [s.args[1] for s in stmts if s isa Expr && s.head == :(=)]
        rhs_exprs = [s.args[2] for s in stmts if s isa Expr && s.head == :(=)]
        error("@for block: could not infer loop dimension. " *
              "All RHS expressions inferred as scalar, not vector.\n" *
              "  LHS variables: $(join(lhs_names, ", "))\n" *
              "  RHS expressions: $(join(rhs_exprs, "; "))\n" *
              "Hint: at least one operand must be a declared Vector from @constants or @params, " *
              "or a broadcast (.+, .-, .*, ./) over one.")
    end

    idx = gensym(:i)
    allocs = Expr[]
    loop_body = Expr[]

    # Hoist matrix-vector products from all RHS expressions
    hoisted = Tuple{Symbol, Any, Any}[]
    hoisted_stmts = Expr[]
    for s in stmts
        s isa Expr && s.head == :(=) || continue
        lhs = s.args[1]
        new_rhs = _hoist_matvec(s.args[2], env, hoisted)
        s = :($lhs = $new_rhs)

        shape = _infer_shape(s.args[2], env)
        if shape.kind == _shape_vector
            push!(allocs, :($lhs = Vector{Float64}(undef, $len_expr)))
            preamble = Expr[]
            body = _scalarize(s.args[2], idx, env, preamble)
            append!(loop_body, preamble)
            push!(loop_body, :($lhs[$idx] = $body))
            env[lhs] = _vector_shape(len_expr)
        else
            push!(loop_body, s)
        end
    end

    pre_loop = Expr[]
    for (sym, mat_ex, vec_ex) in hoisted
        push!(pre_loop, :($sym = $mat_ex * $vec_ex))
        env[sym] = _vector_shape(len_expr)
    end

    return quote
        $(pre_loop...)
        $(allocs...)
        @inbounds @simd for $idx in 1:$len_expr
            $(loop_body...)
        end
    end
end

"""Expand `@for target += sum(broadcast_expr)`."""
function _expand_for_sum(stmt, env)
    rhs = stmt.args[2]
    inner = rhs.args[2]

    shape = _infer_shape(inner, env)

    if shape.kind == _shape_unknown
        unknowns = _find_unknown_symbols(inner, env)
        error("@for: cannot determine shape of 'target += sum($inner)'. " *
              "Unknown variables: $(join(unknowns, ", ")). " *
              "All variables in @for must be declared in @constants, @params, " *
              "or a preceding @for/@let block.")
    end
    if shape.kind != _shape_vector
        error("@for: 'sum($inner)' does not contain a vector expression. " *
              "@for target += sum(...) requires a broadcast expression inside sum().")
    end
    len_expr = shape.len

    # Hoist matrix-vector products out of the loop
    hoisted = Tuple{Symbol, Any, Any}[]
    inner = _hoist_matvec(inner, env, hoisted)
    pre_loop = Expr[]
    for (sym, mat_ex, vec_ex) in hoisted
        push!(pre_loop, :($sym = $mat_ex * $vec_ex))
        env[sym] = _vector_shape(len_expr)
    end

    idx = gensym(:i)
    preamble = Expr[]
    body = _scalarize(inner, idx, env, preamble)

    return quote
        $(pre_loop...)
        @inbounds @simd for $idx in 1:$len_expr
            $(preamble...)
            target += $body
        end
    end
end

"""Check if an expression is `target += sum(vector_expr)`."""
function _is_target_sum(ex, env)
    ex isa Expr || return false
    ex.head == :(+=) || return false
    ex.args[1] == :target || return false
    rhs = ex.args[2]
    rhs isa Expr && rhs.head == :call && rhs.args[1] == :sum || return false
    inner_shape = _infer_shape(rhs.args[2], env)
    return inner_shape.kind == _shape_vector
end

"""
    @let name[k in 1:K] = expr

Precompute a vector of cached values in the @logjoint body.
Expands to `name = Vector{Float64}(undef, K); @inbounds for k in 1:K; name[k] = expr; end`.
The resulting `name` can be indexed as `name[k]` in subsequent statements.
"""
function _expand_let(body, env)
    # body is:  :(name[k in lo:hi] = rhs)
    lhs = body.args[1]   # :(name[k in lo:hi])
    rhs = body.args[2]   # the expression

    name = lhs.args[1]
    in_expr = lhs.args[2]  # :(call(in, k, lo:hi))
    in_expr isa Expr && in_expr.head == :call && in_expr.args[1] == :in ||
        error("@let: expected name[iter in range] = expr, got: $body")

    iter_var = in_expr.args[2]
    range_expr = in_expr.args[3]

    # extract length from range
    if range_expr isa Expr && range_expr.head == :call && range_expr.args[1] == :(:)
        lo = range_expr.args[2]
        hi = range_expr.args[3]
    else
        error("@let: expected explicit lo:hi range, got: $range_expr")
    end

    len_expr = lo == 1 ? hi : :($hi - $lo + 1)
    env[name] = _vector_shape(len_expr)

    return quote
        $name = Vector{Float64}(undef, $len_expr)
        @inbounds for $iter_var in $lo:$hi
            $name[$iter_var] = $rhs
        end
    end
end

"""Walk statement list, expand @for and @let annotations, maintain shape env."""
function _expand_for_annotations(stmts, param_specs, data_fields)
    env = _build_shape_env(param_specs, data_fields)
    output = Expr[]

    for s in stmts
        s isa Expr || (push!(output, s); continue)

        if s.head == :macrocall && !isempty(s.args) && s.args[1] == Symbol("@for")
            body_args = filter(a -> !(a isa LineNumberNode), s.args[2:end])
            isempty(body_args) && (push!(output, s); continue)
            body = body_args[1]

            if body isa Expr && body.head == :block
                push!(output, _expand_for_block(body, env))
            elseif body isa Expr && body.head == :(=)
                push!(output, _expand_for_assign(body, env))
            elseif body isa Expr && body.head == :(+=) && _is_target_sum(body, env)
                push!(output, _expand_for_sum(body, env))
            else
                push!(output, body)
            end

        elseif s.head == :macrocall && !isempty(s.args) && s.args[1] == Symbol("@let")
            body_args = filter(a -> !(a isa LineNumberNode), s.args[2:end])
            isempty(body_args) && (push!(output, s); continue)
            body = body_args[1]
            body isa Expr && body.head == :(=) ||
                error("@let: expected @let name[k in 1:K] = expr")
            push!(output, _expand_let(body, env))

        else
            push!(output, s)
            if s.head == :(=) && s.args[1] isa Symbol
                env[s.args[1]] = _infer_shape_permissive(s.args[2], env)
            end
        end
    end
    output
end

# ═══════════════════════════════════════════════════════════════════════════════
# CPU log_mix inlining (if/else + log1p version)
# ═══════════════════════════════════════════════════════════════════════════════

"""
Inline `log_mix` calls into closure-free log-sum-exp loops.
CPU version: uses if/else + log1p for numerical stability.
"""
function _inline_log_mix(ex)
    ex isa Expr || return ex
    ex = Expr(ex.head, [_inline_log_mix(a) for a in ex.args]...)

    weights, params, body = _extract_log_mix(ex)
    weights === nothing && return ex

    j = if params isa Symbol
        params
    elseif params isa Expr && params.head == :tuple && length(params.args) == 1
        params.args[1]
    else
        return ex
    end

    acc = gensym(:lse_acc)
    lp  = gensym(:lse_lp)
    jj  = gensym(:lse_j)
    body_1  = _subst_var(body, j, 1)
    body_jj = _subst_var(body, j, jj)

    return quote
        $acc = log($weights[1]) + $body_1
        for $jj in 2:length($weights)
            $lp = log($weights[$jj]) + $body_jj
            if $lp > $acc
                $acc = $lp + log1p(exp($acc - $lp))
            else
                $acc = $acc + log1p(exp($lp - $acc))
            end
        end
        $acc
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Compile-time arithmetic helpers (constant-fold when both args are Int)
# ═══════════════════════════════════════════════════════════════════════════════

_add(a::Int, b::Int) = a + b
_add(a, b) = :($a + $b)
_sub(a::Int, b::Int) = a - b
_sub(a, b) = :($a - $b)
_mul(a::Int, b::Int) = a * b
_mul(a, b) = :($a * $b)
_div(a::Int, b::Int) = div(a, b)
_div(a, b) = :(div($a, $b))

# ═══════════════════════════════════════════════════════════════════════════════
# @skate macro — CPU codegen
# ═══════════════════════════════════════════════════════════════════════════════

"""Recursively check if an expression contains `target +=`."""
function _contains_target_accum(ex)
    ex isa Expr || return false
    (ex.head == :(+=) && length(ex.args) >= 1 && ex.args[1] == :target) && return true
    return any(_contains_target_accum, ex.args)
end

"""Warn if a statement is a bare lpdf call (result not accumulated into target)."""
function _warn_bare_lpdf(ex, model_name)
    ex isa Expr || return
    # Direct bare call at statement level
    if _is_bare_lpdf_call(ex)
        fname = ex.args[1]
        @warn("[@skate $model_name] Bare call to '$fname(...)' in @logjoint — " *
              "return value is discarded. Did you mean 'target += $fname(...)'?")
        return
    end
    # Recurse into blocks (but not into += which is valid)
    if ex.head == :block
        for a in ex.args
            _warn_bare_lpdf(a, model_name)
        end
    elseif ex.head == :for && length(ex.args) >= 2
        _warn_bare_lpdf(ex.args[2], model_name)
    end
end

"""
    @skate ModelName begin
        @constants begin ... end
        @params begin ... end
        @logjoint begin ... end
    end

Define a Bayesian model. Generates:
- `ModelNameData` struct for holding constants/data
- `make(data::ModelNameData) → ModelLogDensity` to build the compiled model

**Blocks:**
- `@constants` — Declare data fields with types (e.g. `N::Int`, `X::Matrix{Float64}`)
- `@params` — Declare parameters to sample. Scalars use `name::Float64`, constrained
  params use `name = param(Float64; lower=0.0)`, vectors/matrices use
  `name = param(Vector{Float64}, K)`. Supports `simplex=true`, `ordered=true`.
- `@logjoint` — The log-joint density. Accumulate via `target += lpdf(...)`.
  Use `@for begin ... end` for zero-allocation broadcast-to-loop unrolling.
"""
macro skate(model_name::Symbol, body::Expr)
    body.head == :block || error("@skate expects begin...end block")
    data_blk = params_blk = model_blk = nothing
    found_blocks = Symbol[]
    for expr in body.args
        expr isa Expr && expr.head == :macrocall || continue
        sym = expr.args[1]
        blk = last(filter(a -> a isa Expr, expr.args))

        if sym == Symbol("@constants")
            data_blk = blk
            push!(found_blocks, Symbol("@constants"))
        elseif sym == Symbol("@data")
            # Accept @data as alias for @constants
            data_blk = blk
            push!(found_blocks, Symbol("@data"))
        elseif sym == Symbol("@params")
            params_blk = blk
            push!(found_blocks, Symbol("@params"))
        elseif sym == Symbol("@logjoint")
            model_blk = blk
            push!(found_blocks, Symbol("@logjoint"))
        elseif sym == Symbol("@model")
            # Common mistake from Stan users
            error("@skate: found '@model' — did you mean '@logjoint'?")
        end
    end

    missing = String[]
    data_blk === nothing && push!(missing, "@constants")
    params_blk === nothing && push!(missing, "@params")
    model_blk === nothing && push!(missing, "@logjoint")
    if !isempty(missing)
        found_str = isempty(found_blocks) ? "none" : join(found_blocks, ", ")
        error("@skate $model_name: missing block(s): $(join(missing, ", ")). " *
              "Found: $found_str. " *
              "A model requires @constants, @params, and @logjoint blocks.")
    end

    data_fields, data_names = _parse_constants(data_blk)
    dn = Set(data_names)

    data_struct_name = Symbol(string(model_name) * "Data")

    param_specs = _parse_params(params_blk, dn)

    # Build inline unpack + transform + jacobian statements
    unpack_stmts = Expr[]
    constrain_stmts = Expr[]
    idx = 1
    dim_expr::Union{Int,Expr} = 0

    for s in param_specs
        c = s.constraint_expr
        if s.container == :scalar
            push!(unpack_stmts, :($(s.name) = transform($c, q[$idx])))
            push!(unpack_stmts, :(log_jac += log_abs_det_jacobian($c, q[$idx])))
            push!(constrain_stmts, :($(s.name) = transform($c, q[$idx])))
            idx = _add(idx, 1)
            dim_expr = _add(dim_expr, 1)

        elseif s.container == :vector
            n = s.sizes[1]
            stop = _sub(_add(idx, n), 1)
            push!(unpack_stmts, :($(s.name) = @view q[$idx : $stop]))
            jac_loop = quote
                for _i in $idx : $stop
                    log_jac += log_abs_det_jacobian($c, q[_i])
                end
            end
            push!(unpack_stmts, jac_loop)
            push!(constrain_stmts, :($(s.name) = [transform($c, q[_i]) for _i in $idx : $stop]))
            idx = _add(idx, n)
            dim_expr = _add(dim_expr, n)

        elseif s.container == :simplex
            K = s.sizes[1]
            Km1 = _sub(K, 1)
            stop = _sub(_add(idx, Km1), 1)
            _x = gensym(:sx)
            _lj = gensym(:slj)
            push!(unpack_stmts, quote
                $_x, $_lj = simplex_transform(@view q[$idx : $stop])
                $(s.name) = $_x
                log_jac += $_lj
            end)
            push!(constrain_stmts, :($(s.name) = first(simplex_transform(@view q[$idx : $stop]))))
            idx = _add(idx, Km1)
            dim_expr = _add(dim_expr, Km1)

        elseif s.container == :ordered
            K = s.sizes[1]
            stop = _sub(_add(idx, K), 1)
            _x = gensym(:ox)
            _lj = gensym(:olj)
            push!(unpack_stmts, quote
                $_x, $_lj = ordered_transform(@view q[$idx : $stop])
                $(s.name) = $_x
                log_jac += $_lj
            end)
            push!(constrain_stmts, :($(s.name) = transform(OrderedConstraint(), @view q[$idx : $stop])))
            idx = _add(idx, K)
            dim_expr = _add(dim_expr, K)

        elseif s.container == :matrix
            K = s.sizes[1]
            D = s.sizes[2]
            od = s.ordered_dim

            _mat = gensym(:mat)
            if od == -2
                # Row-wise simplex with ordered column 1
                Dm1 = _sub(D, 1)
                total = _mul(K, Dm1)

                push!(unpack_stmts, quote
                    $_mat = Matrix{Float64}(undef, $K, $D)
                    log_jac += ordered_simplex_matrix!($_mat, @view(q[$idx : $(_sub(_add(idx, total), 1))]), $K, $D)
                    $(s.name) = $_mat
                end)

                push!(constrain_stmts, quote
                    $_mat = Matrix{Float64}(undef, $K, $D)
                    ordered_simplex_matrix!($_mat, @view(q[$idx : $(_sub(_add(idx, total), 1))]), $K, $D)
                    $(s.name) = $_mat
                end)

                idx = _add(idx, total)
                dim_expr = _add(dim_expr, total)
            elseif od == -1
                # Row-wise simplex: each row is a D-simplex with D-1 free params
                Dm1 = _sub(D, 1)
                total = _mul(K, Dm1)
                _k_var = gensym(:k)
                _cs = gensym(:cs)
                _ce = gensym(:ce)

                push!(unpack_stmts, quote
                    $_mat = Matrix{Float64}(undef, $K, $D)
                    for $_k_var in 1:$K
                        $_cs = $idx + ($_k_var - 1) * $Dm1
                        $_ce = $_cs + $Dm1 - 1
                        log_jac += simplex_transform!(@view($_mat[$_k_var, :]), @view(q[$_cs : $_ce]))
                    end
                    $(s.name) = $_mat
                end)

                push!(constrain_stmts, quote
                    $_mat = Matrix{Float64}(undef, $K, $D)
                    for $_k_var in 1:$K
                        $_cs = $idx + ($_k_var - 1) * $Dm1
                        $_ce = $_cs + $Dm1 - 1
                        simplex_transform!(@view($_mat[$_k_var, :]), @view(q[$_cs : $_ce]))
                    end
                    $(s.name) = $_mat
                end)

                idx = _add(idx, total)
                dim_expr = _add(dim_expr, total)
            elseif od > 0
                total = _mul(K, D)
                _d_var = gensym(:d)
                _cs = gensym(:cs)
                _ce = gensym(:ce)
                _ox = gensym(:ox)
                _olj = gensym(:olj)

                push!(unpack_stmts, quote
                    $_mat = Matrix{Float64}(undef, $K, $D)
                    for $_d_var in 1:$D
                        $_cs = $idx + ($_d_var - 1) * $K
                        $_ce = $_cs + $K - 1
                        if $_d_var == $od
                            $_ox, $_olj = ordered_transform(@view q[$_cs : $_ce])
                            $_mat[:, $_d_var] = $_ox
                            log_jac += $_olj
                        else
                            $_mat[:, $_d_var] .= @view q[$_cs : $_ce]
                        end
                    end
                    $(s.name) = $_mat
                end)

                push!(constrain_stmts, quote
                    $_mat = Matrix{Float64}(undef, $K, $D)
                    for $_d_var in 1:$D
                        $_cs = $idx + ($_d_var - 1) * $K
                        $_ce = $_cs + $K - 1
                        if $_d_var == $od
                            $_mat[:, $_d_var] = transform(OrderedConstraint(), @view q[$_cs : $_ce])
                        else
                            $_mat[:, $_d_var] .= @view q[$_cs : $_ce]
                        end
                    end
                    $(s.name) = $_mat
                end)

                idx = _add(idx, total)
                dim_expr = _add(dim_expr, total)
            else
                total = _mul(K, D)
                stop = _sub(_add(idx, total), 1)
                push!(unpack_stmts, :($(s.name) = reshape(@view(q[$idx : $stop]), $K, $D)))
                push!(constrain_stmts, :($(s.name) = reshape(q[$idx : $stop], $K, $D)))

                idx = _add(idx, total)
                dim_expr = _add(dim_expr, total)
            end

        elseif s.container == :chol_corr
            D = s.sizes[1]
            n_free = _div(_mul(D, _sub(D, 1)), 2)
            stop = _sub(_add(idx, n_free), 1)
            _x = gensym(:ccl)
            _lj = gensym(:cclj)
            push!(unpack_stmts, quote
                $_x, $_lj = corr_cholesky_transform(@view(q[$idx : $stop]), $D)
                $(s.name) = $_x
                log_jac += $_lj
            end)
            push!(constrain_stmts, :($(s.name) = first(corr_cholesky_transform(@view(q[$idx : $stop]), $D))))
            idx = _add(idx, n_free)
            dim_expr = _add(dim_expr, n_free)

        elseif s.container == :chol_corr_batch
            K = s.sizes[1]
            D = s.sizes[2]
            per_elem = _div(_mul(D, _sub(D, 1)), 2)
            total = _mul(K, per_elem)

            _arr = gensym(:cca)
            _k = gensym(:cck)
            _cs = gensym(:ccstart)
            _ce = gensym(:ccend)
            _lj = gensym(:cclj)

            push!(unpack_stmts, quote
                $_arr = zeros(Float64, $D, $D, $K)
                for $_k in 1:$K
                    $_cs = $idx + ($_k - 1) * $per_elem
                    $_ce = $_cs + $per_elem - 1
                    $_lj = corr_cholesky_transform!(@view($_arr[:, :, $_k]), @view(q[$_cs : $_ce]), $D)
                    log_jac += $_lj
                end
                $(s.name) = $_arr
            end)

            push!(constrain_stmts, quote
                $_arr = zeros(Float64, $D, $D, $K)
                for $_k in 1:$K
                    $_cs = $idx + ($_k - 1) * $per_elem
                    $_ce = $_cs + $per_elem - 1
                    corr_cholesky_transform!(@view($_arr[:, :, $_k]), @view(q[$_cs : $_ce]), $D)
                end
                $(s.name) = $_arr
            end)

            idx = _add(idx, total)
            dim_expr = _add(dim_expr, total)
        end
    end

    param_names = Set(s.name for s in param_specs)

    # ── @logjoint body validation ─────────────────────────────────────────────

    model_lines = _lines(model_blk)

    # Check 1: warn if @logjoint has no target += statements
    has_target_accum = any(model_lines) do s
        s isa Expr || return false
        (s.head == :(+=) && s.args[1] == :target) && return true
        # Also check inside for loops
        _contains_target_accum(s)
    end
    has_target_accum || @warn(
        "[@skate $model_name] @logjoint body has no 'target +=' statements. " *
        "The log-density will always be 0.0. Did you forget 'target +='?")

    # Check 2: warn on bare lpdf calls (result discarded)
    for s in model_lines
        _warn_bare_lpdf(s, model_name)
    end

    # Check 3: check for undefined variable references
    all_known = Set{Symbol}()
    union!(all_known, dn)
    union!(all_known, param_names)
    union!(all_known, _KNOWN_LPDF_NAMES)
    for s in [:target, :log_mix, :log, :exp, :log1p, :sqrt, :abs, :max, :min,
              :clamp, :sum, :length, :size, :view, :nothing, :pi, :Inf, :NaN, :inv]
        push!(all_known, s)
    end
    # Collect locally assigned variables in the body
    local_assigns = Set{Symbol}()
    _collect_assigns(model_lines, local_assigns)
    union!(all_known, local_assigns)

    refs = Set{Symbol}()
    for s in model_lines
        _collect_refs(s, refs)
    end
    # Filter to only symbols that look like user variables (not Julia builtins)
    undefined = setdiff(refs, all_known)
    # Remove numeric-looking or single-char iteration vars (i, j, k, etc. are fine in for loops)
    filter!(s -> length(string(s)) > 1, undefined)
    for undef_name in undefined
        suggestion = _suggest_lpdf(undef_name)
        if suggestion !== nothing
            @warn("[@skate $model_name] Unknown function '$undef_name' in @logjoint. " *
                  "Did you mean '$suggestion'?")
        end
    end

    # Check 4: check for unknown lpdf-like function calls (typo detection)
    call_names = Set{Symbol}()
    for s in model_lines
        _collect_call_names(s, call_names)
    end
    for name in call_names
        sname = string(name)
        if (endswith(sname, "_lpdf") || endswith(sname, "_lccdf") || endswith(sname, "_lcdf")) &&
           name ∉ _KNOWN_LPDF_NAMES
            suggestion = _suggest_lpdf(name)
            hint = suggestion !== nothing ? " Did you mean '$suggestion'?" : ""
            @warn("[@skate $model_name] Unknown density function '$name' in @logjoint.$hint")
        end
    end

    raw_stmts = [_rewrite_data_refs(s, dn, param_names) for s in model_lines]
    expanded_stmts = _expand_for_annotations(raw_stmts, param_specs, data_fields)
    model_stmts = [_inline_log_mix(_auto_view(s)) for s in expanded_stmts]

    for s in model_stmts
        if _has_closure(s)
            @warn "[@skate $model_name] Closure detected in @logjoint body after inlining. " *
                  "Closures that capture both data and parameters will cause " *
                  "EnzymeRuntimeActivityError. Refactor to avoid closures or " *
                  "use log_mix(weights) do j; ... end (which is auto-inlined)."
            break
        end
    end

    nt_fields = [Expr(:(=), s.name, s.name) for s in param_specs]

    out = quote
        @kwdef struct $data_struct_name
            $(data_fields...)
        end

        function make(data::$data_struct_name)
            dim = $dim_expr

            ℓ = function(q::Vector{Float64})
                log_jac = 0.0
                $(unpack_stmts...)

                target = 0.0
                $(model_stmts...)
                return target + log_jac
            end

            constrain = function(q::AbstractVector{Float64})
                $(constrain_stmts...)
                return $(Expr(:tuple, nt_fields...))
            end
            return ModelLogDensity(dim, ℓ, constrain)
        end;
    end

    return esc(out)
end
