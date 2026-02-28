
import Base.@kwdef

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
    # Don't rewrite the LHS of assignments
    if ex.head == :(=)
        return Expr(:(=), ex.args[1],
                    _rewrite_data_refs(ex.args[2], data_names, param_names))
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

function _parse_constants(block::Expr)
    fields = Expr[]
    names = Symbol[]
    for line in _lines(block)
        line isa Expr && line.head == :(::) || continue
        var = line.args[1]
        typespec = line.args[2]
        var isa Symbol || @warn "Expected a Symbol, got $(typeof(var)), ignoring." continue
        push!(fields, :($var::$(_dsl_to_julia_type(typespec))))
        push!(names, var)
    end
    fields,names
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
    ordered_dim::Int        # matrix only: 0 = none, N = column N gets ordered constraint
end

function _parse_params(block::Expr, data_names::Set{Symbol})
    specs = _ParamSpec[]
    for line in _lines(block)
        line isa Expr || continue

        if line.head == :(::)
            var = line.args[1]
            T = line.args[2]
            var isa Symbol || error(
                "@params: '$var::$T' — bare annotations only support Float64. " *
                "For vectors and matrices use param(Vector{Float64}, n, ...)")
            push!(specs, _ParamSpec(var, :(IdentityConstraint()), :scalar, [], 0))

        elseif line.head == :(=)
            var = line.args[1]
            rhs = line.args[2]
            rhs isa Expr && rhs.head == :call && rhs.args[1] == :param || error(
                "@params: '$var = $rhs' — expected a call to param(...)")
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

@enum _ShapeKind _shape_scalar _shape_vector _shape_matrix

struct _ShapeInfo
    kind::_ShapeKind
    len::Any        # vector length expr, or nothing
    ncols::Any      # matrix col count, or nothing
end

_scalar_shape() = _ShapeInfo(_shape_scalar, nothing, nothing)
_vector_shape(len) = _ShapeInfo(_shape_vector, len, nothing)
_matrix_shape(nrows, ncols) = _ShapeInfo(_shape_matrix, nrows, ncols)

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

"""Recursive shape inference for expressions."""
function _infer_shape(ex, env::Dict{Any, _ShapeInfo})
    ex isa Number && return _scalar_shape()

    if ex isa Symbol
        haskey(env, ex) && return env[ex]
        return _scalar_shape()
    end

    ex isa Expr || return _scalar_shape()

    if _is_data_ref(ex)
        key = _data_key(ex)
        haskey(env, key) && return env[key]
        return _scalar_shape()
    end

    # Dot-call: f.(args...)
    if ex.head == :. && length(ex.args) == 2 &&
       ex.args[2] isa Expr && ex.args[2].head == :tuple
        for a in ex.args[2].args
            s = _infer_shape(a, env)
            s.kind == _shape_vector && return s
        end
        return _scalar_shape()
    end

    if ex.head == :call
        op = ex.args[1]
        operands = ex.args[2:end]

        if _is_dot_op(op)
            for a in operands
                s = _infer_shape(a, env)
                s.kind == _shape_vector && return s
            end
            return _scalar_shape()
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
            return _scalar_shape()
        end

        op == :sum && return _scalar_shape()

        for a in operands
            s = _infer_shape(a, env)
            s.kind == _shape_vector && return s
        end
        return _scalar_shape()
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
            return _scalar_shape()
        end

        if base_shape.kind == _shape_matrix && length(indices) == 2
            if _is_slice_index(indices[1])
                return _matrix_shape(base_shape.len, nothing)
            end
        end
    end

    return _scalar_shape()
end

"""Core: scalarize a broadcast expression for loop index `idx`."""
function _scalarize(ex, idx::Symbol, env::Dict{Any, _ShapeInfo}, preamble::Vector{Expr})
    shape = _infer_shape(ex, env)

    if shape.kind == _shape_scalar
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

"""Expand a single @for assignment: `@for y = broadcast_expr`."""
function _expand_for_assign(stmt, env)
    lhs = stmt.args[1]
    rhs = stmt.args[2]
    shape = _infer_shape(rhs, env)
    len_expr = shape.len

    idx = gensym(:i)
    preamble = Expr[]
    body = _scalarize(rhs, idx, env, preamble)

    env[lhs] = _vector_shape(len_expr)

    return quote
        $lhs = Vector{Float64}(undef, $len_expr)
        for $idx in 1:$len_expr
            $(preamble...)
            $lhs[$idx] = $body
        end
    end
end

"""Expand a fused @for block: multiple assignments in one loop."""
function _expand_for_block(block, env)
    stmts = _lines(block)
    isempty(stmts) && return block

    len_expr = nothing
    for s in stmts
        s isa Expr && s.head == :(=) || continue
        shape = _infer_shape(s.args[2], env)
        if shape.kind == _shape_vector && shape.len !== nothing
            len_expr = shape.len
            break
        end
    end
    len_expr === nothing && error("@for block: could not infer loop dimension")

    idx = gensym(:i)
    allocs = Expr[]
    loop_body = Expr[]

    for s in stmts
        s isa Expr && s.head == :(=) || continue
        lhs = s.args[1]
        rhs = s.args[2]

        shape = _infer_shape(rhs, env)
        if shape.kind == _shape_vector
            push!(allocs, :($lhs = Vector{Float64}(undef, $len_expr)))
            preamble = Expr[]
            body = _scalarize(rhs, idx, env, preamble)
            append!(loop_body, preamble)
            push!(loop_body, :($lhs[$idx] = $body))
            env[lhs] = _vector_shape(len_expr)
        else
            push!(loop_body, s)
        end
    end

    return quote
        $(allocs...)
        for $idx in 1:$len_expr
            $(loop_body...)
        end
    end
end

"""Expand `@for target += sum(broadcast_expr)`."""
function _expand_for_sum(stmt, env)
    rhs = stmt.args[2]
    inner = rhs.args[2]

    shape = _infer_shape(inner, env)
    len_expr = shape.len

    idx = gensym(:i)
    preamble = Expr[]
    body = _scalarize(inner, idx, env, preamble)

    return quote
        for $idx in 1:$len_expr
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

"""Walk statement list, expand @for annotations, maintain shape env."""
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
        else
            push!(output, s)
            if s.head == :(=) && s.args[1] isa Symbol
                env[s.args[1]] = _infer_shape(s.args[2], env)
            end
        end
    end
    output
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

macro skate(model_name::Symbol, body::Expr)
    body.head == :block || error("@skate expects begin...end block")
    data_blk = params_blk = model_blk = nothing
    for expr in body.args
        expr isa Expr && expr.head == :macrocall || continue
        sym = expr.args[1]
        blk = last(filter(a -> a isa Expr, expr.args))

        sym == Symbol("@constants")     && (data_blk = blk)
        sym == Symbol("@params")   && (params_blk = blk)
        sym == Symbol("@logjoint") && (model_blk = blk)
    end

    data_blk !== nothing || error("Missing @constants block")
    params_blk !== nothing || error("Missing @params block")
    model_blk !== nothing || error("Missing @logjoint block")

    data_fields, data_names = _parse_constants(data_blk)
    dn = Set(data_names)

    data_struct_name = Symbol(string(model_name) * "_DataSet")

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
            total = _mul(K, D)

            _mat = gensym(:mat)
            if od > 0
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
            else
                stop = _sub(_add(idx, total), 1)
                push!(unpack_stmts, :($(s.name) = reshape(@view(q[$idx : $stop]), $K, $D)))
                push!(constrain_stmts, :($(s.name) = reshape(q[$idx : $stop], $K, $D)))
            end

            idx = _add(idx, total)
            dim_expr = _add(dim_expr, total)

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

    make_model_name = Symbol("make_" * lowercase(string(model_name)))
    param_names = Set(s.name for s in param_specs)
    raw_stmts = [_rewrite_data_refs(s, dn, param_names) for s in _lines(model_blk)]
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

        function $make_model_name(data::$data_struct_name)
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
