# XLA/Reactant-compatible code generation for @xlaspec macro.

"""Replace `var` with `replacement` throughout an expression."""
function _xla_subst_var(ex, var::Symbol, replacement)
    ex isa Symbol && ex == var && return replacement
    ex isa Expr || return ex
    return Expr(ex.head, [_xla_subst_var(a, var, replacement) for a in ex.args]...)
end

"""
Extract (weights, params, body) from a `log_mix` call, or `nothing` if not recognized.
Supports two forms:
  - `log_mix(weights) do j; body; end`   → Expr(:do, ...)
  - `log_mix(weights, j -> body)`        → Expr(:call, ..., Expr(:->,...))
"""
function _xla_extract_log_mix(ex)
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

"""
Inline `log_mix` calls into closure-free branchless log-sum-exp loops.
XLA version: no control flow (no if/else), uses max + log(exp+exp).
"""
function _xla_inline_log_mix(ex)
    ex isa Expr || return ex
    ex = Expr(ex.head, [_xla_inline_log_mix(a) for a in ex.args]...)

    weights, params, body = _xla_extract_log_mix(ex)
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
    body_1  = _xla_subst_var(body, j, 1)
    body_jj = _xla_subst_var(body, j, jj)

    _m = gensym(:lse_m)
    return quote
        $acc = log($weights[1]) + $body_1
        for $jj in 2:length($weights)
            $lp = log($weights[$jj]) + $body_jj
            $_m = max($acc, $lp)
            $acc = $_m + log(exp($acc - $_m) + exp($lp - $_m))
        end
        $acc
    end
end

"""Check if an expression contains any closure (`->` or `do`) nodes."""
function _xla_has_closure(ex)
    ex isa Expr || return false
    (ex.head == :-> || ex.head == :do) && return true
    return any(_xla_has_closure, ex.args)
end

"""Check if an index expression represents a slice (`:` or a range like `1:n`)."""
_xla_is_slice_index(ex) = ex === :(:) ||
    (ex isa Expr && ex.head == :call && !isempty(ex.args) && ex.args[1] == :(:))

"""
Automatically wrap matrix/array slices (e.g. `x[i, :]`) with `view(...)`.
Existing explicit `@view(...)` calls are preserved as-is.
"""
function _xla_auto_view(ex)
    ex isa Expr || return ex
    if ex.head == :macrocall && !isempty(ex.args) && ex.args[1] == Symbol("@view")
        return ex
    end
    ex = Expr(ex.head, [_xla_auto_view(a) for a in ex.args]...)
    if ex.head == :ref && length(ex.args) >= 2 && any(_xla_is_slice_index, ex.args[2:end])
        return Expr(:call, :view, ex.args...)
    end
    return ex
end

"""Rewrite bare data-name symbols in an expression to `data.name`."""
function _xla_rewrite_data_refs(ex, data_names::Set{Symbol}, param_names::Set{Symbol})
    ex isa Symbol && ex ∈ data_names && ex ∉ param_names && return :(data.$ex)
    ex isa Expr || return ex
    if ex.head == :(=)
        return Expr(:(=), ex.args[1],
                    _xla_rewrite_data_refs(ex.args[2], data_names, param_names))
    end
    return Expr(ex.head, [_xla_rewrite_data_refs(a, data_names, param_names) for a in ex.args]...)
end

function _xla_resolve_size(arg, dn::Set{Symbol})
    arg isa Integer && return arg
    arg isa Symbol  || error("@params: size argument must be an integer literal or a @data name, got: $arg")
    arg ∈ dn        || error("@params: size argument ':$arg' is not declared in @data")
    return :(data.$arg)
end
_xla_lines(b::Expr) = filter(x -> !(x isa LineNumberNode), b.args)

# ═══════════════════════════════════════════════════════════════════════════════
# XLA type parametrization helpers
# ═══════════════════════════════════════════════════════════════════════════════

"""Check if a Julia type expression is an array type (Vector, Matrix, Array)."""
function _xla_is_array_type(T)
    T isa Expr && T.head == :curly || return false
    return T.args[1] in (:Vector, :Matrix, :Array)
end

"""Convert concrete array type to abstract bound: Vector{Float64} → AbstractVector.

Element type is dropped so that Reactant TracedRArrays (whose eltype is
TracedRNumber{Float64}, not Float64) satisfy the bound."""
function _xla_abstract_bound(T)
    base = T.args[1]
    if base == :Vector
        return :AbstractVector
    elseif base == :Matrix
        return :AbstractMatrix
    elseif base == :Array
        ndim = T.args[end]
        return Expr(:curly, :AbstractArray, :(<:Any), ndim)
    end
    return T
end

"""Parse @constants block. Returns (fields, names, type_params, concrete_fields).

`fields` — struct field exprs (parametric for arrays).
`concrete_fields` — always concrete types, used for shape inference in _build_shape_env.
`type_params` — type parameter exprs for parametric struct definition.
"""
function _xla_parse_constants(block::Expr)
    fields = Expr[]
    concrete_fields = Expr[]
    names = Symbol[]
    type_params = Expr[]
    param_idx = 0
    for line in _xla_lines(block)
        line isa Expr && line.head == :(::) || continue
        var = line.args[1]
        typespec = line.args[2]
        var isa Symbol || @warn "Expected a Symbol, got $(typeof(var)), ignoring." continue
        julia_type = _xla_dsl_to_julia_type(typespec)
        # Always store concrete version for shape inference
        push!(concrete_fields, :($var::$julia_type))
        if _xla_is_array_type(julia_type)
            param_idx += 1
            T_sym = Symbol("_T", param_idx)
            push!(type_params, :($T_sym <: $(_xla_abstract_bound(julia_type))))
            push!(fields, :($var::$T_sym))
        else
            push!(fields, :($var::$julia_type))
        end
        push!(names, var)
    end
    fields, names, type_params, concrete_fields
end

function _xla_dsl_to_julia_type(spec)
    spec isa Symbol && return spec
    spec isa Expr && spec.head == :curly && return spec
    spec isa Expr && spec.head == :call && return spec.args[1]
    return spec
end

# ═══════════════════════════════════════════════════════════════════════════════

macro xlaspec(model_name::Symbol, body::Expr)
    body.head == :block || error("@xlaspec expects begin...end block")
    data_blk = params_blk = model_blk = nothing
    for expr in body.args
        expr isa Expr && expr.head == :macrocall || continue
        sym = expr.args[1]
        blk = last(filter(a -> a isa Expr, expr.args))

        sym == Symbol("@constants")      && (data_blk = blk)
        sym == Symbol("@params")         && (params_blk = blk)
        sym == Symbol("@xlalogjoint")    && (model_blk = blk)
    end

    data_blk !== nothing || error("Missing @constants block")
    params_blk !== nothing || error("Missing @params block")
    model_blk !== nothing || error("Missing @xlalogjoint block")

    data_fields, data_names, type_params, _ = _xla_parse_constants(data_blk)
    dn = Set(data_names)

    data_struct_name = Symbol(string(model_name) * "_DataSet")

    param_specs = _xla_parse_params(params_blk, dn)

    # Build inline unpack + transform + jacobian statements
    unpack_stmts = Expr[]
    constrain_stmts = Expr[]
    idx = 1
    dim_expr::Union{Int,Expr} = 0

    _add(a::Int, b::Int) = a + b
    _add(a, b) = :($a + $b)
    _sub(a::Int, b::Int) = a - b
    _sub(a, b) = :($a - $b)
    _mul(a::Int, b::Int) = a * b
    _mul(a, b) = :($a * $b)
    _div(a::Int, b::Int) = div(a, b)
    _div(a, b) = :(div($a, $b))

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
                    $_mat = similar(q, $K, $D)
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
                    $_mat = similar(q, $K, $D)
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
                $_arr = zeros(eltype(q), $D, $D, $K)
                for $_k in 1:$K
                    $_cs = $idx + ($_k - 1) * $per_elem
                    $_ce = $_cs + $per_elem - 1
                    $_lj = corr_cholesky_transform!(@view($_arr[:, :, $_k]), @view(q[$_cs : $_ce]), $D)
                    log_jac += $_lj
                end
                $(s.name) = $_arr
            end)

            push!(constrain_stmts, quote
                $_arr = zeros(eltype(q), $D, $D, $K)
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
    raw_stmts = [_xla_rewrite_data_refs(s, dn, param_names) for s in _xla_lines(model_blk)]
    model_stmts = [_xla_inline_log_mix(_xla_auto_view(s)) for s in raw_stmts]

    # Warn about closures
    for s in model_stmts
        if _xla_has_closure(s)
            @warn "[@xlaspec $model_name] Closure detected in @xlalogjoint body after inlining."
            break
        end
    end

    nt_fields = [Expr(:(=), s.name, s.name) for s in param_specs]

    # Build struct definition — parametric if array fields exist
    struct_def = if isempty(type_params)
        :(@kwdef struct $data_struct_name
            $(data_fields...)
        end)
    else
        :(@kwdef struct $data_struct_name{$(type_params...)}
            $(data_fields...)
        end)
    end

    out = quote
        $struct_def

        function $make_model_name(data::$data_struct_name)
            dim = $dim_expr

            ℓ = function(q, data)
                log_jac = zero(eltype(q))
                $(unpack_stmts...)

                target = zero(eltype(q))
                $(model_stmts...)
                return target + log_jac
            end

            constrain = function(q, data)
                $(constrain_stmts...)
                return $(Expr(:tuple, nt_fields...))
            end
            return ModelLogDensity(dim, ℓ, constrain, data)
        end;
    end

    return esc(out)
end

"""Map (lo, hi) keyword values to the appropriate Constraint expression."""
function _xla_make_constraint_expr(lo, hi)
    isnothing(lo) && isnothing(hi)  && return :(IdentityConstraint())
    !isnothing(lo) && isnothing(hi) && return :(LowerBounded($lo))
    isnothing(lo) && !isnothing(hi) && return :(UpperBounded($hi))
    return :(Bounded($lo, $hi))
end

struct _XlaParamSpec
    name::Symbol
    constraint_expr::Expr
    container::Symbol
    sizes::Vector{Any}
    ordered_dim::Int
end

function _xla_parse_params(block::Expr, data_names::Set{Symbol})
    specs = _XlaParamSpec[]
    for line in _xla_lines(block)
        line isa Expr || continue

        if line.head == :(::)
            var = line.args[1]
            T = line.args[2]
            var isa Symbol || error(
                "@params: '$var::$T' — bare annotations only support Float64. " *
                "For vectors and matrices use param(Vector{Float64}, n, ...)")
            push!(specs, _XlaParamSpec(var, :(IdentityConstraint()), :scalar, [], 0))

        elseif line.head == :(=)
            var = line.args[1]
            rhs = line.args[2]
            rhs isa Expr && rhs.head == :call && rhs.args[1] == :param || error(
                "@params: '$var = $rhs' — expected a call to param(...)")
            push!(specs, _xla_param_to_spec(var, rhs.args[2:end], data_names))
        end
    end
    specs
end

"""Convert `param(T, sizes...; lower=…, upper=…)` to _XlaParamSpec."""
function _xla_param_to_spec(name::Symbol, args, dn::Set{Symbol})
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

    constraint = _xla_make_constraint_expr(lo, hi)

    if T == :Float64
        is_simplex && error("@params: simplex not supported for scalars")
        isempty(sz_args) || error("@params: param(Float64) takes no positional size arguments")
        return _XlaParamSpec(name, constraint, :scalar, [], 0)
    end

    if T isa Expr && T.head == :curly
        base = T.args[1]
        elem = T.args[2]
        if base == :Vector && elem == :Float64
            length(sz_args) == 1 || error("@params: param(Vector{Float64}, n) takes one size argument")
            n = _xla_resolve_size(sz_args[1], dn)
            if is_simplex
                (lo !== nothing || hi !== nothing) &&
                    error("@params: simplex params cannot have bounds")
                return _XlaParamSpec(name, :(SimplexConstraint()), :simplex, [n], 0)
            end
            if is_ordered
                (lo !== nothing || hi !== nothing) &&
                    error("@params: ordered params cannot have bounds")
                return _XlaParamSpec(name, :(OrderedConstraint()), :ordered, [n], 0)
            end
            return _XlaParamSpec(name, constraint, :vector, [n], 0)
        elseif base == :Matrix && elem == :Float64
            length(sz_args) == 2 || error("@params: param(Matrix{Float64}, n, m) takes two size arguments")
            sz1 = _xla_resolve_size(sz_args[1], dn)
            sz2 = _xla_resolve_size(sz_args[2], dn)
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
            return _XlaParamSpec(name, constraint, :matrix, [sz1, sz2], ordered_col)
        end
    end

    if T == :CholCorr
        (lo !== nothing || hi !== nothing) &&
            error("@params: CholCorr params cannot have bounds")
        is_simplex && error("@params: simplex not supported for CholCorr")
        is_ordered !== false && error("@params: ordered not supported for CholCorr")

        if length(sz_args) == 1
            D = _xla_resolve_size(sz_args[1], dn)
            return _XlaParamSpec(name, :(IdentityConstraint()), :chol_corr, [D], 0)
        elseif length(sz_args) == 2
            K = _xla_resolve_size(sz_args[1], dn)
            D = _xla_resolve_size(sz_args[2], dn)
            return _XlaParamSpec(name, :(IdentityConstraint()), :chol_corr_batch, [K, D], 0)
        else
            error("@params: param(CholCorr, ...) takes 1 (D) or 2 (K, D) size arguments")
        end
    end

    error("@params: unsupported type in param(): $T")
end
