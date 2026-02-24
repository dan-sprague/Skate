
include("bijections.jl")
import Base.@kwdef

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

function _parse_data(block::Expr)
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


macro spec(model_name::Symbol, body::Expr)
    body.head == :block || error("@specsheet expects begin...end block")
    data_blk = params_blk = model_blk = nothing
    for expr in body.args
        expr isa Expr && expr.head == :macrocall || continue
        sym = expr.args[1]
        blk = last(filter(a -> a isa Expr, expr.args))

        sym == Symbol("@data")     && (data_blk = blk)
        sym == Symbol("@params")   && (params_blk = blk)
        sym == Symbol("@logjoint") && (model_blk = blk)
    end

    data_blk !== nothing || error("Missing @data block")
    params_blk !== nothing || error("Missing @params block")
    model_blk !== nothing || error("Missing @logjoint block")

    data_fields, data_names = _parse_data(data_blk)
    dn = Set(data_names)

    data_struct_name = Symbol(string(model_name) * "_DataSet")

    param_specs = _parse_params(params_blk, dn)

    # Build inline unpack + transform + jacobian statements
    # No process_params, no transforms tuple — everything emitted directly
    unpack_stmts = Expr[]
    prealloc_stmts = Expr[]   # allocated once outside the closure
    idx = 1                   # tracks position in flat q vector (Int or Expr)
    dim_expr::Union{Int,Expr} = 0

    _add(a::Int, b::Int) = a + b
    _add(a, b) = :($a + $b)
    _sub(a::Int, b::Int) = a - b
    _sub(a, b) = :($a - $b)

    for s in param_specs
        c = s.constraint_expr
        if s.container == :scalar
            push!(unpack_stmts, :($(s.name) = transform($c, q[$idx])))
            push!(unpack_stmts, :(log_jac += log_abs_det_jacobian($c, q[$idx])))
            idx = _add(idx, 1)
            dim_expr = _add(dim_expr, 1)

        elseif s.container == :vector
            n = s.sizes[1]  # Int literal or :(data.N)
            stop = _sub(_add(idx, n), 1)
            push!(unpack_stmts, :($(s.name) = @view q[$idx : $stop]))
            jac_loop = quote
                for _i in $idx : $stop
                    log_jac += log_abs_det_jacobian($c, q[_i])
                end
            end
            push!(unpack_stmts, jac_loop)
            idx = _add(idx, n)
            dim_expr = _add(dim_expr, n)

        elseif s.container == :simplex
            K = s.sizes[1]  # simplex dimension (K)
            Km1 = _sub(K, 1)  # unconstrained dimension (K-1)
            stop = _sub(_add(idx, Km1), 1)
            buf_name = Symbol("_simplex_buf_", s.name)
            push!(prealloc_stmts, :($buf_name = Vector{Float64}(undef, $K)))
            push!(unpack_stmts, quote
                log_jac += simplex_transform!($buf_name, @view q[$idx : $stop])
                $(s.name) = $buf_name
            end)
            idx = _add(idx, Km1)
            dim_expr = _add(dim_expr, Km1)
        end
    end

    make_model_name = Symbol("make_" * lowercase(string(model_name)))
    param_names = Set(s.name for s in param_specs)
    model_stmts = [_rewrite_data_refs(s, dn, param_names) for s in _lines(model_blk)]

    out = quote
        @kwdef struct $data_struct_name
            $(data_fields...)
        end

        function $make_model_name(data::$data_struct_name)
            dim = $dim_expr
            $(prealloc_stmts...)

            ℓ = function(q::Vector{Float64})
                log_jac = 0.0
                $(unpack_stmts...)

                target = 0.0
                $(model_stmts...)
                return target + log_jac
            end

            return ModelLogDensity(dim, ℓ)
        end
    end

    return esc(out)
end

ex = quote
    @data begin
        x::Int
        y::Float64
    end
    @params begin
        alpha = param(Vector{Float64}, p)
        gamma = param(Vector{Float64}, p, lower = 0.0, upper = 1.0)

        beta::Float64
    end
end

dump(_lines(ex))



"""Map (lo, hi) keyword values to the appropriate Constraint expression."""
function _make_constraint_expr(lo, hi)
    isnothing(lo) && isnothing(hi)  && return :(IdentityConstraint())
    !isnothing(lo) && isnothing(hi) && return :(LowerBounded($lo))
    isnothing(lo) && !isnothing(hi) && return :(UpperBounded($hi))
    return :(Bounded($lo, $hi))
end

struct _ParamSpec
    name::Symbol
    constraint_expr::Expr   # e.g. :(LowerBounded(0.0))
    container::Symbol       # :scalar | :vector | :simplex | :matrix
    sizes::Vector{Any}
end


function _parse_params(block::Expr,data_names::Set{Symbol})
    specs = _ParamSpec[]
    for line in _lines(block)
        line isa Expr || continue

        if line.head == :(::)
            var = line.args[1]
            T = line.args[2]
            var isa Symbol || error(
                "@params: '$var::$T' — bare annotations only support Float64. " *
                "For vectors and matrices use param(Vector{Float64}, n, ...)")
            push!(specs, _ParamSpec(var, :(IdentityConstraint()), :scalar, []))

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

    # Julia's parser puts kwargs in a :parameters node before positional args
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

    # Extract bounds and flags from keywords
    lo = hi = nothing
    is_simplex = false
    for a in kw_args
        a.args[1] == :lower   && (lo = a.args[2])
        a.args[1] == :upper   && (hi = a.args[2])
        a.args[1] == :simplex && (is_simplex = a.args[2])
    end

    constraint = _make_constraint_expr(lo, hi)

    # Scalar
    if T == :Float64
        is_simplex && error("@params: simplex not supported for scalars")
        isempty(sz_args) || error("@params: param(Float64) takes no positional size arguments")
        return _ParamSpec(name, constraint, :scalar, [])
    end

    # Vector or Matrix
    if T isa Expr && T.head == :curly
        base = T.args[1]
        elem = T.args[2]
        if base == :Vector && elem == :Float64
            length(sz_args) == 1 || error("@params: param(Vector{Float64}, n) takes one size argument")
            n = _resolve_size(sz_args[1], dn)
            if is_simplex
                (lo !== nothing || hi !== nothing) &&
                    error("@params: simplex params cannot have bounds")
                return _ParamSpec(name, :(SimplexConstraint()), :simplex, [n])
            end
            return _ParamSpec(name, constraint, :vector, [n])
        elseif base == :Matrix && elem == :Float64
            length(sz_args) == 2 || error("@params: param(Matrix{Float64}, n, m) takes two size arguments")
            constraint != :(IdentityConstraint()) && error("@params: bounded matrices not supported")
            sz1 = _resolve_size(sz_args[1], dn)
            sz2 = _resolve_size(sz_args[2], dn)
            return _ParamSpec(name, constraint, :matrix, [sz1, sz2])
        end
    end

    error("@params: unsupported type in param(): $T")
end


@macroexpand @spec Model begin
    @data begin
        N::Int
        x::Vector{Float64}
        y::Vector{Float64}
    end

    @params begin
        mu::Float64
        sigma = param(Float64; lower = 0.0)
        alpha = param(Vector{Float64}, 10; lower = 0.0)
    end

    @logjoint begin
        for i in 1:data.N
            target += normal_lpdf(x[i], mu, sigma)
            target += normal_lpdf(y[i], transform(LowerBounded(0.0), alpha[i]), sigma)
        end
    end
end