using Enzyme

f(x::Float64) = exp(x)

function (@main)(args::Vector{String})::Cint
    if isempty(args)
        Core.println("Usage: testprogram <x>")
        return Cint(1)
    end

    x = parse(Float64, args[1])

    # Use invokelatest to prevent type inference from triggering Enzyme codegen at compile time
    grad = Base.invokelatest(Enzyme.autodiff, Forward, f, Duplicated(x, 1.0))[1]

    Core.println("f(x)  = exp($(x)) = $(f(x))")
    Core.println("f'(x) = $(grad)")

    return Cint(0)
end
