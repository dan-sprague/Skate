ex = :(1 + 2)

typeof(ex)

dump(ex)

eval(ex)

macro sayhello(name)
    return :(println("Hello ", $name))
end

@sayhello("Skater")

dump(@macroexpand @sayhello("Skater"))

ex = :(begin x = 1; y = 2 end)
ex.args
filter(x -> !(x isa LineNumberNode), ex.args)   # just the "real" lines


macro mystruct(name, body)
    
    body.head == :block || error("Expected a block of expressions")
    filter!(x -> !(x isa LineNumberNode), body.args)

    fields = body.args
    esc(quote
        struct $name
            $(fields...)
        end
    end)
end

