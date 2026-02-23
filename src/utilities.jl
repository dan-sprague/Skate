
"""
    log_sum_exp(x)

The 'Bare Metal' stability trick for Logit/Softmax math.
"""
function log_sum_exp(x)
    max_x = maximum(x)
    return max_x + log(sum(exp.(x .- max_x)))
end