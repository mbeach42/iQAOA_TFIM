half(x) = div(x, 2)
third(x) = div(x, 3)

function mymod1(x::Int, y::Int)
    if x == 0
        return y
    elseif x == y + 1
        return 1
    else
        return x
    end
end

function unit_tuple(D::Int, i::Int, val::Int)
    tmp = zeros(Int, D)
    tmp[i] = val
    return CartesianIndex(Tuple(tmp))
end
