@doc raw"""
    Kdiff(K_guess,n_par,m_par)

Calculate the difference between the capital stock that is assumed and the capital
stock that prevails under that guessed capital stock's implied prices when
households face idiosyncratic income risk (Aiyagari model).

Requires global functions `employment(K,A,m_par)`, `interest(K,A,N,m_par)`,
`wage(K,A,N,m_par)`, `output(K,TFP,N,m_par)`, and [`Ksupply()`](@ref).

# Arguments
- `K_guess::Float64`: capital stock guess
- `n_par::NumericalParameters`, `m_par::ModelParameters`
"""
function Kdiff(K_guess::Float64, n_par::NumericalParameters, m_par::ModelParameters,
    initial::Bool = true, Vm_guess::AbstractArray = zeros(1, 1), distr_guess::AbstractArray = zeros(1, 1, 1))
    #----------------------------------------------------------------------------
    # Calculate other prices from capital stock
    #----------------------------------------------------------------------------
    # #----------------------------------------------------------------------------
    # # Array (inc) to store incomes
    # # inc[1] = labor income , inc[2] = rental income,
    # # inc[3]= liquid assets income, inc[4] = capital liquidation income
    # #----------------------------------------------------------------------------
    Paux            = n_par.Π^1000          # Calculate ergodic ince distribution from transitions
    distr_y         = Paux[1, :]            # stationary income distribution
    N               = employment(K_guess, 1.0./(m_par.μ*m_par.μw), m_par)
    r               = interest(K_guess, 1.0./m_par.μ, N, m_par) + 1.0 
    w               = wage(K_guess, 1.0./m_par.μ, N, m_par)
    Y               = output(K_guess, 1.0, N, m_par)      
    profits         = (1.0 .- 1.0./m_par.μ) .*Y
    unionprofits    = (1.0 .- 1.0/m_par.μw) .* w .* N
    
    LC              = 1.0./m_par.μw *w.*N  
    taxrev          = ((n_par.grid_y/n_par.H).*LC)-m_par.τ_lev.*((n_par.grid_y/n_par.H).*LC).^(1.0-m_par.τ_prog)
    taxrev[end]     = n_par.grid_y[end].*profits - m_par.τ_lev.*(n_par.grid_y[end].*profits).^(1.0-m_par.τ_prog)
    incgrossaux     = ((n_par.grid_y/n_par.H).*LC)
    incgrossaux[end]= n_par.grid_y[end].*profits
    av_tax_rate     = dot(distr_y, taxrev)./(dot(distr_y,incgrossaux))
    
    incgross, inc, eff_int = 
    incomes(n_par, m_par, 1.0 ./ m_par.μw, 1.0, 
            r, m_par.τ_prog, m_par.τ_lev, n_par.H, 1.0, w,N,profits,unionprofits, av_tax_rate)
    #----------------------------------------------------------------------------
    # Initialize policy function (guess/stored values)
    #----------------------------------------------------------------------------

    # initial guess consumption and marginal values (if not set)
    if initial
        c_guess     = inc[1] .+  inc[3].*(n_par.mesh_m.>0)
        if any(any(c_guess .< 0.0))
            @warn "negative consumption guess"
        end
        Vm          = eff_int .* mutil(c_guess)
        distr       = n_par.dist_guess
    else
        Vm          = Vm_guess
        distr       = distr_guess
    end
    #----------------------------------------------------------------------------
    # Calculate supply of funds for given prices
    #----------------------------------------------------------------------------
    KS              = Ksupply(r, n_par, m_par, Vm, inc, eff_int)
    K               = KS[1].*(1.0 .- m_par.ψ)                                 # capital
    Vm              = KS[end-1]                                                 # marginal value of liquid assets
    distr           = KS[end]                                                   # stationary distribution  
    diff            = K - K_guess                                               # excess supply of funds
    return diff, Vm, distr
end
