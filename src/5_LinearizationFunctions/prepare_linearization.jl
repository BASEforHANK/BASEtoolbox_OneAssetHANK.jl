
@doc raw"""
    prepare_linearization(KSS, VmSS, VkSS, distrSS, n_par, m_par)

Compute a number of equilibrium objects needed for linearization.

# Arguments
- `KSS`: steady-state capital stock
- `VmSS`, `VkSS`: marginal value functions
- `distrSS::Array{Float64,3}`: steady-state distribution of idiosyncratic states, computed by [`Ksupply()`](@ref)
- `n_par::NumericalParameters`,`m_par::ModelParameters`

# Returns
- `XSS::Array{Float64,1}`, `XSSaggr::Array{Float64,1}`: steady state vectors produced by [`@writeXSS()`](@ref)
- `indexes`, `indexes_aggr`: `struct`s for accessing `XSS`,`XSSaggr` by variable names, produced by [`@make_fn()`](@ref),
        [`@make_fnaggr()`](@ref)
- `compressionIndexes::Array{Array{Int,1},1}`: indexes for compressed marginal value functions (``V_m`` and ``V_k``)
- `Copula(x,y,z)`: function that maps marginals `x`,`y`,`z` to approximated joint distribution, produced by
        [`BASEforHANK.myinterpolate3()`](@ref)
- `n_par::NumericalParameters`,`m_par::ModelParameters`
- `CDF_SS`, `CDF_m`, `CDF_k`, `CDF_y`: cumulative distribution functions (joint and marginals)
- `distrSS::Array{Float64,3}`: steady state distribution of idiosyncratic states, computed by [`Ksupply()`](@ref)
"""
function prepare_linearization(KSS, VmSS, distrSS, n_par, m_par)
    if n_par.verbose
        println("Running reduction step on steady-state value functions to prepare linearization")
    end
    # ------------------------------------------------------------------------------
    # STEP 1: Evaluate StE to calculate steady state variable values
    # ------------------------------------------------------------------------------
    # Calculate other equilibrium quantities
    Paux      = n_par.Π^1000                                              # Calculate ergodic ince distribution from transitions
    distr_y   = Paux[1, :] 
    NSS       = employment(KSS, 1.0./(m_par.μ*m_par.μw), m_par)
    rSS       = interest(KSS, 1.0./m_par.μ, NSS, m_par) + 1.0 
    wSS       = wage(KSS, 1.0./m_par.μ, NSS, m_par)
    YSS       = output(KSS, 1.0, NSS, m_par)                              # stationary income distribution
    
    profitsSS       = (1.0 .- 1.0./m_par.μ) .*YSS
    unionprofitsSS  = (1.0 .- 1.0/m_par.μw) .* wSS .* NSS
    LC              = 1.0./m_par.μw *wSS.*NSS  
    taxrev          = ((n_par.grid_y/n_par.H).*LC)-m_par.τ_lev.*((n_par.grid_y/n_par.H).*LC).^(1.0-m_par.τ_prog)
    taxrev[end]     =  n_par.grid_y[end].*profitsSS - m_par.τ_lev.*( n_par.grid_y[end].*profitsSS).^(1.0-m_par.τ_prog)
    incgrossaux     = ((n_par.grid_y/n_par.H).*LC)
    incgrossaux[end]=  n_par.grid_y[end].*profitsSS
    av_tax_rateSS   = dot(distr_y, taxrev)./(dot(distr_y,incgrossaux))
    incgross, incnet, eff_int = 
            incomes(n_par, m_par, 1.0 ./ m_par.μw, 1.0, 
                    rSS, m_par.τ_prog, m_par.τ_lev, n_par.H, 1.0, wSS, NSS, profitsSS, unionprofitsSS, av_tax_rateSS)
    # obtain other steady state variables
    TotAssetSS, TransitionMatSS, c_n_starSS, m_n_starSS, VmSS, distrSS =Ksupply(rSS, n_par, m_par, VmSS, incnet, eff_int)
    
    VmSS        = log.(invmutil.(VmSS))
    
    # Produce distributional summary statistics
    distr_m_SS, distr_y_SS, TOP10WshareSS, TOP10IshareSS,TOP10InetshareSS, GiniWSS, GiniXSS, GiniCSS, GiniISS, sdlogySS = 
        distrSummaries(distrSS, c_n_starSS, n_par, incnet,incgross, m_par)
    
    # ------------------------------------------------------------------------------
    # STEP 2: Dimensionality reduction
    # ------------------------------------------------------------------------------
    # 2 a.) Discrete cosine transformation of marginal value functions
    # ------------------------------------------------------------------------------
    compressionIndexesVm  = select_comp_ind(VmSS,n_par.reduc_value) # store indexes of retained coefficients for Vm
        # ------------------------------------------------------------------------------
    # 2b.) Select polynomials for copula perturbation
    # ------------------------------------------------------------------------------
    SELECT                = [ ((i+j)<=n_par.reduc_copula) & 
                             (!(i==1) & !(j ==1)) 
                             for i = 1:n_par.nm_copula, j = 1:n_par.ny_copula]
    
    compressionIndexesCOP = findall(SELECT[:])     # store indices of selected coeffs 

    # ------------------------------------------------------------------------------
    # 2c.) Store Compression Indexes
    # ------------------------------------------------------------------------------
    compressionIndexes    = Array{Array{Int,1},1}(undef ,2)       # Container to store all retained coefficients in one array
    compressionIndexes[1] = compressionIndexesVm
    compressionIndexes[2] = compressionIndexesCOP
    
    
    # ------------------------------------------------------------------------------
    # 2d.) Produce marginals
    # ------------------------------------------------------------------------------
    CDF_SS               = cumsum(cumsum(distrSS,dims=1),dims=2) # Calculate CDF from PDF
    distr_m_SS           = sum(distrSS,dims=2)[:]            # Marginal distribution (pdf) of liquid assets
    distr_y_SS           = sum(distrSS,dims=1)[:]            # Marginal distribution (pdf) of income
    CDF_m                = cumsum(distr_m_SS[:])          # Marginal distribution (cdf) of liquid assets
    CDF_y                = cumsum(distr_y_SS[:])          # Marginal distribution (cdf) of income
  
    # Calculate interpolation nodes for the copula as those elements of the marginal distribution 
    # that yield close to equal aggregate shares in liquid wealth, illiquid wealth and income.
    # Entrepreneur state treated separately. 
    @set! n_par.copula_marginal_m = copula_marg_equi(distr_m_SS, n_par.grid_m, n_par.nm_copula)
    @set! n_par.copula_marginal_y = copula_marg_equi_y(distr_y_SS, n_par.grid_y, n_par.ny_copula)

    # ------------------------------------------------------------------------------
    # DO NOT DELETE OR EDIT NEXT LINE! This is needed for parser.
    # aggregate steady state marker
    # @include "../3_Model/input_aggregate_steady_state.jl"
    
    # write to XSS vector
    @writeXSS
    
    # produce indexes to access XSS etc.
    indexes               = produce_indexes(n_par, compressionIndexesVm, compressionIndexesCOP)
    indexes_aggr          = produce_indexes_aggr(n_par)
    
    @set! n_par.ntotal    = length(vcat(compressionIndexes...)) + (n_par.ny + n_par.nm - 2 + n_par.naggr) 
    @set! n_par.nstates   = n_par.ny +  n_par.nm - 2 + n_par.naggrstates + length(compressionIndexes[2]) # add to no. of states the coefficients that perturb the copula
    @set! n_par.ncontrols = length(vcat(compressionIndexes[1]...)) + n_par.naggrcontrols
    @set! n_par.LOMstate_save = zeros(n_par.nstates, n_par.nstates)
    @set! n_par.State2Control_save = zeros(n_par.ncontrols, n_par.nstates)
    @set! n_par.nstates_r   = copy(n_par.nstates)
    @set! n_par.ncontrols_r = copy(n_par.ncontrols)
    @set! n_par.ntotal_r    = copy(n_par.ntotal)
    @set! n_par.PRightStates= I[1:n_par.nstates,1:n_par.nstates]
    @set! n_par.PRightAll   = I[1:n_par.ntotal,1:n_par.ntotal]

    if n_par.n_agg_eqn != n_par.naggr - length(n_par.distr_names)
        @warn("Inconsistency in number of aggregate variables and equations")
        println(n_par.naggr - length(n_par.distr_names))
        println(n_par.n_agg_eqn)
    end

    return XSS, XSSaggr, indexes, indexes, indexes_aggr, compressionIndexes, n_par, #=
            =# m_par, CDF_SS, CDF_m, CDF_y, distrSS       
end

function copula_marg_equi_y(distr_i, grid_i, nx)
    
    CDF_i        = cumsum(distr_i[:])          # Marginal distribution (cdf) of liquid assets
    aux_marginal = collect(range(CDF_i[1], stop = CDF_i[end], length = nx))

    x2 = 1.0
    for i = 2:nx-1
        equi(x1)            = equishares(x1, x2, grid_i[1:end-1], distr_i[1:end-1], nx-1) 
        x2                  = find_zero(equi, (1e-9, x2))
        aux_marginal[end-i] = x2
    end

    aux_marginal[end]   = CDF_i[end]
    aux_marginal[1]     = CDF_i[1]
    aux_marginal[end-1] = CDF_i[end-1]
    copula_marginal     = copy(aux_marginal)
    jlast               = nx
    for i = nx-1:-1:1
        j = locate(aux_marginal[i], CDF_i) + 1
        if jlast == j 
            j -=1
        end
        jlast = j
        copula_marginal[i] = CDF_i[j]
    end
    return copula_marginal
end

function copula_marg_equi(distr_i, grid_i, nx)

    CDF_i        = cumsum(distr_i[:])          # Marginal distribution (cdf) of liquid assets
    aux_marginal = collect(range(CDF_i[1], stop = CDF_i[end], length = nx))

    x2 = 1.0
    for i = 1:nx-1
        equi(x1)            = equishares(x1, x2, grid_i, distr_i, nx) 
        x2                  = find_zero(equi ,(1e-9, x2))
        aux_marginal[end-i] = x2
    end

    aux_marginal[end] = CDF_i[end]
    aux_marginal[1]   = CDF_i[1]
    copula_marginal   = copy(aux_marginal)
    jlast             = nx
    for i = nx-1:-1:1
        j = locate(aux_marginal[i], CDF_i) + 1
        if jlast == j 
            j -=1
        end
        jlast = j
        copula_marginal[i] = CDF_i[j]
    end
    return copula_marginal
end

function equishares(x1, x2, grid_i, distr_i, nx) 
        
    FN_Wshares = cumsum(grid_i .* distr_i) ./ sum(grid_i .* distr_i)
    Wshares    = diff(mylinearinterpolate(cumsum(distr_i), FN_Wshares, [x1; x2]))
    dev_equi   = Wshares .- 1.0 ./ nx

    return dev_equi
end