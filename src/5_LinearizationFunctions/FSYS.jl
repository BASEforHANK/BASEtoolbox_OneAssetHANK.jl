@doc raw"""
    Fsys(X, XPrime, Xss, m_par, n_par, indexes, Γ, compressionIndexes, DC, IDC, DCD, IDCD)

Equilibrium error function: returns deviations from equilibrium around steady state.

Split computation into *Aggregate Part*, handled by [`Fsys_agg()`](@ref),
and *Heterogeneous Agent Part*.

# Arguments
- `X`,`XPrime`: deviations from steady state in periods t [`X`] and t+1 [`XPrime`]
- `Xss`: states and controls in steady state
- `Γ`, `DC`, `IDC`, `DCD`,`IDCD`: transformation matrices to retrieve marginal distributions [`Γ`],
    marginal value functions [`DC`,`IDC`], and the (linear) interpolant of the copula [`DCD`,`IDCD`] from deviations
- `indexes`,`compressionIndexes`: access `Xss` by variable names
    (DCT coefficients of compressed ``V_m`` and ``V_k`` in case of `compressionIndexes`)

# Example
```jldoctest
julia> # Solve for steady state, construct Γ,DC,IDC as in SGU()
julia> Fsys(zeros(ntotal),zeros(ntotal),XSS,m_par,n_par,indexes,Γ,compressionIndexes,DC,IDC)
*ntotal*-element Array{Float64,1}:
 0.0
 0.0
 ...
 0.0
```
"""
function Fsys(X::AbstractArray, XPrime::AbstractArray, Xss::Array{Float64,1}, m_par::ModelParameters,
              n_par::NumericalParameters, indexes::IndexStruct, Γ::Array{Array{Float64,2},1},
              compressionIndexes::Array{Array{Int,1},1}, DC::Array{Array{Float64,2},1},
              IDC::Array{Adjoint{Float64,Array{Float64,2}},1}, DCD::Array{Array{Float64,2},1},
              IDCD::Array{Adjoint{Float64,Array{Float64,2}},1}; only_F = true) 
              # The function call with Duals takes
              # Reserve space for error terms
    F = zeros(eltype(X),size(X))

    ############################################################################
    #            I. Read out argument values                                   #
    ############################################################################
    # rougly 10% of computing time, more if uncompress is actually called

    ############################################################################
    # I.1. Generate code that reads aggregate states/controls
    #      from steady state deviations. Equations take the form of:
    # r       = exp.(Xss[indexes.rSS] .+ X[indexes.r])
    # rPrime  = exp.(Xss[indexes.rSS] .+ XPrime[indexes.r])
    ############################################################################

    # @generate_equations(aggr_names)
    @generate_equations()


    ############################################################################
    # I.2. Read out  perturbed distributions
    ############################################################################

    # Copula parameters (deviations from steads state)
    θD           = uncompress(compressionIndexes[2], X[indexes.COP], DCD, IDCD)
    COP_Dev      = reshape(copy(θD[:]),(n_par.nm_copula, n_par.ny_copula))
    COP_Dev      = pdf_to_cdf(COP_Dev)
   
    θDPrime      = uncompress(compressionIndexes[2], XPrime[indexes.COP], DCD, IDCD)
    COP_DevPrime = reshape(copy(θDPrime),(n_par.nm_copula, n_par.ny_copula))
    COP_DevPrime = pdf_to_cdf(COP_DevPrime)

    # marginal distributions (pdfs, including steady state)
    distr_m       = Xss[indexes.distr_m_SS] .+ Γ[1] * X[indexes.distr_m]
    distr_y       = Xss[indexes.distr_y_SS] .+ Γ[2] * X[indexes.distr_y]
    
    distr_m_Prime = Xss[indexes.distr_m_SS] .+ Γ[1] * XPrime[indexes.distr_m]
    distr_y_Prime = Xss[indexes.distr_y_SS] .+ Γ[2] * XPrime[indexes.distr_y]

    # marginal distributions (cdfs) 
    CDF_m         = cumsum(distr_m[:])
    CDF_y         = cumsum(distr_y[:])

    CDF_m_Prime   = cumsum(distr_m_Prime[:])
    CDF_y_Prime   = cumsum(distr_y_Prime[:])
    
    ############################################################################
    # I.3. Read out steady state distributions
    ############################################################################

    # steads state cdfs (on value grid)
    CDF_m_SS      = cumsum(Xss[indexes.distr_m_SS]) .+ zeros(eltype(θD),n_par.nm)
    CDF_y_SS      = cumsum(Xss[indexes.distr_y_SS]) .+ zeros(eltype(θD),n_par.ny)
    
    # steady state copula (on copula grid)
    COP_SS        = reshape(Xss[indexes.COPSS]      .+ zeros(eltype(θD),1) , (n_par.nm, n_par.ny))
    COP_SS        = pdf_to_cdf(COP_SS)

    # steady state copula marginals (cdfs) 
    s_m_m         = n_par.copula_marginal_m .+ zeros(eltype(θD),1)
    s_m_y         = n_par.copula_marginal_y .+ zeros(eltype(θD),1)
    
    ############################################################################
    # I.4. Produce perturbed joint distribution using the copula
    ############################################################################
    # Copula(x::AbstractVector,y::AbstractVector,z::AbstractVector) = 
    # myAkimaInterp3(CDF_m_SS, CDF_k_SS, CDF_y_SS, COP_SS, x, y, z) .+
    # myAkimaInterp3(s_m_m, s_m_k, s_m_y, COP_Dev, x, y, z)
   
    Copula(x::AbstractVector,y::AbstractVector) = 
    mylinearinterpolate2(CDF_m_SS,  CDF_y_SS, COP_SS, x, y) .+
    mylinearinterpolate2(s_m_m, s_m_y, COP_Dev, x, y)

    CDF_joint     = Copula(CDF_m[:], CDF_y[:]) # roughly 5% of time
    PDF_joint     = cdf_to_pdf(CDF_joint)

    ############################################################################
    # I.5 uncompressing policies/value functions
    ###########################################################################

    VmPrime = mutil.(exp.(Xss[indexes.VmSS] .+ uncompress(compressionIndexes[1], XPrime[indexes.Vm], DC,IDC)))
    
    ############################################################################
    #           II. Auxiliary Variables                                        #
    ############################################################################
    # Transition Matrix Productivity

    Π                  = n_par.Π .+ zeros(eltype(X),1)[1]
    PP                 = ExTransition(m_par.ρ_h,n_par.bounds_y,sqrt(σ))
    Π[1:end-1,1:end-1] = PP.*(1.0-m_par.ζ)

    ############################################################################
    #           III. Error term calculations (i.e. model starts here)          #
    ############################################################################

    ############################################################################
    #           III. 1. Aggregate Part #
    ############################################################################
    F       = Fsys_agg(X, XPrime, Xss, PDF_joint, m_par, n_par, indexes)

    
    ###########################################################################
    # Replace distributional objects used in Fsys_agg
    ###########################################################################
    # Average Human Capital =
    # average productivity (at the productivit grid, used to normalize to 0)
    tax_prog_scale  = (m_par.γ + m_par.τ_prog)/((m_par.γ + τprog))
    H               = dot(distr_y[1:end-1],n_par.grid_y[1:end-1])
    TotAsset        = dot(n_par.grid_m,distr_m[:])
    Htact           = dot(distr_y[1:end-1],(n_par.grid_y[1:end-1]/H).^(tax_prog_scale))
    BDact           = -sum(distr_m.*(n_par.grid_m.<0).*n_par.grid_m)
    
    ############################################################################
    #               III. 2. Heterogeneous Agent Part                           #
    ############################################################################
    # Incomes
    incgross, inc, eff_int = incomes(n_par, m_par, mcw, A, RB./π, τprog, τlev, H, Ht,w,N, profits, unionprofits, av_tax_rate)
    inc[1][:, 1:end-1] = inc[1][:, 1:end-1] .+ log(tauLS) .* exp(Xss[indexes.YSS]) .* (n_par.mesh_y[:, 1:end-1] / H)

    # Calculate Taxes
    tax_prog_scale = (m_par.γ + m_par.τ_prog)/((m_par.γ + τprog))
    LC              = mcw *w.*N ./ Ht 
    taxrev          = ((n_par.grid_y/H).^tax_prog_scale.*LC)-τlev.*((n_par.grid_y/H).^tax_prog_scale.*LC).^(1.0-τprog)
    taxrev[end]     = n_par.grid_y[end].*profits - τlev.*(n_par.grid_y[end].*profits).^(1.0-τprog)
    incgrossaux     = ((n_par.grid_y/H).^tax_prog_scale.*LC)
    incgrossaux[end]= n_par.grid_y[end].*profits
    av_tax_rate_up  = dot(distr_y, taxrev)./(dot(distr_y,incgrossaux))
                             
    # Calculate optimal policies
    # expected margginal values
    eff_intPrime = (RBPrime./πPrime .+ (m_par.Rbar.*(n_par.mesh_m.<=0.0))) 

    EVmPrime = eff_intPrime.*(reshape(VmPrime,(n_par.nm,n_par.ny))*Π')

    c_n_star, m_n_star = EGM_policyupdate(EVmPrime,RB./π,1.0,inc,n_par,m_par, false) # policy iteration (assume HH hold deposits)

    # Update marginal values
    Vm_new = mutil(c_n_star)  # update expected marginal values time t

    # Update distribution
    dist_aux        = DirectTransition(m_n_star, PDF_joint, Π, n_par)
    PDF_jointPrime  = reshape(dist_aux,n_par.nm,n_par.ny)

    #----------------------------------------------------------------------------------------
    # Calculate Error Terms
    #----------------------------------------------------------------------------------------
    # Error terms on marginal values (controls)
    Vm_err        = log.(invmutil.(Vm_new)) .- reshape(Xss[indexes.VmSS],(n_par.nm,n_par.ny))
    Vm_thet       = compress(compressionIndexes[1], Vm_err, DC,IDC)
    F[indexes.Vm] = X[indexes.Vm] .- Vm_thet

    # Error Terms on marginal distribution (in levels, states)
    F[indexes.distr_m] = (dropdims(sum(PDF_jointPrime,dims=(2)),dims=(2)) - distr_m_Prime)[1:end-1]
    F[indexes.distr_y] = ((distr_y'*Π)[:] .- distr_y_Prime[:])[1:end-1]

    # Error Terms on Copula (states)
    # Deviation of iterated copula from fixed copula
    CopulaDevPrime(x::AbstractVector,y::AbstractVector) = 
    mylinearinterpolate2(CDF_m_Prime,  CDF_y_Prime, pdf_to_cdf(PDF_jointPrime), x, y) .-
    mylinearinterpolate2(CDF_m_SS,  CDF_y_SS, COP_SS, x, y)

    CDF_Dev      = CopulaDevPrime(s_m_m,  s_m_y) # interpolate deviations on copula grid
    COP_thet     = compress(compressionIndexes[2], cdf_to_pdf(CDF_Dev - COP_DevPrime), DCD, IDCD) # calculate DCT of deviations
    
    F[indexes.COP] = COP_thet

    # Calculate distribution statistics (generalized moments)
    distr_m_T, distr_y_T, TOP10WshareT, TOP10IshareT,TOP10InetshareT, 
        GiniWT, GiniXT,GiniCT, GiniIT, sdlogyT = distrSummaries(PDF_joint, c_n_star, n_par, inc, incgross, m_par)

    # Error Term on prices/aggregate summary vars (logarithmic, controls)
    #F[indexes.τlev]           = av_tax_rate - av_tax_rate_up
    #F[indexes.T]              = log(T) - log(dot(distr_y, taxrev) + av_tax_rate * (unionprofits))

    F[indexes.K]              = log.(K + B)  - log.(TotAsset)
    F[indexes.BD]             = log.(BD)              - log.(BDact)
    F[indexes.Ht]             = log.(Ht)              - log.(Htact)
    
    # Error Terms on  distribution summaries
    F[indexes.GiniC]          = log.(GiniC)           - log.(GiniCT)
    F[indexes.GiniX]          = log.(GiniX)           - log.(GiniXT)
    F[indexes.TOP10Ishare]    = log.(TOP10Ishare)     - log.(TOP10IshareT);
    F[indexes.TOP10Inetshare] = log.(TOP10Inetshare)  - log.(TOP10InetshareT);
    F[indexes.TOP10Wshare]    = log.(TOP10Wshare)     - log.(TOP10WshareT);
    F[indexes.sdlogy]         = log.(sdlogy)          - log.(sdlogyT)


    if only_F
        return F
    else
        return F, c_a_star, m_a_star, k_a_star, c_n_star, m_n_star, Vk_new, Vm_new, taxrev
    end
end
