# This file has been automatically generated by PreprocessInputs.jl. Any user inputs might be overwritten!




@doc raw"""
    Fsys_agg(X, XPrime, Xss, distrSS, m_par, n_par, indexes)

Return deviations from aggregate equilibrium conditions.

`indexes` can be both `IndexStruct` or `IndexStructAggr`; in the latter case
(which is how function is called by [`SGU_estim()`](@ref)), variable-vectors
`X`,`XPrime`, and `Xss` only contain the aggregate variables of the model.
"""
function Fsys_agg(X::AbstractArray, XPrime::AbstractArray, Xss::Array{Float64,1},distrSS::AbstractArray, m_par::ModelParameters,
              n_par::NumericalParameters, indexes::Union{IndexStructAggr,IndexStruct})
              # The function call with Duals takes
              # Reserve space for error terms
    F = zeros(eltype(X),size(X))
    ############################################################################
    #            I. Read out argument values                                   #
    ############################################################################

    ############################################################################
    # I.1. Generate code that reads aggregate states/controls
    #      from steady state deviations. Equations take the form of:
    # r       = exp.(Xss[indexes.rSS] .+ X[indexes.r])
    # rPrime  = exp.(Xss[indexes.rSS] .+ XPrime[indexes.r])
    ############################################################################

    # @generate_equations(aggr_names)
    @generate_equations()

    # Take aggregate model from model file


#------------------------------------------------------------------------------
# THIS FILE CONTAINS THE "AGGREGATE" MODEL EQUATIONS, I.E. EVERYTHING  BUT THE 
# HOUSEHOLD PLANNING PROBLEM. THE lATTER IS DESCRIBED BY ONE EGM BACKWARD STEP AND 
# ONE FORWARD ITERATION OF THE DISTRIBUTION.
#
# AGGREGATE EQUATIONS TAKE THE FORM 
# F[EQUATION NUMBER] = lhs - rhs
#
# EQUATION NUMBERS ARE GENEREATED AUTOMATICALLY AND STORED IN THE INDEX STRUCT
# FOR THIS THE "CORRESPONDING" VARIABLE NEEDS TO BE IN THE LIST OF STATES 
# OR CONTROLS.
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# AUXILIARY VARIABLES ARE DEFINED FIRST
#------------------------------------------------------------------------------
    # Elasticities and steepness from target markups for Phillips Curves
    η               = μ / (μ - 1.0)                                 # demand elasticity
    κ               = η * (m_par.κ / m_par.μ) * (m_par.μ - 1.0)     # implied steepness of phillips curve
    ηw              = μw / (μw - 1.0)                               # demand elasticity wages
    κw              = ηw * (m_par.κw / m_par.μw) * (m_par.μw - 1.0) # implied steepness of wage phillips curve

    # Capital Utilization
    MPK_SS          = exp(Xss[indexes.rSS]) - 1.0 + m_par.δ_0       # stationary equil. marginal productivity of capital
    δ_1             = MPK_SS                                        # normailzation of utilization to 1 in stationary equilibrium
    δ_2             = δ_1 .* m_par.δ_s                              # express second utilization coefficient in relative terms
    # Auxiliary variables
    Kserv           = K * u                                         # Effective capital
    MPKserv         = mc .* Z .* m_par.α .* (Kserv ./ N) .^(m_par.α - 1.0)      # marginal product of Capital
    depr            = m_par.δ_0 + δ_1 * (u - 1.0) + δ_2 / 2.0 * (u - 1.0)^2.0   # depreciation

    Wagesum         = N * w                                         # Total wages in economy t
    WagesumPrime    = NPrime * wPrime                               # Total wages in economy t+1

    YREACTION       = Ygrowth                                       # Policy target variable Y

    distr_ySS       = sum(distrSS, dims=1)                          # marginal income distribution
    Htact           = dot(distr_ySS[1:end-1],(n_par.grid_y[1:end-1]/n_par.H).^((m_par.γ + m_par.τ_prog)/(m_par.γ + τprog)))
    
    # tax progressivity variabels used to calculate e.g. total taxes
    tax_prog_scale  = (m_par.γ + m_par.τ_prog) / ((m_par.γ + τprog))                        # scaling of labor disutility including tax progressivity
    incgross        = ((n_par.grid_y ./ n_par.H) .^ tax_prog_scale .* mcw .* w .* N ./ Ht)  # gross income
    incgross[end]   = (n_par.grid_y[end] .* profits)                                        # gross profit income
    inc             = τlev .* (incgross .^ (1.0 .- τprog))                                  # net income
    taxrev          = incgross .- inc                                                       # taxes paid by income
    TaxAux          = dot(distr_ySS, taxrev)                                                  # total income taxes paid w/o unionprofits
    IncAux          = dot(distr_ySS, incgross)                                                # total taxable income w/o unionprofits
    
    
    ############################################################################
    #           Error term calculations (i.e. model starts here)          #
    ############################################################################

    #-------- States -----------#
    # Error Term on exogeneous States
    # Shock processes
    F[indexes.Gshock]       = log.(GshockPrime)         - m_par.ρ_Gshock * log.(Gshock)     # primary deficit shock
    F[indexes.Tprogshock]   = log.(TprogshockPrime)     - m_par.ρ_Pshock * log.(Tprogshock) # tax shock

    F[indexes.Rshock]       = log.(RshockPrime)         - m_par.ρ_Rshock * log.(Rshock)     # Taylor rule shock
    F[indexes.Sshock]       = log.(SshockPrime)         - m_par.ρ_Sshock * log.(Sshock)     # uncertainty shock

    # Stochastic states that can be directly moved (no feedback)
    F[indexes.A]            = log.(APrime)              - m_par.ρ_A * log.(A)               # (unobserved) Private bond return fed-funds spread (produces goods out of nothing if negative)
    F[indexes.Z]            = log.(ZPrime)              - m_par.ρ_Z * log.(Z)               # TFP
    F[indexes.ZI]           = log.(ZIPrime)             - m_par.ρ_ZI * log.(ZI)             # Investment-good productivity

    F[indexes.μ]            = log.(μPrime./m_par.μ)     - m_par.ρ_μ * log.(μ./m_par.μ)      # Process for markup target
    F[indexes.μw]           = log.(μwPrime./m_par.μw)   - m_par.ρ_μw * log.(μw./m_par.μw)   # Process for w-markup target

    # Endogeneous States (including Lags)
    F[indexes.σ]            = log.(σPrime)              - (m_par.ρ_s * log.(σ) + (1.0 - m_par.ρ_s) *
                                m_par.Σ_n * (log(Y) .- Xss[indexes.YSS]) + log(Sshock))                     # Idiosyncratic income risk (contemporaneous reaction to business cycle)
    # Lags
    F[indexes.Ylag]         = log(YlagPrime)    - log(Y)
    F[indexes.Blag]         = log(BlagPrime) - log(B)
    F[indexes.Ilag]         = log(IlagPrime)    - log(I)
    F[indexes.wlag]         = log(wlagPrime)    - log(w)
    F[indexes.Tlag]         = log(TlagPrime)    - log(T)
    F[indexes.qlag]         = log(qlagPrime)    - log(q)
    F[indexes.Clag]         = log(ClagPrime)    - log(C)
    F[indexes.av_tax_ratelag] = log(av_tax_ratelagPrime) - log(av_tax_rate)
    F[indexes.τproglag]     = log(τproglagPrime) - log(τprog)
    F[indexes.Glag]         = log(GlagPrime)    - log(G)
    F[indexes.qBlag]        = log(qBlagPrime)   - log(qB)

    # Growth rates
    F[indexes.Ygrowth]      = log(Ygrowth)      - log(Y/Ylag)
    F[indexes.Tgrowth]      = log(Tgrowth)      - log(T/Tlag)
    F[indexes.Bgrowth]      = log(Bgrowth)      - log(B/Blag)
    F[indexes.Igrowth]      = log(Igrowth)      - log(I/Ilag)
    F[indexes.wgrowth]      = log(wgrowth)      - log(w/wlag)
    F[indexes.Cgrowth]      = log(Cgrowth)      - log(C/Clag)
    
    #------------ Economic Model from here ---------------
    #  Taylor rule and interest rates
    F[indexes.RB]           = log(RBPrime) - Xss[indexes.RBSS] -
                              ((1 - m_par.ρ_R) * m_par.θ_π) .* log(π) -
                              ((1 - m_par.ρ_R) * m_par.θ_Y) .* log(YREACTION) - 
                              m_par.ρ_R * (log.(RB) - Xss[indexes.RBSS])  - log(Rshock)

    # Expected real bond yield
    F[indexes.rRB]          = log(rRB) - 4.0*log((1.0./qB - m_par.δ_B + 1.0)/πPrime)

    # Tax rule
    F[indexes.τprog]        = log(τprog) - m_par.ρ_P * log(τproglag)  - 
                              (1.0 - m_par.ρ_P) *(Xss[indexes.τprogSS]) - 
                              (1.0 - m_par.ρ_P) * m_par.γ_YP * log(YREACTION) -
                              (1.0 - m_par.ρ_P) * m_par.γ_BP * (log(B)- Xss[indexes.BSS]) - 
                              log(Tprogshock)

    # First tax parameter (level)
    F[indexes.τlev]         = av_tax_rate - TaxAux ./ IncAux  # Union profits are taxed at average tax rate
    
    F[indexes.T]            = log(T) - log(TaxAux + av_tax_rate * unionprofits) # total taxes including tax on unionprofits
    
    F[indexes.av_tax_rate]  = log(av_tax_rate) - m_par.ρ_τ * log(av_tax_ratelag)  - # rule for average tax rate
                                (1.0 - m_par.ρ_τ) * Xss[indexes.av_tax_rateSS] -
                                (1.0 - m_par.ρ_τ) * m_par.γ_Yτ * log(YREACTION) -
                                (1.0 - m_par.ρ_τ) * m_par.γ_Bτ * (log(B) - Xss[indexes.BSS])
    
    # --------- Controls ------------
    # Government variables
    F[indexes.G]      = log(G) - (m_par.ρ_G) * log(Glag)  - (1.0 - m_par.ρ_G) *(Xss[indexes.GSS]) -
                          (1.0 - m_par.ρ_G) * m_par.γ_YG * log(YREACTION) + (1.0 - m_par.ρ_G) * m_par.γ_BG *  (log(B) -
                          Xss[indexes.BSS])  - log(Gshock)
                          
    F[indexes.tauLS]  = log(tauLS) .* exp(Xss[indexes.YSS]) - m_par.γ_Bτ2 * log(BgrowthPrime)
    

    F[indexes.B]       =  log(BPrime .- (1.0 .- m_par.δ_B) .* B ./ qBlag ./ π .* qB) .- 
                          log(B ./ qBlag ./ π .- T .+ G .+ log(tauLS) .* exp(Xss[indexes.YSS]))

    # Phillips Curve to determine equilibrium markup, output, factor incomes 
    F[indexes.mc]      = (log.(π)- Xss[indexes.πSS]) - κ *(mc - 1 ./ μ ) -
                                m_par.β * ((log.(πPrime) - Xss[indexes.πSS]) .* YPrime ./ Y) 
    
    # Wage Dynamics
    F[indexes.πw]      = log.(w ./ wlag) - log.(πw ./ π)                   # Definition of real wage inflation

    # Wage Phillips Curve 
    F[indexes.mcw]     = (log.(πw)- Xss[indexes.πwSS]) - (κw * (mcw - 1 ./ μw) +
                                m_par.β * ((log.(πwPrime) - Xss[indexes.πwSS]) .* WagesumPrime ./ Wagesum))
    # worker's wage = mcw * firm's wage
    F[indexes.mcww]    = log.(mcww) - log.(mcw * w)                        # wages that workers receive

    
    # Capital utilisation
    F[indexes.u]       = MPKserv  -  q * (δ_1 + δ_2 * (u - 1.0))           # Optimality condition for utilization

    # Prices
    F[indexes.r]       = log.(r) - log.(1 + MPKserv * u - q * depr )       # rate of return on capital

    F[indexes.w]       = log.(w) - log.(wage(Kserv, Z * mc, N, m_par))     # wages that firms pay
  
    F[indexes.q]       = 1.0 - ZI * q * (1.0 - m_par.ϕ / 2.0 * (Igrowth - 1.0)^2.0 - # price of capital investment adjustment costs
                         m_par.ϕ * (Igrowth - 1.0) * Igrowth)  -
                         m_par.β * ZIPrime * qPrime * m_par.ϕ * (IgrowthPrime - 1.0) * (IgrowthPrime)^2.0
    
    # Profits
    F[indexes.profits] = log.(profits)  - log.(Y .* (1.0 - mc) .+ q .* (KPrime .- (1.0 .- depr) .* K) .- I .+ log.(bankprofits))  # profits of the monopolistic resellers

    F[indexes.bankprofits]  = log.(bankprofits) .- (B./ π.* ((1.0 .- m_par.δ_B) .* qB ./ qBlag .+ 1.0./ qBlag .- RB) .+
                                                    K.*((q .+ r .- 1.0) .- qlag.*RB./π)) # gains/losses from owning capital stock
    F[indexes.unionprofits] = log.(unionprofits)  - log.(w.*N .* (1.0 - mcw))  # profits of the monopolistic unions
                     
    # Asset market prices and premia
    F[indexes.LPXA]         = log.(LPXA)                - 4 * (log((qPrime + rPrime - 1.0)/q)  - log.((1.0./qB - m_par.δ_B + 1.0)/πPrime)) # ex-ante liquidity premium
    F[indexes.LP]           = log.(LP)         -          4 * (log((q + r - 1.0)/qlag) -  log.((1.0./qBlag - m_par.δ_B + 1.0)/π))  # ex-post liquidity premium
    F[indexes.qB]           = log.(RBPrime .* qB) .- log.(1.0  .+ (1.0 .- m_par.δ_B) .* qBPrime)
    F[indexes.π]            = log.(RBPrime./πPrime.*APrime) - log.((qPrime + rPrime .- 1.0)./q) # Bond market clearing: Capital arbitrage
    

    # Aggregate Quantities
    F[indexes.I]            = KPrime .-  K .* (1.0 .- depr)  .- ZI .* I .* (1.0 .- m_par.ϕ ./ 2.0 .* (Igrowth -1.0).^2.0)           # Capital accumulation equation
    F[indexes.N]            = log.(N) - log.(((1.0 - τprog) * τlev * (mcw .* w).^(1.0 - τprog)).^(1.0 / (m_par.γ + τprog)) .* Ht)   # labor supply
    F[indexes.Y]            = log.(Y) - log.(Z .* N .^(1.0 .- m_par.α) .* Kserv .^m_par.α)                                          # production function
    F[indexes.C]            = log.(Y .- G .- I .- BD*m_par.Rbar .+ (A .- 1.0) .* RB .* B ./ π) .- log(C)                            # Resource constraint

    # Error Term on prices/aggregate summary vars (logarithmic, controls), here difference to SS value averages
    F[indexes.BY]           = log.(BY)    - log.(B/Y)                                                               # Bond to Output ratio
    F[indexes.TY]           = log.(TY)    - log.(T/Y)                                                               # Tax to output ratio
    
    # Distribution summary statistics used in this file (using the steady state distrubtion in case). 
    # Lines here generate a unit derivative (distributional summaries do not change with other aggregate vars).
    F[indexes.K]            = log.(K + B)     - log.(exp.(Xss[indexes.KSS]).+ exp.(Xss[indexes.BSS]))                                                        # Capital market clearing
    F[indexes.BD]           = log.(BD)    - Xss[indexes.BDSS]                                                       # IOUs            
    
    # Add distributional summary stats that do change with other aggregate controls/prices and with estimated parameters
    F[indexes.Ht]           = log.(Ht)    - log.(Htact)

    # other dsitributional statistics not used in other aggregate equations and not changing with parameters, 
    # but potentially with other aggregate variables are NOT included here. They are found in FSYS.



    # @include "../3_Model/input_aggregate_model.jl"

    return F
end


