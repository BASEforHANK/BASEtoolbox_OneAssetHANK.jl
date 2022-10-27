# Set aggregate steady state variabel values
ASS       = 1.0
ZSS       = 1.0
ZISS      = 1.0
μSS       = m_par.μ
μwSS      = m_par.μw
τprogSS   = m_par.τ_prog
τlevSS    = m_par.τ_lev

σSS          = 1.0
τprog_obsSS  = 1.0
GshockSS     = 1.0
RshockSS     = 1.0
TprogshockSS = 1.0
SshockSS  = 1.0

RBSS      = rSS
LPSS      = (rSS / RBSS)^4
LPXASS    = (rSS / RBSS)^4
ISS       = m_par.δ_0 * KSS

πSS       = 1.0
πwSS      = 1.0
rRBSS = (RBSS./πSS)^4

BSS       = TotAssetSS*(m_par.ψ)
BDSS      = -sum(distr_m_SS.*(n_par.grid_m.<0).*n_par.grid_m)

# Calculate taxes and government expenditures
TSS       = dot(distr_y_SS, taxrev) + av_tax_rateSS*((1.0 .- 1.0 ./ m_par.μw).*wSS.*NSS)
GSS       = TSS - (RBSS - 1.0)*BSS

CSS       = (YSS - m_par.δ_0 * KSS - GSS - m_par.Rbar*BDSS)

qBSS = 1.0 ./ (RBSS .- 1.0 + m_par.δ_B)
bankprofitsSS = 1.0
qBlagSS  = qBSS

qSS       = 1.0
mcSS      = 1.0 ./ m_par.μ
mcwSS     = 1.0 ./ m_par.μw
mcwwSS    = wSS * mcwSS
uSS       = 1.0
profitsSS = (1.0 - mcSS).*YSS
unionprofitsSS = (1.0 - mcwSS) .* wSS .* NSS
tauLSSS = 1.0

BYSS   = BSS / YSS
TYSS   = TSS / YSS
TlagSS = TSS

YlagSS = YSS
BlagSS = BSS
GlagSS = GSS
IlagSS = ISS
wlagSS = wSS
qlagSS = qSS
ClagSS = CSS
av_tax_ratelagSS = av_tax_rateSS
τproglagSS       = τprogSS
GlagSS = GSS

YgrowthSS = 1.0
BgrowthSS = 1.0
IgrowthSS = 1.0
wgrowthSS = 1.0
CgrowthSS = 1.0
TgrowthSS = 1.0
HtSS      = 1.0

KGSS =1.0
