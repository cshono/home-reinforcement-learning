function update_sps(m::SmartHomeMDP, s::SmartHomeState, a::SmartHomeAction, rng)
    if a.dhsp != 0 || a.dcsp != 0
        # If action=adjust, deterministically set thermostat
        hsp = min(max(s.hsp+a.dhsp, SP_MIN), SP_MAX)
        csp = min(max(s.csp+a.dcsp, SP_MIN), SP_MAX)
        #return s.hsp + a.dhsp, s.csp + a.dcsp
        return hsp, csp
    else
        # Else, probabilistic adjustment based on thermal comfort
        if rand(rng) < m.prob_sp_adj
            # Adjust setpoints
            setpoints = [round(x) for x in rand(rng, Distributions.Uniform(m.tcomf_lo, m.tcomf_hi),2)] # TODO: Figure out why not working
            center = (m.tcomf_lo + m.tcomf_hi)/2
            #setpoints = [round(min(max(x, m.tcomf_lo), m.tcomf_hi)) for x in rand(rng, Normal(center,2),2)]
            return minimum(setpoints), maximum(setpoints)
        else
            # Maintain previous SPs
            return s.hsp, s.csp
        end
    end
end

function hv_model(hsp, csp, odt, rng)
    # Fitted regression model based hv_model data analysis.
    # Model was fit to predict kWh/day
    # d_hv = a*cdd + b*hdd + c + N(0,RMSE)

    cdd = max(odt-csp,0) # cdd in an hour
    hdd = max(hsp-odt,0) # hdd in an hour

    a = 2.1898; b = 0.9476; c = 0.5394; RMSE = 6.146;
    error = rand(rng, Normal(0,RMSE), 1)[1]

    d_hv = a*cdd + b*hdd + c + error
    d_hv = round(d_hv*1000/float(TOD_RESOLUTION, digits=-2) # kWh/day -> Wh/hr(to nearest 100Wh)

    return d_hv
end

function hv_model_simple(hsp, csp, odt, rng)
    # Extremely simplified model to work with SIMPLE DATA
    cdd = max(odt-csp,0) # cdd in an hour
    hdd = max(hsp-odt,0) # hdd in an hour

    a = 1; b = 0.2; RMSE = 0.05;
    error = rand(rng, Normal(0,RMSE), 1)[1]

    d_hv = a*cdd + b*hdd + error
    d_hv = max(min(round(d_hv),10), 0)

    return d_hv
end

function update_d_(d_, tod)
    # TODO: fit model dependent on TOD
    # TODO: adjust clamp when realistic data (maybe functionalize clamp)
    # Maybe fit a BN to this data?
    noise_d_ = float(LOAD_NOISE)
    d_ = float(LOAD_SCHEDULE[tod]* AVG_DAILY_LOAD) #OVERRIDE PREVIOUS VALUE

    d_ += rand(rng, Normal(0,noise_d_), 1)[1] # Probabilistic Change
    d_ = max(min(round(d_),10), 0) # Clamp to 0-10 int
    #d_ = max(min(round(d_),10000,0)
    return d_
end

function update_occ(occ, tod)
    # TODO: fit model dependent on TOD
    # TODO: adjust clamp when realistic data (maybe functionalize clamp)
    # Maybe fit a BN to this data?
    p_change_occ = 0.3
    p_occ = OCC_SCHEDULE[tod]

    #return rand(rng) < p_change_occ ? !occ : occ
    return rand(rng) < p_occ
end

function update_odt(tod)
    # Fitted to a sine wave with noise
    # TODO: need to adjust equation params when converting to realistic data
    # TODO: adjust clamp when realistic data (maybe functionalize clamp)
    # TODO: fit model dependent on TOD
    # Maybe fit a BN to this data?

    noise_odt = float(ODT_NOISE);
    temp_max = float(ODT_MAX);
    temp_min = float(ODT_MIN);
    tod_per_day = float(TOD_RESOLUTION);

    #noise_odt = 1;
    #temp_max = 7.5;
    #temp_min = 2.5;
    #tod_per_day = 5;

    amp = (temp_max - temp_min) / 2
    mean = (temp_max + temp_min) / 2

    odt = amp*sin((tod+1.75)*pi/(tod_per_day/2)) + mean
    odt += rand(rng, Normal(0,noise_odt), 1)[1]
    #odt = max(min(round(odt),temp_max), temp_min) # Clamp to 0-10 int

    return odt
end

function update_tou(tod)
    #TOU_SCHEDULE = [2,2,3,4,2];
    # TODO: Change tou_schedule to take as model input rather than hard code
    tou = TOU_SCHEDULE[tod]
    return tou
end


# MDP Generative Model
function POMDPs.gen(m::SmartHomeMDP, s::SmartHomeState, a::SmartHomeAction, rng)
    # transition model
    t = s.t + 1 # Deterministic, fixed
    #tod = rem(s.tod + 1, 24) # Deterministic, fixed
    tod = rem(s.tod, TOD_RESOLUTION) + 1 # TODO: THIS IS FOR SIMPLE MODEL

    #d_ = m.D_[s.t+1] # Deterministic, fixed (should/could it be probabilistic?)
    #occ = OCC[s.t+1] # Deterministic, fixed (should/could it be probabilistic?)
    #odt = ODT[s.t+1] # Deterministic, fixed
    #tou = TOU[s.t+1] # Deterministic, fixed

    d_ = update_d_(s.d_, tod) # Probabilistic
    occ = update_occ(s.occ, tod) # Probabilistic
    odt = update_odt(tod) # Probabilistic
    tou = update_tou(tod) # By Schedule


    (hsp, csp) = update_sps(m, s, a, rng)
    #d_hv = hv_model(hsp, csp, odt, rng) # Based on actual fit
    #d_hv = hv_model_simple(hsp, csp, s.odt, rng)
    d_hv = hv_model(hsp, csp, s.odt, rng)
    soc = min(max(s.soc + a.c, 0), SOC_MAX)
    rmt = min(max(hsp,odt),csp)

    sp = SmartHomeState(d_hv, d_, soc, rmt, occ, hsp, csp, tod, odt, tou, t)


    # observation model
    # N/A

    # reward model
    r = -tou * (s.d_ + s.d_hv + soc - s.soc)
    r += (s.rmt > m.tcomf_hi || s.rmt < m.tcomf_lo) ? m.penalty_discomf : 0
    r += (s.soc > m.soc_max || s.soc < 0) ? m.penalty_soc : 0
    r += (s.hsp > s.csp) ? m.penalty_sp : 0 # HSP must be less than or equal to CSP


    # create and return a NamedTuple
    return (sp=sp, r=r) # For MDP
end
