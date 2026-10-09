
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from xgboost import XGBRegressor

SEED=20261009

ROUTES = pd.DataFrame([
('DEN-IBZ','Dénia–Ibiza',60,150,55),('DEN-FOR','Dénia–Formentera',58,145,54),('DEN-PMI','Dénia–Palma',125,230,75),
('VLC-IBZ','Valencia–Ibiza',88,185,63),('VLC-PMI','Valencia–Palma',140,245,80),('BCN-PMI','Barcelona–Palma',112,220,72),
('BCN-IBZ','Barcelona–Ibiza',151,260,86),('PMI-IBZ','Palma–Ibiza',69,160,58)], columns=['route_id','route_name','distance_nm','sched_minutes','base_fare'])
VESSELS = pd.DataFrame([
('LF-Aitana','fast_ro_pax',1250,350,31.0,0.030,5200),('LF-Montgó','fast_ro_pax',1150,320,30.5,0.031,5000),
('LF-Ifach','fast_ro_pax',1000,290,29.0,0.032,4700),('LF-CapBlanc','ro_pax',900,500,24.0,0.036,3800),
('LF-Segaria','ro_pax',1050,550,24.5,0.035,4000),('LF-Castell','ro_pax',850,450,23.5,0.037,3650),
('LF-Marina','ro_pax',800,430,23.0,0.038,3500),('LF-Levante','ro_pax',950,520,24.0,0.036,3900),
('LF-Garbí','eco_ro_pax',900,480,22.5,0.033,3300),('LF-Llebeig','eco_ro_pax',880,470,22.0,0.032,3200)],
columns=['vessel_id','vessel_class','pax_capacity','lane_m_capacity','design_speed_kn','base_fuel_coef','hotel_load_kw'])
CAPTAINS=[f'C{i:02d}' for i in range(1,19)]


def generate_voyages(n=7200, seed=SEED):
    rng=np.random.default_rng(seed)
    route_probs=np.array([.16,.11,.14,.14,.12,.12,.11,.10]); route_probs/=route_probs.sum()
    route_idx=rng.choice(len(ROUTES),n,p=route_probs)
    vessel_idx=rng.choice(len(VESSELS),n,p=[.10,.10,.09,.11,.11,.10,.10,.10,.10,.09])
    dates=pd.Timestamp('2026-01-01')+pd.to_timedelta(rng.integers(0,273,n),unit='D')
    hour=rng.choice([7,9,11,13,15,17,19,21],n,p=[.11,.12,.13,.12,.12,.14,.15,.11])
    r=ROUTES.iloc[route_idx].reset_index(drop=True)
    v=VESSELS.iloc[vessel_idx].reset_index(drop=True)
    season=np.where(pd.Series(dates).dt.month.isin([6,7,8,9]),1.12,np.where(pd.Series(dates).dt.month.isin([1,2]),.82,.96))
    weekend=np.where(pd.Series(dates).dt.dayofweek>=4,1.10,1.0)
    load_factor=np.clip(rng.beta(7,3,n)*season*weekend,0.35,0.99)
    pax=np.floor(v.pax_capacity.values*load_factor).astype(int)
    lane_load=np.clip(rng.beta(4,4,n)*(.8+.35*load_factor),0.12,0.99)
    lane_m=np.floor(v.lane_m_capacity.values*lane_load).astype(int)
    sea_state=np.clip(rng.poisson(1.35,n),0,5)
    wind_kn=np.clip(rng.normal(12+3*sea_state,6,n),0,42)
    headwind=np.clip(rng.normal(0,0.55,n),-1,1)
    wave_m=np.clip(.25+sea_state*.45+rng.normal(0,.18,n),.1,3.2)
    current_kn=np.clip(rng.normal(0,.55,n),-1.6,1.6)
    target_speed=np.where(v.vessel_class.values=='fast_ro_pax',rng.normal(27.0,1.5,n),rng.normal(20.8,1.2,n))
    weather_penalty=.10*sea_state+.015*wind_kn*np.maximum(headwind,0)
    speed_kn=np.clip(target_speed-weather_penalty+current_kn*.20+rng.normal(0,.55,n),16,31)
    cargo_tonnes = pax*0.09 + lane_m*0.48
    load_norm = np.clip((pax/v.pax_capacity.values)*.55 + (lane_m/v.lane_m_capacity.values)*.45,0,1.05)
    captains=rng.choice(CAPTAINS,n)
    captain_effects={c:e for c,e in zip(CAPTAINS,rng.normal(0,.012,len(CAPTAINS)))}
    # add a few stable but small practice effects
    captain_effects.update({'C03':-0.022,'C07':0.019,'C12':-0.017,'C16':0.021})
    cap_eff=np.array([captain_effects[c] for c in captains])
    # daily propulsion tonnage follows cubic speed; voyage fuel multiplies by duration.
    resistance = 1 + .055*sea_state + .012*wind_kn*np.maximum(headwind,0) + .07*load_norm + .025*np.maximum(wave_m-1,0)
    hours=r.distance_nm.values/np.maximum(speed_kn,1)
    propulsion_tph = v.base_fuel_coef.values * (speed_kn/20.0)**3 * 20.0
    aux_tph = v.hotel_load_kw.values/1000*0.18
    fuel_t = (propulsion_tph*hours*resistance + aux_tph*hours) * (1+cap_eff) * rng.lognormal(0,.035,n)
    fuel_t=np.clip(fuel_t,.8,None)
    # schedule / punctuality
    port_delay=np.clip(rng.gamma(1.6,3.4,n)+2.2*np.maximum(load_factor-.85,0)*10+rng.normal(0,1.5,n),0,35)
    sailing_minutes=hours*60
    actual_minutes=sailing_minutes+port_delay
    scheduled_minutes=np.where(v.vessel_class.values=='fast_ro_pax', r.distance_nm.values/28.0*60+15, r.distance_nm.values/21.0*60+15)
    delay_min=np.maximum(actual_minutes-scheduled_minutes,0)
    on_time15=delay_min<=15
    # economics
    fare = r.base_fare.values*(.88+.35*load_factor)*(1+0.08*np.where(pd.Series(dates).dt.month.isin([7,8]),1,0))*rng.normal(1,.035,n)
    revenue=pax*fare + lane_m*rng.uniform(1.5,2.7,n)
    fuel_price=rng.normal(690,35,n)
    fuel_cost=fuel_t*fuel_price
    co2_t=fuel_t*3.114
    ch4_t=fuel_t*np.where(v.vessel_class.values=='fast_ro_pax',0.00002,0.00003)
    n2o_t=fuel_t*0.00006
    co2e_t=co2_t + ch4_t*28 + n2o_t*265
    eua_price=np.clip(rng.normal(78,7,n),55,105)
    # analytical ETS accrual factor; legal exemptions not modelled
    ets_scope_factor=1.0
    ets_cost=co2e_t*eua_price*ets_scope_factor
    variable_other = r.distance_nm.values*(19+0.025*lane_m)+pax*.72
    fixed_trip=np.where(v.vessel_class.values=='fast_ro_pax',4800,4100)+r.distance_nm.values*4.0
    margin=revenue-fuel_cost-ets_cost-variable_other-fixed_trip
    df=pd.DataFrame({
        'voyage_id':[f'V{i+1:06d}' for i in range(n)],'date':pd.to_datetime(dates),'hour':hour,
        'route_id':r.route_id.values,'route_name':r.route_name.values,'distance_nm':r.distance_nm.values,'scheduled_minutes':np.where(v.vessel_class.values=='fast_ro_pax', r.distance_nm.values/28.0*60+15, r.distance_nm.values/21.0*60+15),
        'vessel_id':v.vessel_id.values,'vessel_class':v.vessel_class.values,'captain_id':captains,
        'pax_capacity':v.pax_capacity.values,'lane_m_capacity':v.lane_m_capacity.values,'passengers':pax,'lane_m_used':lane_m,'load_factor':load_factor,'cargo_tonnes':cargo_tonnes,
        'sea_state':sea_state,'wave_m':wave_m,'wind_kn':wind_kn,'headwind_index':headwind,'current_kn':current_kn,
        'speed_kn':speed_kn,'sailing_minutes':sailing_minutes,'port_delay_min':port_delay,'actual_minutes':actual_minutes,'delay_min':delay_min,'on_time15':on_time15,
        'fuel_tonnes':fuel_t,'fuel_price_eur_t':fuel_price,'fuel_cost_eur':fuel_cost,'co2_tonnes':co2_t,'co2e_tonnes':co2e_t,'eua_price_eur_t':eua_price,'ets_cost_eur':ets_cost,
        'avg_fare_eur':fare,'revenue_eur':revenue,'variable_other_eur':variable_other,'fixed_trip_eur':fixed_trip,'margin_eur':margin
    })
    return df.sort_values(['date','hour','voyage_id']).reset_index(drop=True)

FEATURES_NUM=['distance_nm','passengers','lane_m_used','load_factor','cargo_tonnes','sea_state','wave_m','wind_kn','headwind_index','current_kn','speed_kn','hour']
FEATURES_CAT=['route_id','vessel_id','vessel_class']

def model_matrix(df):
    X=df[FEATURES_NUM+FEATURES_CAT].copy()
    X=pd.get_dummies(X,columns=FEATURES_CAT,dtype=float)
    return X

def fit_xgb(df, random_state=SEED):
    X=model_matrix(df); y=df['fuel_tonnes'].values
    split=int(len(df)*.80)
    Xtr,Xte=X.iloc[:split],X.iloc[split:]; ytr,yte=y[:split],y[split:]
    model=XGBRegressor(n_estimators=420,max_depth=5,learning_rate=.045,subsample=.9,colsample_bytree=.9,reg_lambda=1.2,objective='reg:squarederror',random_state=random_state,n_jobs=4)
    model.fit(Xtr,ytr)
    pred=model.predict(Xte)
    metrics={'MAE_t':float(mean_absolute_error(yte,pred)),'RMSE_t':float(mean_squared_error(yte,pred)**.5),'R2':float(r2_score(yte,pred)),'WAPE_pct':float(np.sum(np.abs(yte-pred))/np.sum(np.abs(yte))*100),'n_train':len(ytr),'n_test':len(yte)}
    return model,X,metrics,pd.DataFrame({'actual':yte,'pred':pred,'residual':yte-pred},index=df.index[split:])

def cluster_voyages(df,k=5):
    cols=['speed_kn','distance_nm','fuel_tonnes','delay_min']
    Z=StandardScaler().fit_transform(df[cols])
    km=KMeans(n_clusters=k,n_init=20,random_state=SEED).fit(Z)
    out=df.copy(); out['voyage_cluster']=km.labels_
    prof=out.groupby('voyage_cluster')[cols].mean().round(3)
    sil=float(silhouette_score(Z,km.labels_,sample_size=min(2500,len(df)),random_state=SEED))
    return out,prof,sil

def adjusted_captains(df):
    X=model_matrix(df); y=df['fuel_tonnes'].values
    cv=KFold(n_splits=5,shuffle=True,random_state=SEED)
    base=XGBRegressor(n_estimators=260,max_depth=4,learning_rate=.055,subsample=.9,colsample_bytree=.9,reg_lambda=1.4,objective='reg:squarederror',random_state=SEED,n_jobs=4)
    pred=cross_val_predict(base,X,y,cv=cv,n_jobs=1,method='predict')
    z=df[['voyage_id','captain_id','route_id','vessel_id','load_factor','sea_state','wind_kn','speed_kn','fuel_tonnes']].copy()
    z['expected_fuel_t']=pred
    z['adj_residual_pct']=100*(z.fuel_tonnes-z.expected_fuel_t)/z.expected_fuel_t
    z['load_bin']=pd.qcut(z.load_factor,4,labels=False,duplicates='drop')
    z['weather_bin']=pd.cut(z.sea_state,[-1,0,1,2,3,5],labels=False)
    z['wind_bin']=pd.qcut(z.wind_kn,3,labels=False,duplicates='drop')
    z['stratum']=z.vessel_id.astype(str)+'|'+z.route_id.astype(str)+'|'+z.load_bin.astype(str)+'|'+z.weather_bin.astype(str)+'|'+z.wind_bin.astype(str)
    stratum_mean=z.groupby('stratum').fuel_tonnes.transform('mean')
    z['matched_residual_pct']=100*(z.fuel_tonnes-stratum_mean)/stratum_mean
    rows=[]
    for cap,g in z.groupby('captain_id'):
        n=len(g); mean=g.adj_residual_pct.mean(); se=g.adj_residual_pct.std(ddof=1)/np.sqrt(n)
        matched=g.matched_residual_pct.mean(); unique_pairs=g[['route_id','vessel_id']].drop_duplicates().shape[0]
        support=min(1,n/250)*min(1,unique_pairs/14)
        rows.append([cap,n,unique_pairs,mean,mean-1.96*se,mean+1.96*se,matched,support])
    res=pd.DataFrame(rows,columns=['captain_id','n_voyages','route_vessel_pairs','adjusted_fuel_residual_pct','ci95_low','ci95_high','matched_residual_pct','support_score'])
    res['comparison_valid']=(res.n_voyages>=120)&(res.support_score>=.60)
    res['interpretation']=np.where(~res.comparison_valid,'Insufficient comparable support',np.where((res.ci95_high<0)&(res.matched_residual_pct<0),'Lower fuel than context-adjusted expectation',np.where((res.ci95_low>0)&(res.matched_residual_pct>0),'Higher fuel than context-adjusted expectation','No robust difference')))
    return z,res.sort_values('adjusted_fuel_residual_pct')

def optimizer_grid(row, eua_price=80, fuel_price=700):
    speeds=np.arange(max(16,row['speed_kn']-4), min(31,row['speed_kn']+2.01), .25)
    rec=[]
    for s in speeds:
        hours=row['distance_nm']/s
        resistance=1+.055*row['sea_state']+.012*row['wind_kn']*max(row['headwind_index'],0)+.07*row['load_factor']+.025*max(row['wave_m']-1,0)
        base_coef={'fast_ro_pax':.0305,'ro_pax':.0364,'eco_ro_pax':.0325}.get(row['vessel_class'],.035)
        prop_tph=base_coef*(s/20)**3*20
        aux_tph=(4200/1000)*.18
        fuel=(prop_tph*hours*resistance+aux_tph*hours)
        co2e=fuel*3.114 + fuel*.000025*28 + fuel*.00006*265
        sail_min=hours*60
        total_min=sail_min+row['port_delay_min']
        delay=max(total_min-row['scheduled_minutes'],0)
        ets=co2e*eua_price
        fuelcost=fuel*fuel_price
        other=row['variable_other_eur']+row['fixed_trip_eur']
        margin=row['revenue_eur']-fuelcost-ets-other
        rec.append([s,fuel,co2e,total_min,delay,delay<=15,fuelcost,ets,margin])
    out=pd.DataFrame(rec,columns=['speed_kn','fuel_tonnes','co2e_tonnes','total_minutes','delay_min','on_time15','fuel_cost_eur','ets_cost_eur','margin_eur'])
    feasible=out[out.on_time15]
    best=(feasible if len(feasible) else out).sort_values(['margin_eur','fuel_tonnes'],ascending=[False,True]).iloc[0]
    return out,best
