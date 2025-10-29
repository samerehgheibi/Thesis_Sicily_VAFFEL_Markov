# -*- coding: utf-8 -*-
# line_cli_dual.py — Dual-mode fragility (SIMPLE or FULL eq.12) with clear prints

import os, sys, math
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- Paths ----------
FAILURES_CSV = r"C:\Users\samer\OneDrive\Desktop\thesis\دیتا بیس ها\per_way_node_day_wind_minmax_failures_one_node_per_year_no_repeat.csv"
TOWERS_CSV   = r"C:\Users\samer\OneDrive\Desktop\thesis\دیتا بیس ها\tower.csv"
CIRCUITS_CSV = r"C:\Users\samer\OneDrive\Desktop\thesis\در مورد اطلاعات تاور ولتاژ و همه ریزه کاری ها\way_id_220_380_circuits_from_cables با فرمول.csv"
OUTPUT_DIR   = r"C:\Users\samer\OneDrive\Desktop\چک"

# ---------- Utils ----------
def say(x): print(x, flush=True)
def ask_float(prompt, default=None):
    while True:
        s = input(prompt).strip()
        if s=="" and default is not None: return float(default)
        try: return float(s)
        except: print("Please enter a numeric value.", flush=True)

def try_read_csv(path, name):
    say(f"[1] Reading {name}: {path}")
    for enc in ("utf-8-sig","utf-8","cp1256","latin1"):
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            say(f"    -> OK (C) {enc}, shape={df.shape}")
            return df
        except Exception as e1:
            say(f"    -> C failed ({enc}): {e1}")
            try:
                df = pd.read_csv(path, encoding=enc, engine="python", on_bad_lines='skip')
                say(f"    -> OK (python, skip) {enc}, shape={df.shape}")
                return df
            except Exception as e2:
                say(f"    -> python failed ({enc}): {e2}")
    say(f"[!] Cannot read {name}."); sys.exit(1)

def pick_col(df_or_cols, candidates, required=False, name="df"):
    cols = df_or_cols if isinstance(df_or_cols,(list,tuple)) else list(df_or_cols.columns)
    for c in candidates:
        if c in cols: return c
    if required:
        say(f"[!] Required column not found in {name}. Tried: {candidates}")
        say(f"    Available in {name}: {cols}"); sys.exit(1)
    return None

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    from math import radians, sin, cos, asin, sqrt
    dphi = radians(lat2-lat1); dlmb = radians(lon2-lon1)
    phi1 = radians(lat1); phi2 = radians(lat2)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlmb/2)**2
    return 2*R*asin(sqrt(a))

# Robust Φ(z)
def normal_cdf(z):
    z = np.asarray(z, dtype=float)
    try:
        from math import erf
        erf_vec = np.vectorize(erf)
        return 0.5*(1.0 + erf_vec(z/np.sqrt(2.0)))
    except Exception:
        try:
            from scipy.special import erf as sp_erf
            return 0.5*(1.0 + sp_erf(z/np.sqrt(2.0)))
        except Exception:
            t = 1.0/(1.0+0.2316419*np.abs(z))
            poly = t*(0.319381530 + t*(-0.356563782 + t*(1.781477937 + t*(-1.821255978 + t*1.330274429))))
            phi  = (1/np.sqrt(2*np.pi))*np.exp(-0.5*z*z)
            cdf  = 1 - phi*poly
            cdf  = np.where(z<0, 1-cdf, cdf)
            return cdf

def lognormal_cdf(x, mu, sigma):
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    m = x > 0
    if np.any(m):
        out[m] = normal_cdf((np.log(x[m]) - np.log(mu))/sigma)
    return out

# ---------- Load ----------
os.makedirs(OUTPUT_DIR, exist_ok=True)
df_fail = try_read_csv(FAILURES_CSV, "failures")
df_tow  = try_read_csv(TOWERS_CSV,   "towers")
df_circ = try_read_csv(CIRCUITS_CSV, "circuits")

# Column mapping
way_candidates  = ["way_id","WAY_ID","wayId","id_way","way","WayID"]
lat_candidates  = ["lat","latitude","LAT","Latitude","y","Y"]
lon_candidates  = ["lon","longitude","LON","Longitude","x","X"]
seq_candidates  = ["order","sequence","seq","index","idx","position","node_index"]
circ_candidates = ["line_circuits","circuits","num_circuits","circuits_count","circuit","Circuiti","cables_to_circuits"]
volt_candidates = ["voltage_kV","voltage_kv","voltage","Voltage_KV","Voltage_kv","kv","KV","Nominal_kV","nominal_kv"]
wind_candidates = ["wind_speed","ws","wind","wind_max","max_wind","V","v100","wind100"]
time_candidates = ["datetime","date","day","timestamp","time","Date"]

fail_way = pick_col(df_fail, way_candidates, True, "failures")
tow_way  = pick_col(df_tow,  way_candidates, True, "towers")
circ_way = pick_col(df_circ, way_candidates, True, "circuits")
tow_lat  = pick_col(df_tow, lat_candidates, True, "towers")
tow_lon  = pick_col(df_tow, lon_candidates, True, "towers")
tow_seq  = pick_col(df_tow, seq_candidates, False, "towers")
circ_col = pick_col(df_circ, circ_candidates, False, "circuits")
volt_col = pick_col(df_circ, volt_candidates, False, "circuits")

# ---------- Choose WAYs ----------
ids = sorted(df_fail[fail_way].dropna().astype(str).unique())
print("------------------------------------------------")
print("Selectable WAY IDs (with at least one failure):")
for i,w in enumerate(ids,1): print(f"{i:>3}. {w}")
print("------------------------------------------------")
user = input("Paste one or more WAY IDs separated by commas (or type ALL): ").strip()
selected = ids if user.lower()=="all" else [s.strip() for s in user.split(",") if s.strip()]
selected = [s for s in selected if s in ids]
if not selected: print("[!] No valid WAY IDs."); sys.exit(0)

# ---------- Helpers ----------
def segments_geometry(way_id):
    sub = df_tow[df_tow[tow_way].astype(str)==str(way_id)].dropna(subset=[tow_lat, tow_lon]).copy()
    if sub.empty: return np.nan, 0, [], "No tower coords."
    if tow_seq and tow_seq in sub.columns: sub=sub.sort_values(tow_seq); note=None
    else: sub=sub.reset_index(drop=True); note="Sequence column not found; used input order."
    lengths=[]; total=0.0
    for i in range(len(sub)-1):
        d=haversine_km(float(sub.iloc[i][tow_lat]), float(sub.iloc[i][tow_lon]),
                       float(sub.iloc[i+1][tow_lat]), float(sub.iloc[i+1][tow_lon]))
        lengths.append(d); total+=d
    return total, len(lengths), lengths, note

def circuits_voltage(way_id):
    r = df_circ[df_circ[circ_way].astype(str)==str(way_id)]
    if r.empty: return None, None
    row = r.iloc[0]
    circuits=None
    if circ_col and circ_col in r.columns:
        raw=str(row[circ_col]).strip().lower()
        try: circuits=int(float(raw))
        except:
            if any(k in raw for k in ["2","double","doppio"]): circuits=2
            elif any(k in raw for k in ["1","single","singolo"]): circuits=1
    voltage=None
    if volt_col and volt_col in r.columns:
        v=pd.to_numeric(row[volt_col], errors="coerce"); voltage=None if pd.isna(v) else float(v)
    return circuits, voltage

def wind_summary(sub_fail):
    wcol = pick_col(sub_fail, wind_candidates, True, "failures(selected)")
    tcol = pick_col(sub_fail, time_candidates, False, "failures(selected)")
    ws = pd.to_numeric(sub_fail[wcol], errors="coerce").dropna()
    if ws.empty: return None
    vmax, mean = float(ws.max()), float(ws.mean())
    p99  = float(np.percentile(ws.values.astype("float32"), 99))
    years=1.0; date="N/A"
    if tcol:
        tt=pd.to_datetime(sub_fail[tcol], errors="coerce")
        if tt.notna().any():
            days=(tt.max()-tt.min()).days
            if days>0: years=max(1.0, days/365.25)
        try: date=str(sub_fail.loc[ws.idxmax(), tcol])
        except: pass
    return dict(wcol=wcol, tcol=tcol, vmax=vmax, mean=mean, p99=p99, years=years, date=date)

def fit_fragility(get_p_line_series, years_k, lambda_B_line, anchor_exposure):
    def objective(theta):
        mu, sigma = theta
        if (mu<=0) or (sigma<=0): return 1e9
        p_line = get_p_line_series(mu, sigma)
        lam_hat = p_line.sum()/max(years_k,1e-9)
        p_anchor = float(lognormal_cdf(np.array([anchor_exposure]), mu, sigma)[0])
        return (lambda_B_line - lam_hat)**2 + (lambda_B_line**2)*(p_anchor-0.01)**2

    try:
        import scipy.optimize as opt
        inits = [(1.0,0.8),(5.0,0.8),(10.0,1.2)]
        best=(1e18, (1.0,0.8))
        for mu0,sg0 in inits:
            res = opt.minimize(objective, x0=[mu0,sg0], method="L-BFGS-B",
                               bounds=[(1e-6, 1e6),(0.25,5.0)])
            if res.fun < best[0]: best=(res.fun, tuple(res.x))
        mu, sigma = best[1]
        p_line = get_p_line_series(mu, sigma)
        lam_hat = p_line.sum()/max(years_k,1e-9)
        return float(mu), float(sigma), float(lam_hat), "scipy"
    except Exception:
        mu_grid = np.geomspace(1e-4, 1e3, 50)
        sigma_grid = np.linspace(0.25,5.0,40)
        best=(1e18, None, None)
        for mu in mu_grid:
            for sg in sigma_grid:
                val = objective([mu,sg])
                if val < best[0]: best=(val, mu, sg)
        mu, sigma = best[1], best[2]
        p_line = get_p_line_series(mu, sigma)
        lam_hat = p_line.sum()/max(years_k,1e-9)
        return float(mu), float(sigma), float(lam_hat), "grid"

# ---------- Main ----------
for wid in selected:
    print(f"\n>>> WAY ID: {wid}")
    L, Nseg, seg_lengths, note = segments_geometry(wid)
    circuits, kv = circuits_voltage(wid)
    if kv is not None: print(f"- Voltage: {kv:.1f} kV")
    if circuits is not None:
        print(f"- Circuit Type: {'Double' if circuits>=2 else 'Single'} (circuits={circuits})")
    if L==L: print(f"- Total Length: {L:.5f} km")
    print(f"- Number of Segments: {Nseg}")
    if seg_lengths: print(f"- Max Segment Length: {max(seg_lengths):.5f} km")
    if note: print(f"  [Note] {note}")

    sub = df_fail[df_fail[fail_way].astype(str)==str(wid)].copy()
    if sub.empty: print("- No failure rows for this WAY. Skip."); continue

    stat = wind_summary(sub)
    if stat:
        print(f"- Historical Max Wind: {stat['vmax']:.2f} m/s ({stat['date']})")
        print(f"- Mean Wind: {stat['mean']:.2f} m/s ; 99th Perc: {stat['p99']:.2f} m/s")
        years_k = stat['years']
    else:
        print("- Wind Summary: N/A"); years_k=1.0

    # --- BAYES
    print("-- Please input parameters --")
    N_fail = ask_float("N (failures to use): ")
    T_years = ask_float("T (time period in years): ")
    if L and L>0 and T_years>0 and N_fail>=0:
        lam_per_km = N_fail/(L*T_years)
        beta = (1/lam_per_km) if lam_per_km>0 else float("inf")
        alpha=1.0
        lam_B_per_km = (alpha+N_fail)/(beta+T_years) if np.isfinite(beta) else float("nan")
        lam_B_line   = lam_B_per_km * L
        print(f"- Prior λ (per km·year): {lam_per_km:.6f}")
        print(f"- Posterior λ_B (per km·year): {lam_B_per_km:.6f}")
        print(f"- β = 1/λ = {beta:.6f}")
        print("\n[Definition] Bayesian β parameter:")
        print("  β_Bayes = 1 / λ")
        print("  -> Represents the scale parameter of the Gamma prior distribution for failure rate.")
        print("  -> Used for statistical updating of line failure probability over time.\n")

        print(f"- Posterior λ_B_line (per line·year): {lam_B_line:.6f}")
    else:
        print("- Skipped Bayesian calc."); lam_B_line = np.nan

    wcol = stat["wcol"] if stat else pick_col(sub, wind_candidates, True, "failures(selected)")
    sub["wind_speed"] = pd.to_numeric(sub[wcol], errors="coerce")
    sub = sub[sub["wind_speed"].notna()].copy()
    w_thresh = ask_float("Wind threshold w_thresh (m/s): ")
    alpha_w  = ask_float("Alpha_w (scale, default=1 -> Enter): ", default=1.0)
    print(f"- Using threshold={w_thresh} m/s, alpha_w={alpha_w}")

    sub["wind_exposure"] = alpha_w * L * np.clip(sub["wind_speed"]-w_thresh, 0.0, None)**3
    out_main = os.path.join(OUTPUT_DIR, f"failures_with_exposure_{wid}.csv")
    keep_cols = [c for c in sub.columns if c not in ["wind_speed"]] + ["wind_speed","wind_exposure"]
    sub.to_csv(out_main, index=False, columns=keep_cols)
    print(f"- Saved: {out_main}")

    mode = input("Model mode? (SIMPLE / FULL): ").strip().upper()
    if mode not in ("SIMPLE","FULL"):
        mode = "SIMPLE"
    print(f"- MODE = {mode}")

    wind_series = sub["wind_speed"].values.astype(float)
    lmax = max(seg_lengths) if seg_lengths else 0.0
    seg_lengths_arr = np.asarray(seg_lengths, dtype=float)

    if mode=="SIMPLE":
        expo_rep = alpha_w * lmax * np.clip(wind_series - w_thresh, 0.0, None)**3
        anchor_expo = alpha_w * lmax * (0.5**3)
        def get_p_line_series(mu, sigma):
            return lognormal_cdf(expo_rep, mu, sigma)
    else:
        if len(seg_lengths_arr)==0:
            expo_rep = alpha_w * lmax * np.clip(wind_series - w_thresh, 0.0, None)**3
            anchor_expo = alpha_w * lmax * (0.5**3)
            def get_p_line_series(mu, sigma):
                return lognormal_cdf(expo_rep, mu, sigma)
        else:
            base = np.clip(wind_series - w_thresh, 0.0, None)**3
            expos = alpha_w * np.outer(base, seg_lengths_arr)
            anchor_expo = alpha_w * (seg_lengths_arr.max() if seg_lengths_arr.size>0 else 0.0) * (0.5**3)
            def get_p_line_series(mu, sigma):
                p_seg = lognormal_cdf(expos, mu, sigma)
                keep = np.clip(1.0 - p_seg, 1e-12, 1.0)
                p_line = 1.0 - np.prod(keep, axis=1)
                return p_line

    if not (np.isfinite(lam_B_line) and lam_B_line>0):
        print("- Skip fragility fit (invalid λ_B_line)."); continue
    mu, sigma, lam_hat, how = fit_fragility(get_p_line_series, years_k, lam_B_line, anchor_expo)
    print("- Fragility fit results")
    print(f"    mu = {mu:.6f}, sigma = {sigma:.6f}, method={how}")
    print(f"    lambda_hat (from p_t series) = {lam_hat:.6f}")
    print(f"    target lambda_B_line        = {lam_B_line:.6f}")
    if lam_B_line>0: print(f"    relative error = {100.0*abs(lam_hat-lam_B_line)/lam_B_line:.2f}%")
    # --------- Extra plot: Fragility Curves with Different μ and σ ---------
    try:
        xs = np.linspace(0.1, max(sub["wind_speed"].max() * 1.2, 40), 400)
        plt.figure(figsize=(8, 5))

        params = [
            (mu, sigma, f"Fitted (μ={mu:.2f}, σ={sigma:.2f})"),
            (mu * 0.7, sigma * 0.8, f"Lower μ/σ (μ={mu*0.7:.2f}, σ={sigma*0.8:.2f})"),
            (mu * 1.3, sigma * 1.2, f"Higher μ/σ (μ={mu*1.3:.2f}, σ={sigma*1.2:.2f})"),
        ]

        for m, s, label in params:
            y = lognormal_cdf(xs, m, s)
            plt.plot(xs, y, label=label)

        plt.title(f"Fragility Curves with Different μ and σ — WAY {wid}")
        plt.xlabel("Wind exposure ")
        plt.ylabel("Probability of Failure")
        plt.legend()
        plt.grid(True)

        out_mu_sigma = os.path.join(OUTPUT_DIR, f"fragility_mu_sigma_{wid}.png")
        plt.savefig(out_mu_sigma, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"- Saved μ–σ comparison plot: {out_mu_sigma}")
    except Exception as e:
        print(f"- Plot error (μ–σ comparison): {e}")
    if mode=="SIMPLE":
        p_line = get_p_line_series(mu, sigma)
        frag = pd.DataFrame({"wind_speed": wind_series,
                             "exposure_rep": alpha_w * lmax * np.clip(wind_series - w_thresh, 0.0, None)**3,
                             "p_line": p_line})
    else:
        p_line = get_p_line_series(mu, sigma)
        frag = pd.DataFrame({"wind_speed": wind_series, "p_line": p_line})
        frag["exposure_rep"] = alpha_w * (lmax if lmax>0 else (seg_lengths_arr.max() if seg_lengths_arr.size else 0.0)) \
                               * np.clip(wind_series - w_thresh, 0.0, None)**3

    tcol = stat.get("tcol") if stat else None
    if tcol and tcol in sub.columns: frag["datetime"] = sub[tcol].values
    out_frag = os.path.join(OUTPUT_DIR, f"fragility_series_{wid}.csv")
    cols = ["datetime","wind_speed","exposure_rep","p_line"] if "datetime" in frag.columns else ["wind_speed","exposure_rep","p_line"]
    frag.to_csv(out_frag, index=False, columns=cols)
    print(f"- Saved: {out_frag}")

    # --------- Plots ---------
    try:
        xs = np.linspace(0, max(frag["exposure_rep"].max()*1.1, (mu*3 if np.isfinite(mu) else 1.0)), 400)
        p_x = lognormal_cdf(xs, mu, sigma)
        plt.figure(); plt.plot(xs, p_x)
        plt.xlabel("Wind exposure (representative segment)")
        plt.ylabel("Probability of failure")
        plt.title(f"WAY {wid} — Failure probability vs wind exposure ({mode})")
        out_pngA = os.path.join(OUTPUT_DIR, f"probability_vs_exposure_{wid}.png")
        plt.savefig(out_pngA, bbox_inches="tight", dpi=150); plt.close()
        print(f"- Saved plot: {out_pngA}")
    except Exception as e:
        print(f"- Plot error (exposure): {e}")

    try:
        w_min = float(max(0.0, wind_series.min())); w_max = float(wind_series.max())
        xsw = np.linspace(w_min, w_max, 400)
        if mode=="SIMPLE":
            expo_w = alpha_w * lmax * np.clip(xsw - w_thresh, 0.0, None)**3
            p_w = lognormal_cdf(expo_w, mu, sigma)
        else:
            if len(seg_lengths_arr)==0:
                expo_w = alpha_w * (lmax if lmax>0 else 0.0) * np.clip(xsw - w_thresh, 0.0, None)**3
                p_w = lognormal_cdf(expo_w, mu, sigma)
            else:
                base = np.clip(xsw - w_thresh, 0.0, None)**3
                expos = alpha_w * np.outer(base, seg_lengths_arr)
                p_seg = lognormal_cdf(expos, mu, sigma)
                p_w = 1.0 - np.prod(np.clip(1.0 - p_seg, 1e-12, 1.0), axis=1)
        plt.figure(); plt.plot(xsw, p_w)
        plt.xlabel("Wind speed (m/s)"); plt.ylabel("Probability of failure")
        plt.title(f"WAY {wid} — Failure probability vs wind speed ({mode})")
        out_pngB = os.path.join(OUTPUT_DIR, f"probability_vs_windspeed_{wid}.png")
        plt.savefig(out_pngB, bbox_inches="tight", dpi=150); plt.close()
        print(f"- Saved plot: {out_pngB}")
        
    except Exception as e:
        print(f"- Plot error (wind): {e}")
        print("\n-------------------------------------------------------------")
        print("[Definition] Dependency β parameter:")
        print("  q_t = 1 - exp(-β_dependency * (w_t - w_th))")
        print("  -> β_dependency controls how quickly the second circuit’s failure probability grows with wind speed.")
        print("  -> It is independent from Bayesian β (1/λ) and has physical meaning related to wind–line coupling.")
        print("-------------------------------------------------------------\n")
        # -------- Plot: Historical failures vs modelled probability --------
    try:
        # پیدا کردن ستون‌های تاریخ و باد
        time_col = pick_col(sub, ["date", "datetime", "timestamp", "day"], True, "failures")
        wind_col = pick_col(sub, ["wind_speed", "wind", "max_wind", "wind_max"], True, "failures")

        sub_plot = sub.dropna(subset=[time_col, wind_col]).copy()
        sub_plot[time_col] = pd.to_datetime(sub_plot[time_col], errors="coerce")
        sub_plot = sub_plot.dropna(subset=[time_col])
        sub_plot = sub_plot.sort_values(time_col)

        # محاسبه exposure و احتمال خرابی
        Lmax = max(seg_lengths) if seg_lengths else 0.0
        exposure = alpha_w * Lmax * np.clip(sub_plot[wind_col] - w_thresh, 0.0, None)**3
        sub_plot["failure_prob"] = lognormal_cdf(exposure, mu, sigma)

        # رسم نمودار احتمال
        plt.figure(figsize=(10, 5))
        plt.plot(sub_plot[time_col], sub_plot["failure_prob"],
                 color="steelblue", linewidth=1.8, label="Modelled failure probability")

        # فقط تاریخ‌هایی که failure واقعی داشتیم (ستون failure == 1)
        if "failure" in sub_plot.columns:
            fail_dates = sub_plot.loc[sub_plot["failure"] == 1, time_col].unique()
            for fd in sorted(fail_dates):
                plt.axvline(fd, color="red", linestyle="--", linewidth=1.2, alpha=0.8,
                            label="Actual failure" if fd == fail_dates[0] else "")
        else:
            print("⚠️ Column 'failure' not found — no red lines will be shown.")

        plt.title(f"WAY {wid} — Historical Failures vs Modelled Probability")
        plt.xlabel("Date and Time")
        plt.ylabel("Failure Probability")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()

        out_hist = os.path.join(OUTPUT_DIR, f"historical_failures_vs_probability_{wid}.png")
        plt.savefig(out_hist, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"- ✅ Saved improved historical failure plot: {out_hist}")

    except Exception as e:
        print(f"- ⚠️ Error plotting historical failures for WAY {wid}: {e}")

        # ==========================================================
# ============================================================
# SMART MARKOV CHAIN SECTION (Automatic Single/Double Handling)
# ============================================================

print("\n[Markov Chain Simulation Started]")

# 1️⃣ بارگذاری خودکار فایل فیلرها برای این WAY ID
failures_path = os.path.join(OUTPUT_DIR, f"failures_with_exposure_{wid}.csv")
if os.path.exists(failures_path):
    df_failures = pd.read_csv(failures_path)
    print(f"  -> Loaded failures file: {failures_path} (rows={len(df_failures)})")
else:
    raise FileNotFoundError(f"❌ Could not find failure file: {failures_path}")

# 2️⃣ اطمینان از وجود ستون‌های کلیدی
required_cols = ["node_id", "date", "wind_speed"]
for col in required_cols:
    if col not in df_failures.columns:
        raise KeyError(f"❌ Column '{col}' not found in {failures_path}")

# 3️⃣ اگه ستون failure وجود نداشت، مقدار صفر بهش بده
if "failure" not in df_failures.columns:
    df_failures["failure"] = 0

# 4️⃣ اگه fragility_probs ساخته نشده بود، بر اساس سرعت باد بسازش
if 'fragility_probs' not in locals():
    winds = df_failures["wind_speed"].values
    # تابع شکنندگی: احتمال خرابی با افزایش سرعت باد بیشتر میشه
    fragility_probs = np.clip((winds / winds.max()) ** 3, 0, 1)

# ============================================================
# 5️⃣ انتخاب مدل بر اساس تعداد مدارها
# ============================================================

if circuits >= 2:
    print("\n[Markov Chain Simulation for Double Circuit Line]")
    delta = 0.3  # میزان وابستگی بین مدارها

    P_f1 = fragility_probs
    P_f2 = fragility_probs
    P_f2_cond = np.clip(P_f2 + delta * (1 - P_f2), 0, 1)

    P_both = P_f1 * P_f2_cond
    P_one = P_f1 * (1 - P_f2_cond) + (1 - P_f1) * P_f2
    P_none = 1 - (P_one + P_both)

else:
    print("\n[Single Circuit Line – Simplified Probability Model]")
    # چون فقط یه مدار داریم، مدار دوم وجود نداره
    P_failure = fragility_probs
    P_none = 1 - P_failure
    P_one = P_failure
    P_both = np.zeros_like(P_failure)

# ============================================================
# 6️⃣ ساخت دیتافریم خروجی
# ============================================================

df_markov = pd.DataFrame({
    "node_id": df_failures["node_id"],
    "date": df_failures["date"],
    "wind_speed": df_failures["wind_speed"],
    "P_none": P_none,
    "P_one_failed": P_one,
    "P_both_failed": P_both
})

# ============================================================
# 7️⃣ نمایش در ترمینال فقط برای خرابی‌های واقعی
# ============================================================

real_failures = df_failures[df_failures["failure"] == 1].copy()
if not real_failures.empty:
    real_failures = real_failures.merge(
        df_markov, on=["node_id", "date", "wind_speed"], how="left"
    )
    print("\n[Markov Chain Results for REAL Failures Only]")
    print(real_failures[["node_id", "date", "wind_speed", "P_none", "P_one_failed", "P_both_failed"]]
          .to_string(index=False))
else:
    print("\n⚠️ No real failures found in dataset.")

# ============================================================
# 8️⃣ ذخیره خروجی
# ============================================================

out_csv = os.path.join(OUTPUT_DIR, f"markov_full_{wid}.csv")
df_markov.to_csv(out_csv, index=False)
print(f"\n✅ Saved Markov results (all nodes): {out_csv}")
