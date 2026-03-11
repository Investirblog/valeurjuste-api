"""
ValeurJuste — Backend FastAPI v2
Lancer avec : python -m uvicorn main:app --reload --port 8000

Nouveautés v2 :
- EV/EBITDA historique pour les cycliques
- Cache en mémoire (TTL 4h)
- Détection de profil affinée
- Route /api/top pour le top valorisations dynamique
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import re
import warnings
warnings.filterwarnings("ignore")

app = FastAPI(title="ValeurJuste API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── CACHE EN MÉMOIRE ────────────────────────────────────────────────────────
_cache: dict = {}
CACHE_TTL = 4 * 3600  # 4 heures

def cache_get(key: str):
    entry = _cache.get(key)
    if entry and (time.time() - entry["ts"]) < CACHE_TTL:
        return entry["data"]
    return None

def cache_set(key: str, data: dict):
    _cache[key] = {"ts": time.time(), "data": data}

# ─── UNIVERS TOP VALORISATIONS ───────────────────────────────────────────────
TOP_UNIVERSE = {
    "CAC 40": [
        "AI.PA","AIR.PA","BN.PA","BNP.PA","CA.PA","CAP.PA","CS.PA","DG.PA",
        "DSY.PA","ENGI.PA","GLE.PA","HO.PA","KER.PA","LR.PA","MC.PA","ML.PA",
        "OR.PA","ORA.PA","PUB.PA","RI.PA","RMS.PA","RNO.PA","SAF.PA","SAN.PA",
        "SGO.PA","SW.PA","TTE.PA","VIE.PA","VIV.PA","WLN.PA",
    ],
    "DAX": [
        "ADS.DE","ALV.DE","BAYN.DE","BMW.DE","BAS.DE","CON.DE","DB1.DE",
        "DBK.DE","DHL.DE","DTE.DE","EOAN.DE","FRE.DE","HEI.DE","HEN3.DE",
        "IFX.DE","MBG.DE","MRK.DE","MUV2.DE","RWE.DE","SAP.DE","SIE.DE",
        "VOW3.DE","VNA.DE",
    ],
    "S&P 100": [
        "AAPL","ABBV","ABT","ACN","ADBE","AMGN","AMZN","AXP","BA","BAC",
        "BLK","BMY","C","CAT","CMCSA","COP","COST","CRM","CSCO","CVS","CVX",
        "DHR","DIS","EMR","FDX","GD","GE","GILD","GM","GOOG","GS","HD","HON",
        "IBM","INTC","JNJ","JPM","KO","LLY","LMT","LOW","MA","MCD","MDT",
        "MMM","MO","MRK","MS","MSFT","NEE","NFLX","NKE","NVDA","ORCL","PEP",
        "PFE","PG","PM","QCOM","RTX","SBUX","T","TGT","TMO","TSLA","TXN",
        "UNH","UNP","UPS","V","VZ","WFC","WMT","XOM",
    ],
}

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def clean_name(name: str) -> str:
    return re.sub(r"^(L['']|Le |La |Les )", "", name or "", flags=re.IGNORECASE).strip()


def detect_profile(info: dict) -> dict:
    sector      = info.get("sector", "")
    beta        = info.get("beta") or 1.0
    rev_growth  = info.get("revenueGrowth") or 0.0

    CYCLICAL = {"Energy", "Materials", "Industrials", "Consumer Cyclical", "Real Estate"}
    GROWTH   = {"Technology", "Communication Services"}
    HEALTH   = {"Healthcare"}

    if sector in GROWTH or rev_growth > 0.15:
        return {"type":"croissance","label":"Valeur Croissance","icon":"🚀",
                "metric":"pe","metric_label":"P/E",
                "description":"Profil croissance : P/E élevé justifié par l'expansion des bénéfices. Le PEG ratio (P/E ÷ croissance) est la jauge clé."}

    if sector in HEALTH:
        return {"type":"mature","label":"Valeur Défensive","icon":"🏥",
                "metric":"pe","metric_label":"P/E",
                "description":"Secteur défensif à revenus récurrents. Le P/E historique est une référence fiable."}

    if sector in CYCLICAL or beta > 1.35:
        return {"type":"cyclique","label":"Valeur Cyclique","icon":"🔄",
                "metric":"evebitda","metric_label":"EV/EBITDA",
                "description":"Les cycliques ont un P/E trompeur en bas de cycle. L'EV/EBITDA lisse ces distorsions et donne une image plus juste de la valorisation."}

    return {"type":"mature","label":"Valeur Mature","icon":"🏛️",
            "metric":"pe","metric_label":"P/E",
            "description":"Croissance régulière et prévisible. Le P/E moyen historique est la référence la plus fiable pour estimer la juste valeur."}


def get_annual_prices(ticker_obj, years: int) -> pd.Series:
    end   = datetime.today()
    start = end - timedelta(days=365 * years + 90)
    hist  = ticker_obj.history(start=start.strftime("%Y-%m-%d"),
                               end=end.strftime("%Y-%m-%d"), interval="1mo")
    if hist.empty:
        raise ValueError("Aucune donnée de prix disponible.")
    hist.index = pd.to_datetime(hist.index)
    hist["Year"] = hist.index.year
    return hist.groupby("Year")["Close"].last()


def compute_pe_series(ticker_obj, annual_price: pd.Series, info: dict):
    try:
        # income_stmt donne jusqu'à 4 ans, financials aussi — on prend les deux et on fusionne
        eps_s = pd.Series(dtype=float)
        for attr in ["income_stmt", "financials"]:
            try:
                fin = getattr(ticker_obj, attr)
                if "Basic EPS" in fin.index:
                    s = fin.loc["Basic EPS"]
                elif "Diluted EPS" in fin.index:
                    s = fin.loc["Diluted EPS"]
                else:
                    net_income = fin.loc["Net Income"]
                    s = net_income / info.get("sharesOutstanding", 1)
                s.index = pd.to_datetime(s.index).year
                s = s.sort_index()
                # Fusionne sans écraser les valeurs existantes
                for yr, val in s.items():
                    if yr not in eps_s.index:
                        eps_s[yr] = val
            except Exception:
                continue
        eps_s = eps_s.sort_index()
    except Exception:
        eps_s = pd.Series(dtype=float)

    pe_series = {}
    for year in annual_price.index:
        if year in eps_s.index and eps_s[year] and eps_s[year] > 0:
            pe = float(annual_price[year]) / float(eps_s[year])
            # Filtre les P/E aberrants (splits non ajustés, années déficitaires proches de 0)
            if 3 <= pe <= 120:
                pe_series[year] = round(pe, 1)

    if len(pe_series) < 3:
        current_pe = info.get("trailingPE")
        if not current_pe or not (3 <= current_pe <= 120):
            return None
        avg_multiple = round(float(current_pe), 1)
    else:
        # Moyenne robuste : exclut les 10% extrêmes si assez de points
        vals = sorted(pe_series.values())
        if len(vals) >= 6:
            trim = max(1, len(vals) // 10)
            vals = vals[trim:-trim]
        avg_multiple = round(float(np.mean(vals)), 1)

    fv_series = {}
    for year, eps in eps_s.items():
        if eps and eps > 0:
            fv_series[year] = round(float(eps) * avg_multiple, 2)

    current_eps   = info.get("trailingEps") or (float(eps_s.iloc[-1]) if not eps_s.empty else None)
    current_price = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)
    cur_multiple  = round(current_price / float(current_eps), 1) if current_eps and current_eps > 0 else info.get("trailingPE")

    # Juste valeur courante = dernier point de la série (cohérence garantie avec le graphique)
    last_year_fv = fv_series.get(max(fv_series.keys())) if fv_series else None
    cur_fv = last_year_fv

    return {"metric":"pe","metric_label":"P/E","avg_multiple":avg_multiple,
            "current_multiple":cur_multiple,"current_fv":cur_fv,
            "fair_value_series":fv_series,
            "current_eps":round(float(current_eps),2) if current_eps else None,
            "multiple_series":{str(k):v for k,v in pe_series.items()}}


def compute_evebitda_series(ticker_obj, annual_price: pd.Series, info: dict):
    try:
        fin = ticker_obj.financials
        if "EBITDA" in fin.index:
            ebitda_s = fin.loc["EBITDA"]
        elif "EBIT" in fin.index and "Reconciled Depreciation" in fin.index:
            ebitda_s = fin.loc["EBIT"] + fin.loc["Reconciled Depreciation"]
        else:
            return None
        ebitda_s.index = pd.to_datetime(ebitda_s.index).year
        ebitda_s = ebitda_s.sort_index()
    except Exception:
        return None

    shares   = info.get("sharesOutstanding", 1)
    net_debt = (info.get("totalDebt") or 0) - (info.get("totalCash") or 0)

    ev_series = {}
    for year in annual_price.index:
        if year in ebitda_s.index and ebitda_s[year] and ebitda_s[year] > 0:
            ev = float(annual_price[year]) * shares + net_debt
            ratio = ev / float(ebitda_s[year])
            if 2 <= ratio <= 60:  # filtre les ratios aberrants
                ev_series[year] = round(ratio, 1)

    if len(ev_series) < 3:
        cur_ev = info.get("enterpriseToEbitda")
        if not cur_ev or not (2 <= cur_ev <= 60):
            return None
        avg_multiple = round(float(cur_ev), 1)
    else:
        vals = sorted(ev_series.values())
        if len(vals) >= 6:
            trim = max(1, len(vals) // 10)
            vals = vals[trim:-trim]
        avg_multiple = round(float(np.mean(vals)), 1)

    fv_series = {}
    for year, ebitda in ebitda_s.items():
        if ebitda and ebitda > 0:
            fv_eq = (float(ebitda) * avg_multiple - net_debt) / shares
            if fv_eq > 0:
                fv_series[year] = round(fv_eq, 2)

    cur_ebitda   = info.get("ebitda")
    cur_ev_eb    = info.get("enterpriseToEbitda")

    # Juste valeur courante = dernier point de la série (cohérence avec le graphique)
    last_year_fv = fv_series.get(max(fv_series.keys())) if fv_series else None
    cur_fv = last_year_fv

    return {"metric":"evebitda","metric_label":"EV/EBITDA","avg_multiple":avg_multiple,
            "current_multiple":round(float(cur_ev_eb),1) if cur_ev_eb else None,
            "current_fv":cur_fv,"fair_value_series":fv_series,"current_eps":None,
            "multiple_series":{str(k):v for k,v in ev_series.items()}}


def build_chart_data(fv_series: dict, annual_price: pd.Series) -> dict:
    common = sorted(set(annual_price.index) & set(fv_series.keys()))
    if not common:
        raise ValueError("Impossible d'aligner les données de prix et de juste valeur.")
    labels = [str(y) for y in common]
    prices = [round(float(annual_price[y]), 2) for y in common]
    fvs    = [fv_series[y] for y in common]
    return {"years":labels,"prices":prices,"fair_values":fvs,
            "upper_band":[round(v*1.20,2) for v in fvs],
            "lower_band":[round(v*0.85,2) for v in fvs]}


def compute_fcf_series(ticker_obj, annual_price: pd.Series, info: dict):
    """Price/FCF historique : cours / (Free Cash Flow par action)."""
    try:
        cf = ticker_obj.cashflow
        # FCF = Operating Cash Flow - CapEx
        if "Operating Cash Flow" in cf.index and "Capital Expenditure" in cf.index:
            fcf_s = cf.loc["Operating Cash Flow"] + cf.loc["Capital Expenditure"]  # CapEx est négatif
        elif "Free Cash Flow" in cf.index:
            fcf_s = cf.loc["Free Cash Flow"]
        else:
            return None
        fcf_s.index = pd.to_datetime(fcf_s.index).year
        fcf_s = fcf_s.sort_index()
    except Exception:
        return None

    shares = info.get("sharesOutstanding", 1)
    fcf_per_share = fcf_s / shares

    pfcf_series = {}
    for year in annual_price.index:
        if year in fcf_per_share.index and fcf_per_share[year] and fcf_per_share[year] > 0:
            pfcf_series[year] = round(float(annual_price[year]) / float(fcf_per_share[year]), 1)

    if len(pfcf_series) < 3:
        return None
    avg_multiple = round(float(np.mean(list(pfcf_series.values()))), 1)

    fv_series = {}
    for year, fcf_ps in fcf_per_share.items():
        if fcf_ps and fcf_ps > 0:
            fv_series[year] = round(float(fcf_ps) * avg_multiple, 2)

    current_price  = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)
    current_fcf    = info.get("freeCashflow")
    cur_fcf_ps     = float(current_fcf) / shares if current_fcf else None
    cur_multiple   = round(current_price / cur_fcf_ps, 1) if cur_fcf_ps and cur_fcf_ps > 0 else None

    # Juste valeur courante = dernier point de la série
    last_year_fv = fv_series.get(max(fv_series.keys())) if fv_series else None
    cur_fv = last_year_fv

    return {"metric":"fcf","metric_label":"Price/FCF","avg_multiple":avg_multiple,
            "current_multiple":cur_multiple,"current_fv":cur_fv,
            "fair_value_series":fv_series,"current_eps":None,
            "multiple_series":{str(k):v for k,v in pfcf_series.items()}}


def compute_dividend_data(ticker_obj, annual_price: pd.Series, info: dict) -> dict:
    """
    Retourne les dividendes annuels par action et le rendement historique.
    Utilisé pour l'onglet Dividende ET pour les barres sur le graphique principal.
    """
    try:
        divs = ticker_obj.dividends
        if divs.empty:
            return None
        divs.index = pd.to_datetime(divs.index)
        # Convertit en timezone-naive si nécessaire
        if divs.index.tz is not None:
            divs.index = divs.index.tz_localize(None)
        divs_annual = divs.groupby(divs.index.year).sum()
    except Exception:
        return None

    if divs_annual.empty:
        return None

    # Rendement annuel = dividende / cours fin d'année
    yield_series = {}
    for year in divs_annual.index:
        if year in annual_price.index and annual_price[year] > 0:
            yield_series[year] = round(float(divs_annual[year]) / float(annual_price[year]) * 100, 2)

    current_div_rate = info.get("dividendRate") or info.get("trailingAnnualDividendRate") or 0
    current_yield    = info.get("dividendYield") or info.get("trailingAnnualDividendYield") or 0
    payout_ratio     = info.get("payoutRatio")

    # Série pour barres graphique : {année: dividende_par_action}
    div_bars = {str(int(y)): round(float(v), 4) for y, v in divs_annual.items()
                if y in annual_price.index}

    return {
        "has_dividend":      True,
        "current_div_rate":  round(float(current_div_rate), 4) if current_div_rate else 0,
        "current_yield_pct": round(float(current_yield) * 100, 2) if current_yield else 0,
        "payout_ratio":      round(float(payout_ratio) * 100, 1) if payout_ratio else None,
        "div_bars":          div_bars,          # pour barres sur graphique principal
        "yield_series":      {str(int(k)): v for k, v in yield_series.items()},
        "div_annual":        {str(int(y)): round(float(v), 4) for y, v in divs_annual.items()},
    }


def compute_stock_data(ticker_obj, years: int) -> dict:
    info          = ticker_obj.info
    profile       = detect_profile(info)
    annual_price  = get_annual_prices(ticker_obj, years)
    current_price = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)
    prev_close    = float(info.get("previousClose") or current_price)

    if not current_price:
        raise ValueError("Prix indisponible.")

    pe_data       = compute_pe_series(ticker_obj, annual_price, info)
    evebitda_data = compute_evebitda_series(ticker_obj, annual_price, info)
    fcf_data      = compute_fcf_series(ticker_obj, annual_price, info)
    div_data      = compute_dividend_data(ticker_obj, annual_price, info)

    # Sélection de la métrique principale selon le profil
    if profile["metric"] == "evebitda" and evebitda_data:
        primary = evebitda_data
    elif pe_data:
        primary = pe_data
    elif evebitda_data:
        primary = evebitda_data
    else:
        raise ValueError("Données insuffisantes pour calculer la juste valeur.")

    chart       = build_chart_data(primary["fair_value_series"], annual_price)
    current_fv  = primary["current_fv"]
    premium_pct = round((current_price / float(current_fv) - 1) * 100, 1) if current_fv else None
    day_change  = round((current_price / prev_close - 1) * 100, 2) if prev_close else 0

    return {
        "ticker":           info.get("symbol",""),
        "name":             clean_name(info.get("longName") or info.get("shortName","")),
        "sector":           info.get("sector","—"),
        "exchange":         info.get("exchange",""),
        "currency":         info.get("currency","EUR"),
        "market_cap":       info.get("marketCap"),
        "current_price":    round(current_price, 2),
        "day_change_pct":   day_change,
        # Métrique principale
        "primary_metric":   primary["metric"],
        "metric_label":     primary["metric_label"],
        "avg_multiple":     primary["avg_multiple"],
        "current_multiple": primary["current_multiple"],
        "fair_value":       current_fv,
        "premium_pct":      premium_pct,
        # P/E toujours affiché en secondaire
        "current_pe":       pe_data["current_multiple"] if pe_data else info.get("trailingPE"),
        "avg_pe_10y":       pe_data["avg_multiple"] if pe_data else None,
        "current_eps":      pe_data["current_eps"] if pe_data else None,
        # Chart (métrique principale)
        **chart,
        # Données brutes des métriques pour les onglets
        "pe_data":       pe_data,
        "evebitda_data": evebitda_data,
        "fcf_data":      fcf_data,
        "div_data":      div_data,
    }


# ─── ROUTES ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "ValeurJuste API v2 — OK"}


@app.get("/api/stock/{ticker}")
def get_stock(ticker: str, years: int = 10):
    ticker = ticker.upper().strip()
    cache_key = f"{ticker}:{years}"

    cached = cache_get(cache_key)
    if cached:
        cached["_cached"] = True
        return cached

    try:
        t    = yf.Ticker(ticker)
        info = t.info
        if not info or (info.get("regularMarketPrice") is None and info.get("currentPrice") is None):
            raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' introuvable sur Yahoo Finance.")

        profile = detect_profile(info)
        data    = compute_stock_data(t, years)
        result  = {"status":"ok","profile":profile,"data":data,"_cached":False}
        cache_set(cache_key, result)
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/search/{query}")
def search_tickers(query: str):
    try:
        results = yf.Search(query, max_results=8)
        quotes  = results.quotes if hasattr(results, "quotes") else []
        return {
            "status": "ok",
            "results": [
                {"ticker":q.get("symbol",""),"name":q.get("longname") or q.get("shortname",""),
                 "exchange":q.get("exchange",""),"type":q.get("quoteType","")}
                for q in quotes if q.get("quoteType") in ("EQUITY","ETF")
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/top")
def get_top_valuations(market: str = "all", limit: int = 10):
    """Top décotes et primes en temps réel. market = all | CAC 40 | DAX | S&P 100"""
    if market == "all":
        tickers = [t for ts in TOP_UNIVERSE.values() for t in ts]
        mkt_map = {t: m for m, ts in TOP_UNIVERSE.items() for t in ts}
    elif market in TOP_UNIVERSE:
        tickers = TOP_UNIVERSE[market]
        mkt_map = {t: market for t in tickers}
    else:
        raise HTTPException(status_code=400, detail=f"Marché inconnu : {market}")

    results, errors = [], []

    for ticker in tickers:
        cache_key = f"{ticker}:5"
        cached = cache_get(cache_key)
        if cached and cached.get("data"):
            d = cached["data"]
        else:
            try:
                t    = yf.Ticker(ticker)
                info = t.info
                if not info or (info.get("currentPrice") is None and info.get("regularMarketPrice") is None):
                    continue
                profile = detect_profile(info)
                data    = compute_stock_data(t, years=5)
                entry   = {"status":"ok","profile":profile,"data":data,"_cached":False}
                cache_set(cache_key, entry)
                d = data
            except Exception as ex:
                errors.append({"ticker":ticker,"error":str(ex)})
                continue

        if d.get("premium_pct") is not None:
            results.append({
                "ticker":           ticker,
                "name":             d["name"],
                "market":           mkt_map.get(ticker,""),
                "current_price":    d["current_price"],
                "currency":         d["currency"],
                "current_multiple": d["current_multiple"],
                "avg_multiple":     d["avg_multiple"],
                "metric_label":     d["metric_label"],
                "fair_value":       d["fair_value"],
                "premium_pct":      d["premium_pct"],
            })

    results.sort(key=lambda x: x["premium_pct"])
    return {
        "status":      "ok",
        "market":      market,
        "undervalued": results[:limit],
        "overvalued":  list(reversed(results))[:limit],
        "errors":      errors[:5],
    }


@app.delete("/api/cache")
def clear_cache():
    _cache.clear()
    return {"status":"ok","message":"Cache vidé."}
