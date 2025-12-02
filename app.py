# app.py — Prévia de pedidos (Streamlit) sem statsmodels
# Modos:
#   1) Mesmo dia da semana (média últimas N semanas)
#   2) Semana ISO (modelos simples)
#   3) Modelo ML (LightGBM) — em breve (esqueleto)

import os
import numpy as np
import pandas as pd
from datetime import date, timedelta
import streamlit as st

# ==========================
# Página
# ==========================
st.set_page_config(page_title="Prévia de Pedidos — Previsão", page_icon="📦", layout="wide")
st.title("📦 Prévia de Pedidos — Previsão Semanal com Cap de Variação (sem statsmodels)")
st.caption("Modos: (1) Mesmo dia da semana (média últimas N semanas) • (2) Semana ISO (modelos simples) • (3) Modelo ML (LightGBM) — em breve")

# ==========================
# Helpers genéricos
# ==========================
def is_intermittent(series, zero_ratio_threshold=0.30):
    s = pd.Series(series).fillna(0)
    return (s == 0).mean() >= zero_ratio_threshold

def rolling_pi(residuals, alpha=0.20):
    resid = pd.Series(residuals).dropna().values
    if len(resid) == 0:
        return 0.0, 0.0
    lo = np.quantile(resid, alpha/2)
    hi = np.quantile(resid, 1 - alpha/2)
    return float(lo), float(hi)

def cap_by_band(yhat, ref_value, band=0.20, hard_floor=0):
    lo = max(ref_value * (1 - band), hard_floor)
    hi = ref_value * (1 + band)
    return float(np.clip(yhat, lo, hi)), float(lo), float(hi)

def croston_fallback(y, alpha=0.1):
    y = pd.Series(y).fillna(0).astype(float)
    pos = np.where(y.values > 0)[0]
    if len(pos) == 0:
        return 0.0
    z = y[y > 0]
    idx = np.where(y > 0)[0]
    gaps = np.diff(np.r_[idx, len(y)])
    q = float(z.iloc[0])
    a = float(gaps[0] if len(gaps) > 0 else 1)
    for demand, gap in zip(z.iloc[1:], gaps[1:] if len(gaps) > 1 else [1]*(len(z)-1)):
        q = q + alpha * (float(demand) - q)
        a = a + alpha * (float(gap) - a)
    return float(q / max(a, 1e-9))

def holt_linear(y, alpha=0.3, beta=0.1):
    y = pd.Series(y).astype(float)
    if len(y) == 0:
        return pd.Series(dtype=float), 0.0
    l = np.zeros(len(y)); b = np.zeros(len(y))
    l[0] = y.iloc[0]; b[0] = (y.iloc[1]-y.iloc[0]) if len(y) >= 2 else 0.0
    for t in range(1, len(y)):
        l[t] = alpha*y.iloc[t] + (1-alpha)*(l[t-1] + b[t-1])
        b[t] = beta*(l[t]-l[t-1]) + (1-beta)*b[t-1]
    fitted = pd.Series(l + b, index=y.index)
    return fitted, float(max(l[-1] + b[-1], 0))

def monday_of_iso_week(d: date) -> pd.Timestamp:
    return pd.Timestamp(d).to_period('W-MON').start_time

DOW_LABELS = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom']

# --------- formatação pt-BR ---------
def fmt_int_br(v):
    if pd.isna(v): return ""
    return f"{int(round(float(v))):,}".replace(",", ".")

def fmt_float_br(v, casas=2):
    if pd.isna(v): return ""
    s = f"{float(v):,.{casas}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")

def aplicar_fmt_ptbr(df, cols_int=(), cols_float=()):
    df2 = df.copy()
    for c in cols_int:
        if c in df2.columns:
            df2[c] = df2[c].apply(fmt_int_br)
    for c in cols_float:
        if c in df2.columns:
            df2[c] = df2[c].apply(lambda x: fmt_float_br(x, 2))
    return df2

# ==========================
# Forecast — Modo 1 (Mesmo dia da semana)
# ==========================
def prever_mesmo_dia_semana(df_model, data_alvo, janelas=4, banda_relativa=0.20, previas_ref=None):
    df = df_model.copy()
    df['data'] = pd.to_datetime(df['data'])
    df['dow'] = df['data'].dt.weekday  # 0..6

    alvo = pd.to_datetime(data_alvo)
    dow_alvo = alvo.weekday()

    linhas = []
    for prod, g in df.groupby('produto'):
        gg = g[g['dow'] == dow_alvo].sort_values('data')
        hist = gg[gg['data'] < alvo].tail(janelas)['qtd']  # últimas N ocorrências desse DOW

        if len(hist) == 0:
            yhat = 0.0
            sd = 0.0
        else:
            yhat = float(hist.mean())
            sd = float(hist.std(ddof=1)) if len(hist) > 1 else 0.0

        pi_lo = max(yhat - sd, 0.0)
        pi_hi = yhat + sd

        if previas_ref and prod in previas_ref:
            yhat_cap, cap_lo, cap_hi = cap_by_band(yhat, float(previas_ref[prod]), band=banda_relativa, hard_floor=0)
        else:
            base = gg[gg['data'] < alvo].tail(12)['qtd']
            if len(base) >= 2:
                mu, s = float(base.mean()), float(base.std(ddof=1))
                cap_lo = max(mu - 1.5*s, 0.0)
                cap_hi = mu + 1.5*s
            else:
                cap_lo, cap_hi = 0.0, max(yhat*2, yhat+1)
            yhat_cap = float(np.clip(yhat, cap_lo, cap_hi))

        linhas.append({
            'produto': prod,
            'data_alvo': alvo.date(),
            'weekday': dow_alvo,
            'forecast': round(yhat, 2),
            'pi_lo': round(pi_lo, 2),
            'pi_hi': round(pi_hi, 2),
            'forecast_capado': round(yhat_cap, 2),
            'cap_lo': round(cap_lo, 2),
            'cap_hi': round(cap_hi, 2),
            'modelo': f'Media{janelas}x_{DOW_LABELS[dow_alvo]}'
        })

    return pd.DataFrame(linhas).sort_values('produto')

# ==========================
# Forecast — Modo 2 (Semana ISO)
# ==========================
def prever_semana_iso(df_model, semana_alvo_ts, previas_ref=None, banda_relativa=0.20,
                      min_hist=10, alpha_pi=0.20):
    df = df_model.copy()
    df['data'] = pd.to_datetime(df['data'])
    df = (df.assign(semana=df['data'].dt.to_period('W-MON').apply(lambda p: p.start_time))
            .groupby(['produto','semana'], as_index=False)['qtd'].sum()
            .sort_values(['produto','semana']))

    semana_alvo = pd.to_datetime(semana_alvo_ts).to_period('W-MON').start_time
    linhas = []

    for prod, g in df.groupby('produto'):
        s = g.set_index('semana')['qtd'].asfreq('W-MON').fillna(0)
        hist = s[s.index < semana_alvo]
        if len(hist) == 0:
            continue
        try:
            if is_intermittent(hist):
                yhat = croston_fallback(hist.values)
                resid = hist - hist.rolling(4, min_periods=1).mean()
                modelo = 'Croston_fallback'
            elif len(hist) >= min_hist:
                fitted, yhat = holt_linear(hist, alpha=0.3, beta=0.1)
                resid = (hist - fitted).fillna(0)
                modelo = 'Holt_linear'
            else:
                yhat = float(hist.tail(4).mean())
                resid = hist - hist.rolling(4, min_periods=1).mean()
                modelo = 'MA_4w'
        except Exception:
            yhat = float(hist.tail(4).mean())
            resid = hist - hist.rolling(4, min_periods=1).mean()
            modelo = 'MA_4w_fallback'

        resid = resid.dropna()
        if len(resid) >= 8:
            lo_r, hi_r = rolling_pi(resid, alpha=alpha_pi)
        else:
            sd = resid.std(ddof=1) if resid.std(ddof=1) > 0 else 0
            lo_r, hi_r = -sd, sd
        yhat_lo = max(yhat + lo_r, 0)
        yhat_hi = max(yhat + hi_r, 0)

        if previas_ref and prod in previas_ref:
            yhat_cap, band_lo, band_hi = cap_by_band(yhat, float(previas_ref[prod]), band=banda_relativa, hard_floor=0)
        else:
            tail = hist.tail(12)
            mu, sd = tail.mean(), tail.std(ddof=1) if tail.std(ddof=1) > 0 else 0
            k = 1.5
            band_lo, band_hi = max(mu - k*sd, 0), mu + k*sd
            yhat_cap = float(np.clip(yhat, band_lo, band_hi))

        linhas.append({
            'produto': prod,
            'semana_prever': semana_alvo.date(),
            'forecast': round(yhat, 2),
            'pi_lo': round(yhat_lo, 2),
            'pi_hi': round(yhat_hi, 2),
            'forecast_capado': round(yhat_cap, 2),
            'cap_lo': round(band_lo, 2),
            'cap_hi': round(band_hi, 2),
            'modelo': modelo
        })

    return pd.DataFrame(linhas).sort_values('produto')

# ==========================
# Sidebar — parâmetros
# ==========================
st.sidebar.header("Parâmetros")

# Conexão
driver = st.sidebar.selectbox("ODBC Driver", ["ODBC Driver 18 for SQL Server", "ODBC Driver 17 for SQL Server"])
server = st.sidebar.text_input("SERVER", os.getenv("SQL_SERVER", "SEU_SERVIDOR_SQL"))
database = st.sidebar.text_input("DATABASE", os.getenv("SQL_DATABASE", "SUA_BASE"))
uid = st.sidebar.text_input("USER", os.getenv("SQL_USER", "SEU_USUARIO"))
pwd = st.sidebar.text_input("PASSWORD", type="password", value=os.getenv("SQL_PASSWORD", ""))

# Datas do histórico
default_fim = date.today()
default_ini = default_fim - timedelta(weeks=12)
data_ini = st.sidebar.date_input("Data inicial", default_ini)
data_fim = st.sidebar.date_input("Data final (inclusive)", default_fim)

# Nível
modo_agreg = st.sidebar.radio("Nível da previsão", ["Produto", "Produtor+Produto"], horizontal=True)

# Estratégia
estrategias = [
    "Mesmo dia da semana (média últimas N semanas)",
    "Semana ISO (modelos)",
    "Modelo ML (LightGBM) — em breve"
]
estrategia = st.sidebar.selectbox("Estratégia de previsão", estrategias, index=0)

# Parâmetros de cada estratégia
if estrategia.startswith("Mesmo dia"):
    data_alvo = st.sidebar.date_input("Data-alvo (ex.: 05/11/2025)", default_fim)
    janelas = st.sidebar.slider("N de semanas (mesmo dia) a considerar", 2, 12, 4)
elif estrategia.startswith("Semana ISO"):
    data_alvo = st.sidebar.date_input("Qualquer data da semana-alvo (usaremos a segunda-feira da ISO-week)", default_fim)
    janelas = 4
else:
    data_alvo = default_fim
    janelas = 4

# Cap
banda = st.sidebar.slider("Faixa de variação (cap) ±%", 5, 50, 20, step=5) / 100.0

run = st.sidebar.button("Prever", type="primary")

# ==========================
# SQL corrigidos
# ==========================

# 1) MODO PRODUTO — pega do CORTEPRODUTOS, mas só se houver fornecedor válido
SQL_PRODUTO = """
WITH base AS (
    SELECT
        CAST(E.DATADOPEDIDO AS DATE) AS data_pedido,
        D.CODIGOREFERENCIA           AS codigoreferencia,
        D.NOME                       AS nome_produto,
        C.QUANTIDADEPEDIDO           AS qtd_pedido
    FROM K_CM_CORTEPRODUTOS C
    JOIN PD_PRODUTOS     D ON D.HANDLE = C.PRODUTO
    JOIN K_CM_CORTES     E ON E.HANDLE = C.CORTE
    WHERE CAST(E.DATADOPEDIDO AS DATE) BETWEEN ? AND ?
      AND E.STATUS <> 5
      AND E.TIPODEPEDIDO = '1'
      AND EXISTS (
          SELECT 1
          FROM K_CM_CORTEPRODUTOFORNECEDORES A
          LEFT JOIN GN_PESSOAS B ON B.HANDLE = A.PESSOA
          WHERE A.CORTEPRODUTO = C.HANDLE
            AND (B.CODIGO IS NULL OR B.CODIGO NOT IN (1747,1748,879888))
      )
)
SELECT
    data_pedido,
    codigoreferencia,
    MAX(nome_produto)   AS nome_produto,
    SUM(qtd_pedido)     AS qtd
FROM base
GROUP BY data_pedido, codigoreferencia
ORDER BY data_pedido, codigoreferencia;
"""

# 2) MODO PRODUTOR+PRODUTO — usa o que o produtor realmente produziu/ajustou
SQL_PRODUTOR_PRODUTO = """
WITH base AS (
    SELECT
        CAST(E.DATADOPEDIDO AS DATE) AS data_pedido,
        B.CODIGO                      AS cod_produtor,
        B.NOME                        AS nome_produtor,
        D.CODIGOREFERENCIA            AS codigoreferencia,
        D.NOME                        AS nome_produto,
        ISNULL(A.QUANTIDADEPRODUCAO,0) + ISNULL(A.QUANTIDADEPRODUCAOAJUSTADA,0) AS qtd_fornecedor
    FROM K_CM_CORTEPRODUTOFORNECEDORES A
    JOIN K_CM_CORTEPRODUTOS C ON C.HANDLE = A.CORTEPRODUTO
    JOIN PD_PRODUTOS        D ON D.HANDLE = C.PRODUTO
    JOIN K_CM_CORTES        E ON E.HANDLE  = C.CORTE
    LEFT JOIN GN_PESSOAS    B ON B.HANDLE  = A.PESSOA
    WHERE CAST(E.DATADOPEDIDO AS DATE) BETWEEN ? AND ?
      AND E.STATUS <> 5
      AND E.TIPODEPEDIDO = '1'
      AND (B.CODIGO IS NULL OR B.CODIGO NOT IN (1747,1748,879888))
)
SELECT
    data_pedido,
    cod_produtor,
    nome_produtor,
    codigoreferencia,
    nome_produto,
    SUM(qtd_fornecedor) AS qtd
FROM base
GROUP BY data_pedido, cod_produtor, nome_produtor, codigoreferencia, nome_produto
ORDER BY data_pedido, cod_produtor, codigoreferencia;
"""

# ==========================
# Execução
# ==========================
if run:

    # Se o usuário escolher o ML (esqueleto)
    if estrategia.startswith("Modelo ML"):
        st.warning("🧠 Modelo ML (LightGBM): em breve. Vou incluir features (feriados, janelas, clima), "
                   "backtest rolante e previsão P50/P90 com cap de negócio.")
        st.stop()

    try:
        import pyodbc
        conn_str = (
            f"DRIVER={{{driver}}};SERVER={server};DATABASE={database};UID={uid};PWD={pwd};"
            "TrustServerCertificate=yes;"
        )
        conn_str += "Encrypt=yes;" if "ODBC Driver 18" in driver else "Encrypt=no;"

        with st.spinner("Conectando ao SQL Server e lendo dados..."):
            conn = pyodbc.connect(conn_str, timeout=30)

            if modo_agreg == "Produto":
                df = pd.read_sql(SQL_PRODUTO, conn, params=[data_ini, data_fim])

                df = df[df['codigoreferencia'].notna()].copy()
                df['codigoreferencia'] = df['codigoreferencia'].astype(str).str.strip()

                nome_map = (df.groupby('codigoreferencia', as_index=False)['nome_produto']
                              .agg(lambda s: s.dropna().iloc[-1] if len(s.dropna()) else None)
                              .set_index('codigoreferencia')['nome_produto']
                              .to_dict())

                df_model = (df.rename(columns={'codigoreferencia':'produto','data_pedido':'data'})
                              [['produto','data','qtd']])
                df_model = df_model.groupby(['produto','data'], as_index=False)['qtd'].sum()

            else:
                df = pd.read_sql(SQL_PRODUTOR_PRODUTO, conn, params=[data_ini, data_fim])

                df = df[df['codigoreferencia'].notna()].copy()
                df['codigoreferencia'] = df['codigoreferencia'].astype(str).str.strip()
                df['cod_produtor'] = df['cod_produtor'].astype(str).str.strip()

                df['produto_chave'] = df['cod_produtor'] + '|' + df['codigoreferencia']

                nome_map = df.groupby('produto_chave')['nome_produtor'].agg('last').to_dict()

                df_model = (df.rename(columns={'produto_chave':'produto','data_pedido':'data'})
                              [['produto','data','qtd']])
                df_model = df_model.groupby(['produto','data'], as_index=False)['qtd'].sum()

            conn.close()

        dup = df_model.duplicated(['produto','data']).sum()
        st.caption(f"🔎 Linhas duplicadas (produto+data) após agregação: {dup}")

        previas_ref = {}

        if estrategia.startswith("Mesmo dia"):
            res = prever_mesmo_dia_semana(df_model, data_alvo, janelas=janelas, banda_relativa=banda, previas_ref=previas_ref)
            res['nome'] = res['produto'].map(nome_map)
            st.success("Previsão concluída!")
            st.subheader("Resultado")
            st.write(f"Data-alvo: **{pd.to_datetime(data_alvo).date()}** (dia: **{DOW_LABELS[pd.to_datetime(data_alvo).weekday()]}**) "
                     f"— Janela: **{janelas} semanas** — Cap: **±{int(banda*100)}%**")

            rotulos = {
                'produto': 'Código',
                'nome': 'Produto',
                'data_alvo': 'Data da previsão',
                'weekday': 'Dia da semana',
                'forecast': 'Previsão (bruta)',
                'pi_lo': 'Faixa baixa (confiança)',
                'pi_hi': 'Faixa alta (confiança)',
                'cap_lo': 'Limite mínimo (cap)',
                'cap_hi': 'Limite máximo (cap)',
                'forecast_capado': 'Previsão final (após cap)',
                'modelo': 'Método'
            }
            res_exibir = res.rename(columns=rotulos)

            cols_negocio = [
                'Código','Produto','Data da previsão',
                'Previsão (bruta)','Previsão final (após cap)',
                'Limite mínimo (cap)','Limite máximo (cap)','Método'
            ]
            cols_negocio = [c for c in cols_negocio if c in res_exibir.columns]

            res_fmt = aplicar_fmt_ptbr(
                res_exibir[cols_negocio],
                cols_int=['Previsão (bruta)','Previsão final (após cap)','Limite mínimo (cap)','Limite máximo (cap)']
            )

            st.dataframe(res_fmt, use_container_width=True)

            csv = res_exibir[cols_negocio].to_csv(index=False).encode('utf-8-sig')
            st.download_button("⬇️ Baixar CSV", data=csv, file_name="previsao.csv", mime="text/csv")

            # drill-down
            with st.expander("🔎 Ver histórico das últimas N semanas (mesmo dia) de um produto"):
                codigo_escolhido = st.text_input("Código do produto (ex.: 1, 1000354, ...)", "")
                if codigo_escolhido:
                    alvo = pd.to_datetime(data_alvo)
                    dow = alvo.weekday()
                    dfh = df_model[df_model['produto'].astype(str) == str(codigo_escolhido)].copy()
                    if len(dfh):
                        dfh['data'] = pd.to_datetime(dfh['data'])
                        dfh['dow'] = dfh['data'].dt.weekday
                        mesmas = dfh[(dfh['dow']==dow) & (dfh['data'] < alvo)].sort_values('data').tail(janelas)
                        if len(mesmas):
                            aux = mesmas[['data','qtd']].sort_values('data', ascending=False).copy()
                            aux['Quantidade'] = aux['qtd'].apply(fmt_int_br)
                            aux = aux.rename(columns={'data':'Data'})[['Data','Quantidade']]
                            st.write(f"Dia alvo: **{alvo.date()} ({DOW_LABELS[dow]})**")
                            st.write("Ocorrências usadas na média (total do dia, somando todos os produtores válidos):")
                            st.dataframe(aux, use_container_width=True)
                            st.write(f"**Média calculada (previsão bruta):** {fmt_int_br(mesmas['qtd'].mean())}")
                        else:
                            st.info("Sem histórico suficiente para esse dia da semana.")
                    else:
                        st.info("Produto não encontrado no histórico do período filtrado.")

        elif estrategia.startswith("Semana ISO"):
            semana_alvo_ts = monday_of_iso_week(data_alvo)
            res = prever_semana_iso(df_model, semana_alvo_ts, previas_ref=previas_ref, banda_relativa=banda)
            res['nome'] = res['produto'].map(nome_map)
            st.success("Previsão concluída!")
            st.subheader("Resultado")
            st.write(f"Semana-alvo (início ISO-week): **{semana_alvo_ts.date()}** — Cap: **±{int(banda*100)}%**")

            rotulos = {
                'produto': 'Código',
                'nome': 'Produto',
                'semana_prever': 'Início da semana (ISO)',
                'forecast': 'Previsão (bruta)',
                'pi_lo': 'Faixa baixa (confiança)',
                'pi_hi': 'Faixa alta (confiança)',
                'cap_lo': 'Limite mínimo (cap)',
                'cap_hi': 'Limite máximo (cap)',
                'forecast_capado': 'Previsão final (após cap)',
                'modelo': 'Método'
            }
            res_exibir = res.rename(columns=rotulos)

            cols_negocio = [
                'Código','Produto','Início da semana (ISO)',
                'Previsão (bruta)','Previsão final (após cap)',
                'Limite mínimo (cap)','Limite máximo (cap)','Método'
            ]
            cols_negocio = [c for c in cols_negocio if c in res_exibir.columns]

            res_fmt = aplicar_fmt_ptbr(
                res_exibir[cols_negocio],
                cols_int=['Previsão (bruta)','Previsão final (após cap)','Limite mínimo (cap)','Limite máximo (cap)']
            )

            st.dataframe(res_fmt, use_container_width=True)

            csv = res_exibir[cols_negocio].to_csv(index=False).encode('utf-8-sig')
            st.download_button("⬇️ Baixar CSV", data=csv, file_name="previsao.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Erro durante a execução: {e}")
        st.stop()
else:
    st.info("Ajuste os parâmetros e clique em **Prever**.")