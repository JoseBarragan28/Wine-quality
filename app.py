import io,  time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import fitz  
from mistralai import Mistral
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import KFold, cross_val_score


st.set_page_config(page_title="Wine Advisor Bot", page_icon="🍷", layout="wide")

# This keeps chat input fixed at bottom while using tabs
st.markdown("""
<style>
div[data-testid="stChatInput"]{
  position: fixed; bottom: 1rem; left: 1.5rem; right: 1.5rem; z-index: 1000;
}
main .block-container{ padding-bottom: 7rem; }
</style>
""", unsafe_allow_html=True)

st.title("Wine Advisor Bot 🍷")
st.write("Chatbot analítico + modelos predictivos para la toma de decisiones basadas en datos.")

FEATURES = [
    "fixed acidity","volatile acidity","citric acid","residual sugar","chlorides",
    "free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"
]
STRICT_COLUMNS = FEATURES + ["type"]   


#Using models for each type of wine
APP_DIR   = Path(__file__).parent.resolve()
MODEL_DIR = APP_DIR / "models"
MODEL_RED_PATH  = MODEL_DIR / "randomf_final_red_removed.pkl"
MODEL_WHITE_PATH = MODEL_DIR / "randomf_final_white_removed.pkl"
MODEL_PATHS = {"red": MODEL_RED_PATH, "white": MODEL_WHITE_PATH}

api_key = st.secrets["MISTRAL_API_KEY"]
MISTRAL_MODEL = "mistral-small-2506"

# This helps to load models only once
@st.cache_resource(show_spinner=False)
def load_model_for(kind: str):
    path = MODEL_PATHS[kind]
    if not path.exists():
        raise FileNotFoundError(
            f"El modelo no se encontró: {path}\n"
            "Coloca tus modelos en la carpeta /models/ "

        )
    return joblib.load(path)

# ───────────────────────── Session initial state ──────────────────────────
# chat history (tab 1)
if "history" not in st.session_state:            
    st.session_state.history = []

# table(tab 2)    
if "grid_df" not in st.session_state:            
    st.session_state.grid_df = None

 # {pred, rounded, time, kind}
if "grid_result" not in st.session_state:
    st.session_state.grid_result = None         

# results (tab 3)
if "batch_pred_df" not in st.session_state:      
    st.session_state.batch_pred_df = None
if "batch_total_time" not in st.session_state:
    st.session_state.batch_total_time = 0.0


tab1, tab2, tab3, tab4 = st.tabs([" 🤖 aClaraBot - Chatbot especializado ", "🍷 Modelo predictivo (un ejemplar)", "🍷🍷🍷 Modelo predictivo (varios ejemplares)", "📊 Análisis de calidad"])


# ============================ TAB 1 =============================== 
with tab1:
    
    uploaded_docs = st.file_uploader("",type=["pdf", "txt"], accept_multiple_files=True, key="docs_uploader_tab1")

    # Read pdf by extracting text page by page
    file_text = None
    if uploaded_docs:
        doc_texts = []
        for f in uploaded_docs:
            name = f.name.lower()
            if name.endswith(".pdf"):
                with fitz.open(stream=f.read(), filetype="pdf") as doc:
                    doc_texts.append("\n".join(page.get_text() for page in doc))
            else:
                doc_texts.append(f.read().decode("utf-8", errors="ignore"))
        file_text = "\n\n---\n\n".join(doc_texts)
    else:
        st.info("Si lo deseas, puedes subir un documento .pdf o .txt el cual quieras comprender.")


    # Render previous chat
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    
    prompt = st.chat_input("Pregunta lo que quieras…")
    if prompt:

        # system message including uploaded doc text
        if file_text and not any(m["role"] == "system" for m in st.session_state.history):
            st.session_state.history.insert(0, {
                "role": "system",
                "content": f'''
                    Tu objetivo principal es proporcionar al gerente de "Vinho Verde & Co." información clara, concisa y práctica sobre todo aquello relacionado a la industria del vino, estrategias de mercado y operaciones de producción. Esto es para finalmente, facilitar la toma de decisiones rápidas y efectivas, ayudando al gerente a tomar decisiones para mejorar la calidad del vino, optimizar los precios y aumentar las ventas. Debes adaptarte a las siguientes directrices:
                       -Evita tecnicismos y explica cada concepto de manera sencilla, como si hablaras con alguien sin conocimientos previos.
                        -Destaca los puntos más relevantes y sus beneficios directos para el negocio. No incluyas detalles técnicos innecesarios. 
                        -Ofrece sugerencias claras y directas para mejorar la calidad, precios y ventas. 
                        -La explicación debe ser clara, breve y orientada a resultados inmediatos. Sin embargo, esto no implica que se usen términos sin explicar, de manera que cada concepto debe explicarse de forma sencilla, sin asumir que el usuario tiene conocimientos previos. 
                        -Asegúrate de que todas las respuestas sean relevantes y útiles para la toma de decisiones estratégicas y operativas.
                         Ahora bien, el gerente ha subido un documento y necesita que considerando las anteriores restricciones le des una explicación clara de lo que te vaya a preguntar. 
                         A continuacion el documento proporcionado:\n\n{file_text}'''
                
            })

        # Append user message
        st.session_state.history.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)

        client = Mistral(api_key=api_key)
        try:
            resp = client.chat.complete(model=MISTRAL_MODEL, messages=st.session_state.history)
            reply = resp.choices[0].message.content
        except Exception as e:
            reply = f"⚠️ Mistral error: {e}"

        st.session_state.history.append({"role": "assistant", "content": reply})
        st.chat_message("assistant").markdown(reply)

# ============================ TAB 2 ===============================
with tab2:

    # Creating a readable table based on the wine variables
    if st.session_state.grid_df is None:
        features_ext = FEATURES + ["type of wine"]
        rows = []
        for i in range(0, len(features_ext), 2):
            a = features_ext[i]
            b = features_ext[i+1] if i+1 < len(features_ext) else ""
            rows.append({"X": a, "valor X": "", "Y": b, "valor Y": ""})

        # Default values of table as an example
        defaults = {
            "fixed acidity": 7.4, "volatile acidity": 0.70, "citric acid": 0.00,
            "residual sugar": 1.90, "chlorides": 0.076, "free sulfur dioxide": 11.0,
            "total sulfur dioxide": 34.0, "density": 0.9978, "pH": 3.51,
            "sulphates": 0.56, "alcohol": 9.4, "type of wine": "red"
        }
        grid = pd.DataFrame(rows)

        #filling table with the values and variables
        for idx, r in grid.iterrows():
            a, b = r["X"], r["Y"]
            if a in defaults: grid.at[idx, "valor X"] = str(defaults[a])
            if b in defaults: grid.at[idx, "valor Y"] = str(defaults[b])
        st.session_state.grid_df = grid

    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        # Editable table for wine features
        edited = st.data_editor(
            st.session_state.grid_df, use_container_width=True, hide_index=True, key="single_grid_editor",
            column_config={
                "X": st.column_config.Column("X", disabled=True, help="Feature name"),
                "valor X": st.column_config.TextColumn("valor X"),
                "Y": st.column_config.Column("Y", disabled=True, help="Feature name"),
                "valor Y": st.column_config.TextColumn("valor Y"),
            }
        )
        # Button to calculate quality
        if st.button("Calcular calidad ", type="primary", key="predict_grid_btn"):
            try:
                # Collect values from the table
                vals, wine_kind_val = {}, None
                for _, r in edited.iterrows():
                    a, va = r["X"], str(r["valor X"]).strip()
                    b, vb = r["Y"], str(r["valor Y"]).strip()
                    if a:
                        if a == "type of wine":
                            wine_kind_val = va.lower()
                        else:
                            if va == "" or va.lower() == "nan":
                                raise ValueError(f"Falta el valor de '{a}'.")
                            vals[a] = float(va)
                    if b:
                        if b == "type of wine":
                            wine_kind_val = vb.lower()
                        else:
                            if vb == "" or vb.lower() == "nan":
                                raise ValueError(f"Falta el valor de '{b}'.")
                            vals[b] = float(vb)

                #Just in case we have several options
                kind = {"red": "red", "tinto": "red", "rojo": "red", "white": "white", "blanco": "white"}.get(wine_kind_val)
                if kind is None:
                    raise ValueError("Por favor, establece el tipo de vino como 'tinto' o 'blanco'.")

                #If the table is not filled
                missing = [f for f in FEATURES if f not in vals]
                if missing:
                    raise ValueError("Faltan valores para: " + ", ".join(missing))

                X_one = pd.DataFrame([{f: vals[f] for f in FEATURES}])
                
                #time of execution calculation
                model = load_model_for(kind)
                t0 = time.time()
                yhat = model.predict(X_one)
                dur = time.time() - t0
                pred = float(yhat[0]); pred_round = int(np.rint(pred))
                
                #save results to show later
                st.session_state.grid_result = {"pred": pred, "rounded": pred_round, "time": dur, "kind": kind}

                # Now use the editable table created before
                st.session_state.grid_df = edited
                st.success("Predicción completa.")
                st.rerun()

            except Exception as e:
                st.error(f"Error al predecir: {e}")

    with col_right:
        st.subheader("Resultados")
        r = st.session_state.grid_result
        if r is None:
            st.info('''Si lo deseas, ingresa las características de un vino en la tabla y pulsa **Calcular calidad** para predecir su calidad.   
                    Recuerda seguir la convención para unidades y variables usada en Cortez et al. (2009).''')
        else:
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Calidad redondeada", r["rounded"])
            with c2: st.metric("Calidad exacta", f"{r['pred']:.4f}")
            with c3: st.metric("Tiempo (s)", f"{r['time']:.4f}")
        

# ============================ TAB 3 ===============================
with tab3: 
    uploaded_batch = st.file_uploader(" ",type=["csv", "xlsx", "xls"], accept_multiple_files=True, key="batch_uploader_tab3")

    if uploaded_batch:
        # Read several files and concatenate into a DataFrame
        frames = []
        for f in uploaded_batch:
            name = f.name.lower()
            if name.endswith(".csv"):
                try:
                    df = pd.read_csv(f, sep=None, engine="python", encoding="utf-8-sig")
                except Exception:
                    f.seek(0)
                    try:
                        df = pd.read_csv(f, sep=";", encoding="utf-8-sig")
                    except Exception:
                        f.seek(0)
                        df = pd.read_csv(f, encoding="utf-8-sig")
            else:
                df = pd.read_excel(f)

            df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
            frames.append(df)

        df_input = pd.concat(frames, ignore_index=True)

        # Check of convention
        if set(df_input.columns.astype(str)) != set(STRICT_COLUMNS):
            st.error(
                "❌ Por favor recuerde seguir la convención dada.\n\n"
                "**Columnas esperadas (exactamente):**\n"
                f"{', '.join(STRICT_COLUMNS)}\n\n"
                "**Tus columnas:**\n"
                f"{', '.join(df_input.columns.astype(str))}"
            )
            st.stop()

        # order + validate type column
        df_input = df_input[STRICT_COLUMNS]
        types = df_input["type"].astype(str).str.strip().str.lower()
        ok_types = types.isin(["red", "white"])
        if not ok_types.all():
            bad = df_input.index[~ok_types].tolist()
            st.error(
                "❌ La columna **type** debe ser 'red' o 'white' (sin distinción mayúsc./minúsc.).\n"
                f"Filas inválidas: {bad[:50]}" + (" (y más…)" if len(bad) > 50 else "")
            )
            st.stop()

        # Predict per type 
        df_pred = df_input.copy()
        df_pred["pred_quality"] = np.nan
        df_pred["pred_quality_rounded"] = np.nan
        total_time = 0.0

        # Red
        mask_red = types == "red"
        if mask_red.any():
            X_red = df_pred.loc[mask_red, FEATURES]
            model_red = load_model_for("red")

            # prediction time to show later
            t0 = time.time(); y_red = model_red.predict(X_red); dt = time.time() - t0; total_time += dt
            df_pred.loc[mask_red, "pred_quality"] = y_red
            df_pred.loc[mask_red, "pred_quality_rounded"] = np.rint(y_red).astype(int)

            # Predicted values
            y_round = np.rint(y_red).astype(int)
            counts = pd.Series(y_round).value_counts().sort_index()
            avg_red = float(np.mean(y_red))
            

        # White
        mask_white = types == "white"
        if mask_white.any():
            X_white = df_pred.loc[mask_white, FEATURES]
            model_white = load_model_for("white")

            # prediction time to show later
            t1 = time.time(); y_white = model_white.predict(X_white); dt = time.time() - t1; total_time += dt
            df_pred.loc[mask_white, "pred_quality"] = y_white
            df_pred.loc[mask_white, "pred_quality_rounded"] = np.rint(y_white).astype(int)

            # Predicted values
            y_round = np.rint(y_white).astype(int)
            counts = pd.Series(y_round).value_counts().sort_index()
            avg_white = float(np.mean(y_white))


        st.session_state.batch_pred_df = df_pred
        st.session_state.batch_total_time = total_time
    else:
        st.info("Si lo deseas, puedes subir un documento CSV o Excel con la información "
              "de un conjuntos de vinos para predecir la calidad del grupo. Incluye exactamente estas columnas "
              "(en cualquier orden): " + ", ".join(STRICT_COLUMNS))     

    # Results preview + download
    if st.session_state.batch_pred_df is not None:
        
        col_left, col_right = st.columns([3, 2], gap="large")

        with col_left:
         st.dataframe(st.session_state.batch_pred_df, use_container_width=True, hide_index=True)
        with col_right: 
         st.subheader("Resultados") 
         c1, c2 = st.columns(2)
         with c1: st.metric(" # Vinos blancos", f"{mask_white.sum()}")
         with c2: st.metric("Calidad media vino blanco", f"{avg_white:.4f}")
         
         c3, c4 = st.columns(2)
         with c3: st.metric(" # Vinos tintos", f"{mask_red.sum()}")
         with c4: st.metric("Calidad media vino tinto", f"{avg_red:.4f}")

         st.metric("Tiempo de ejecución (s)", f"{total_time:.4f}")

         out = io.StringIO()
         st.session_state.batch_pred_df.to_csv(out, index=False)
         st.download_button("📥 Descargar predicciones (csv)",
                           data=out.getvalue(), file_name="predictions.csv", mime="text/csv")


# ============================ TAB 4 ===============================
with tab4:
    FIGSIZE_HIST      = (5, 3)   # quality
    FIGSIZE_BARH      = (5.5, 3 + 0.25*len(FEATURES))  #  MI
    FIGSIZE_PDP_W, FIGSIZE_PDP_H = 4, 2.8  # partial dependences

    uploaded_eda = st.file_uploader("", type=["csv"], accept_multiple_files=False, key="eda_uploader")

    if uploaded_eda:
        # read CSV with ; and ,clean not useful parts
        try:
            df_eda = pd.read_csv(uploaded_eda, sep=None, engine="python", encoding="utf-8-sig")
        except Exception:
            uploaded_eda.seek(0)
            try:
                df_eda = pd.read_csv(uploaded_eda, sep=";", encoding="utf-8-sig")
            except Exception:
                uploaded_eda.seek(0)
                df_eda = pd.read_csv(uploaded_eda, encoding="utf-8-sig")
        df_eda.columns = [str(c).replace("\ufeff", "").strip() for c in df_eda.columns]

        # check STRICT_COLUMNS + quality
        required_cols = STRICT_COLUMNS + ["quality"]
        if set(df_eda.columns.astype(str)) != set(required_cols):
            st.error(
                "❌ Columnas esperadas (exactamente):\n"
                f"{', '.join(required_cols)}\n\n"
                "Tus columnas:\n"
                f"{', '.join(df_eda.columns.astype(str))}"
            )
            st.stop()

        # just to have consisten dtypes
        df_eda["type"] = df_eda["type"].astype(str).str.strip().str.lower()
        df_eda["quality"] = pd.to_numeric(df_eda["quality"], errors="coerce")

        # ===================== Fila 1: Resumen (izquierda) vs Distribución (derecha) =====================
        col_left_top, col_right_top = st.columns([1.1, 1.2], gap="large")

        with col_left_top:
            st.subheader("Resultados preliminares")
            # number of wines
            mask_white = df_eda["type"] == "white"
            mask_red   = df_eda["type"] == "red"
            n_white = int(mask_white.sum())
            n_red   = int(mask_red.sum())

            # Mean and std of quality per wine

            q_white = df_eda.loc[mask_white, "quality"].dropna()
            q_red   = df_eda.loc[mask_red, "quality"].dropna()

            mean_white = float(q_white.mean()) if len(q_white) else float("nan")
            std_white  = float(q_white.std(ddof=1)) if len(q_white) > 1 else float("nan")

            mean_red   = float(q_red.mean()) if len(q_red) else float("nan")
            std_red    = float(q_red.std(ddof=1)) if len(q_red) > 1 else float("nan")

            c1, c2 = st.columns(2)
            with c1: st.metric("# Vinos blancos", f"{n_white}")
            with c2: st.metric("# Vinos tintos", f"{n_red}")

            c3, c4 = st.columns(2)
            with c3:
                val = f"{mean_white:.4f} ± {std_white:.4f}" if np.isfinite(mean_white) and np.isfinite(std_white) else "N/A"
                st.metric("Calidad media vino blanco", val)
            with c4:
                val = f"{mean_red:.4f} ± {std_red:.4f}" if np.isfinite(mean_red) and np.isfinite(std_red) else "N/A"
                st.metric("Calidad media vino tinto", val)

        with col_right_top:
            st.subheader("Calidad del conjunto")
            q_all = df_eda["quality"].dropna().values
            fig, ax = plt.subplots(figsize=FIGSIZE_HIST)
            bins = np.arange(0, 12, 1)  # bordes
            ax.hist(q_all, bins=bins, edgecolor="black", align="left")
            ax.set_xticks(np.arange(0, 11, 1))
            ax.set_xlabel("Calidad"); ax.set_ylabel("Frecuencia")
            # values on histogram bars
            counts, _ = np.histogram(q_all, bins=bins)
            for i, cnt in enumerate(counts):
                if cnt > 0:
                    ax.text(i, cnt, str(int(cnt)), ha="center", va="bottom", fontsize=9)
            st.pyplot(fig)

        # ===================== Fila 2: Correlaciones (izquierda) vs MI (derecha) =====================
        col_left_mid, col_right_mid = st.columns([1.1, 1.2], gap="large")

        with col_left_mid:
            st.subheader("Analisis de correlaciones lineales y monótonas")

            corr_method = st.radio("Método", ["Pearson", "Spearman"], horizontal=True, key="eda_corr_method")
            method_key = "pearson" if corr_method == "Pearson" else "spearman"

            
            feat_cols = FEATURES
            corr_series = df_eda[feat_cols + ["quality"]].corr(method=method_key)["quality"].drop("quality")
            k = min(7, len(feat_cols))  # top 5
            pos = corr_series.sort_values(ascending=False).head(k).reset_index()
            pos.columns = ["feature", "corr"]
            neg = corr_series.sort_values(ascending=True).head(k).reset_index()
            neg.columns = ["feature", "corr"]

            c1, c2 = st.columns(2)
            with c1:
                st.write(f"**{corr_method} – correlación positiva**")
                st.dataframe(pos, use_container_width=True, hide_index=True)
            with c2:
                st.write(f"**{corr_method} – correlación negativa**")
                st.dataframe(neg, use_container_width=True, hide_index=True)

        with col_right_mid:
            st.subheader("Correlaciones no lineales")
            

            X = df_eda[FEATURES].copy()
            y = df_eda["quality"].values
            mask_ok = ~X.isna().any(axis=1) & ~pd.isna(y)
            Xn = X.loc[mask_ok]; yn = y[mask_ok]

            mi = mutual_info_regression(Xn, yn, random_state=0)
            mi_df = pd.DataFrame({"feature": FEATURES, "Información mutua": mi}).sort_values("Información mutua", ascending=False)

            fig2, ax2 = plt.subplots(figsize=FIGSIZE_BARH)
            ax2.barh(mi_df["feature"], mi_df["Información mutua"])
            ax2.invert_yaxis(); ax2.set_xlabel("Información mutua")
            for i, val in enumerate(mi_df["Información mutua"]):
                ax2.text(val, i, f"{val:.3f}", va="center", ha="left", fontsize=9)
            st.pyplot(fig2)

        # ===================== Fila 3: Dependencia parcial (gráficos pequeños) =====================
        st.subheader("Dependencias parciales respecto a la calidad ")


        Xp = df_eda[FEATURES].copy()
        yp = df_eda["quality"].astype(float).values
        mask_p = ~Xp.isna().any(axis=1) & ~pd.isna(yp)
        Xp = Xp.loc[mask_p]; yp = yp[mask_p]

        rf = RandomForestRegressor(n_estimators=200, random_state=111, n_jobs=-1)
        rf.fit(Xp, yp)
        do_cv = st.checkbox("Calcular métricas ", value=False, key="eda_cv_metrics")
        if do_cv:
        
            cv = KFold(n_splits=5, shuffle=True, random_state=111)

            mse_scores = -cross_val_score(rf, Xp, yp, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1)
            mae_scores = -cross_val_score(rf, Xp, yp, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)
            r2_scores  =  cross_val_score(rf, Xp, yp, cv=cv, scoring="r2", n_jobs=-1)

            rmse_scores = np.sqrt(mse_scores)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("CV RMSE", f"{rmse_scores.mean():.4f}")
            with c2:
                st.metric("CV MAE", f"{mae_scores.mean():.4f}")
            with c3:
                st.metric("CV R²", f"{r2_scores.mean():.4f}")


        sel = st.multiselect("Elija hasta 3 variables para graficar", FEATURES, default=FEATURES[:3], max_selections=3)
        if len(sel) > 0:
            # manual subplot
            
            n = len(sel)
            fig3, axes = plt.subplots(1, n, figsize=(FIGSIZE_PDP_W*n+4, FIGSIZE_PDP_H))
            if n == 1:
                axes = [axes]  # uniform handling
            for ax_i, feat in zip(axes, sel):
                disp = PartialDependenceDisplay.from_estimator(rf, Xp, features=[feat], ax=ax_i)
                ax_i.set_title(feat, fontsize=10)
            st.pyplot(fig3)
    else:
        st.info("Si lo deseas, puedes subir un documento CSV o Excel con la información "
              "de un conjuntos de vinos para analizar la calidad del grupo. Incluye exactamente estas columnas "
              "(en cualquier orden): " + ", ".join(STRICT_COLUMNS + ["quality"]))
