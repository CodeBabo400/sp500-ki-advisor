import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
import numpy as np

# --- 1. KONFIGURATION & STYLING ---
st.set_page_config(
    page_title='S&P 500 KI-Berater',
    page_icon='📈',
    layout='wide'
)

# Custom CSS für schöneres Design und größere Schrift bei Erklärungen
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    h1 { color: #2c3e50; }
    .stMetric { background-color: white; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

# --- 2. FUNKTIONEN ---

@st.cache_data
def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Pfad ggf. anpassen auf 'data' oder 'clean_data'
    file_path = os.path.join(script_dir, 'clean_data', 'SP500_Cleaned.csv')
    
    df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    
    # Feature Engineering (Intern bleiben die Namen englisch für den Code)
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['Dist_MA20'] = df['Close'] / df['MA20'] - 1
    df['Volatility'] = df['Close'].pct_change().rolling(window=10).std()
    
    days_ahead = 5
    df['Future_Return'] = df['Close'].shift(-days_ahead).pct_change(periods=days_ahead) * 100
    df['Target'] = (df['Future_Return'] > 1.0).astype(int)
    
    return df.dropna()

@st.cache_resource
def train_model(df):
    features = ['Open', 'High', 'Low', 'Close', 'Change_Percent', 'MA20', 'MA50', 'Dist_MA20', 'Volatility']
    X = df[features]
    y = df['Target']
    
    split = int(len(df) * 0.7)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    
    return model, acc, prec, features

# --- 3. DATEN LADEN ---
try:
    df = load_data()
    model, accuracy, precision, feature_names = train_model(df)
except Exception as e:
    st.error(f"Fehler beim Laden: {e}")
    st.stop()

# --- 4. SIDEBAR ---
st.sidebar.title('Einstellungen ⚙️')

min_date = df.index.min().date()
max_date = df.index.max().date()

start_date, end_date = st.sidebar.date_input(
    "Zeitraum wählen:",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    help="Wähle den Zeitraum, der in den Grafiken angezeigt werden soll."
)

filtered_df = df.loc[str(start_date):str(end_date)]

# --- 5. HAUPTBEREICH ---

st.title('📈 S&P 500 KI-Investment-Berater')

# Metriken mit verständlichen Namen und Hilfetexten
col1, col2 = st.columns(2)
col1.metric(
    "KI-Genauigkeit (Allgemein)", 
    f"{accuracy:.2%}", 
    help="Wie oft liegt die KI insgesamt richtig (egal ob sie Kaufen oder Warten empfiehlt)?"
)
col2.metric(
    "Trefferquote bei Kaufempfehlung", 
    f"{precision:.2%}", 
    delta_color="normal", 
    help="Die wichtigste Zahl: Wenn die KI sagt 'Kauf jetzt!', wie oft macht man dann wirklich Gewinn?"
)

st.markdown("---")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Markt-Check", "🔍 Detail-Analyse", "🤖 KI-Simulator"])

# --- TAB 1: ÜBERSICHT ---
with tab1:
    st.subheader("Wie steht der Markt gerade?")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Aktueller Preis", f"{filtered_df['Close'].iloc[-1]:.2f} $", help="Preis für einen Anteil am S&P 500")
    c2.metric("Nervosität (Volatilität)", f"{filtered_df['Volatility'].mean():.2%}", help="Durchschnittliche Schwankung pro Tag. Hoch = Riskant, Niedrig = Ruhig.")
    c3.metric("Gute Kauf-Chancen", f"{filtered_df['Target'].sum()}", help="Anzahl der Tage im Zeitraum, an denen ein Kauf profitabel gewesen wäre.")
    c4.metric("Gewinn (Buy & Hold)", f"{(filtered_df['Close'].iloc[-1] - filtered_df['Close'].iloc[0]):.2f} $", help="Gewinn, wenn man einfach am ersten Tag gekauft und gehalten hätte.")

    # Chart mit verständlichen Legenden
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df['Close'], name='Tatsächlicher Preis', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df['MA20'], name='Kurzfrist-Trend (20 Tage)', line=dict(color='blue', width=1, dash='dot')))
    fig.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df['MA50'], name='Langfrist-Trend (50 Tage)', line=dict(color='orange', width=1, dash='dot')))
    fig.update_layout(title="Preisentwicklung und Trends", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    
# --- TAB 2: ANALYSE ---
with tab2:
    st.subheader("Wann gewinnt man an der Börse?")
    st.write("Wir haben die Vergangenheit analysiert: Was unterscheidet Gewinner-Tage von Verlierer-Tagen?")
    
    col1, col2 = st.columns(2)
    
    # Daten für Plot vorbereiten (Text statt 0/1 für die Legende)
    df_plot = filtered_df.copy()
    df_plot['Erfolg'] = df_plot['Target'].map({0: 'Kein Erfolg', 1: 'Erfolg (>1% Gewinn)'})
    
    with col1:
        st.markdown("#### 1. Wie viel 'Action' brauchen wir?")
        
        # FIX: Verständliche Achsen-Beschriftungen mittels 'labels'
        fig_vio = px.violin(df_plot, 
                            y="Volatility", 
                            x="Erfolg", 
                            color="Erfolg", 
                            box=True, 
                            title="Vergleich: Markt-Ruhe vs. Markt-Unruhe",
                            # HIER PASSIERT DIE MAGIE:
                            labels={
                                'Volatility': 'Tägliche Schwankung (Je höher, desto wilder)',
                                'Erfolg': 'Ergebnis nach 5 Tagen',
                                'count': 'Anzahl der Tage'
                            },
                            color_discrete_map={'Kein Erfolg': '#EF553B', 'Erfolg (>1% Gewinn)': '#00CC96'} 
                           )
        st.plotly_chart(fig_vio, use_container_width=True)
        
    with col2:
        st.markdown("#### 2. Ist der Preis gerade günstig?")
        
        fig_hist = px.histogram(df_plot, 
                                x="Dist_MA20", 
                                color="Erfolg", 
                                title="Analyse: Lohnt sich der Einstieg bei hohen oder tiefen Preisen?",
                                barmode="overlay",
                                # HIER SIND DIE NEUEN, KRISTALLKLAREN LABELS:
                                labels={
                                    'Dist_MA20': 'Preis-Lage (Links = überverkauft | Rechts = überkauft)',
                                    'count': 'Anzahl Tage',
                                    'Erfolg': 'Ergebnis des Kaufs'
                                },
                                color_discrete_map={'Kein Erfolg': '#EF553B', 'Erfolg (>1% Gewinn)': '#00CC96'}
                               )
        
        # Die Nulllinie bleibt als Orientierung wichtig
        fig_hist.add_vline(x=0, line_width=2, line_dash="dash", line_color="black", 
                           annotation_text="Durchschnittspreis", 
                           annotation_position="top right")
        
        # Update für das Layout, damit die Achsen-Titel auch sicher ganz angezeigt werden
        fig_hist.update_layout(yaxis_title="Anzahl der Tage")
        
        st.plotly_chart(fig_hist, use_container_width=True)
    with st.expander("ℹ️ Lesehilfe zu den Grafiken"):
        st.write("""
        **Linke Grafik (Nervosität):**
        * **Y-Achse:** Zeigt an, wie stark die Kurse an diesen Tagen geschwankt haben.
        * **Erkenntnis:** Schau dir die **grüne Form** an. Liegt ihr "dicker Bauch" höher als bei der roten? Das heißt: Um schnell Gewinne zu machen, muss der Markt sich bewegen. Wenn die Kurse stillstehen, gewinnt man selten.

        **Rechte Grafik (Preis-Niveau):**
        * **X-Achse:** * **Links von der Linie (Minus-Bereich):** Der Preis war *günstiger* als der Durchschnitt ("Schnäppchen").
            * **Rechts von der Linie (Plus-Bereich):** Der Preis war *teurer* als der Durchschnitt ("Hype").
        * **Erkenntnis:** Wo siehst du mehr Grün? Oft lohnt es sich einzusteigen, wenn der Preis kurz unter die Nulllinie getaucht ist.
        """)
        
        
# --- TAB 3: MACHINE LEARNING ---
with tab3:
    st.header("🔮 KI-Simulator")
    st.write("Stell dir vor, du stehst heute vor einer Kaufentscheidung. Gib die Markt-Daten ein, und die KI sagt dir ihre Meinung.")
    
    with st.form("prediction_form"):
        st.subheader("Markt-Daten eingeben")
        
        c1, c2 = st.columns(2)
        last_row = df.iloc[-1]
        
        with c1:
            input_close = st.number_input("Aktueller Preis ($)", value=float(last_row['Close']), help="Der heutige Schlusskurs.")
            input_ma20 = st.number_input("Kurzfristiger Trend (MA20)", value=float(last_row['MA20']), help="Der Durchschnittspreis der letzten 20 Tage.")
            input_open = st.number_input("Eröffnungspreis ($)", value=float(last_row['Open']))
        
        with c2:
            input_ma50 = st.number_input("Langfristiger Trend (MA50)", value=float(last_row['MA50']))
            input_vol = st.number_input("Aktuelle Nervosität (Volatilität)", value=float(last_row['Volatility']), format="%.4f", help="Ein Wert wie 0.01 bedeutet 1% Schwankung.")
            
            # Berechneter Wert (nur zur Info)
            calc_dist = input_close / input_ma20 - 1
            st.caption(f"💡 Der Preis liegt aktuell **{calc_dist:.2%}** im Vergleich zum kurzfristigen Trend.")

        submit = st.form_submit_button("🚀 Soll ich kaufen?")
    
    if submit:
        # Daten für Modell vorbereiten
        input_data = pd.DataFrame({
            'Open': [input_open],
            'High': [input_open * 1.01], 
            'Low': [input_open * 0.99], 
            'Close': [input_close],
            'Change_Percent': [0.0], 
            'MA20': [input_ma20],
            'MA50': [input_ma50],
            'Dist_MA20': [calc_dist],
            'Volatility': [input_vol]
        })
        
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]
        
        st.divider()
        if prediction == 1:
            st.success(f"### ✅ KI-Empfehlung: KAUFEN")
            st.write(f"Die KI ist sich zu **{proba[1]:.0%}** sicher, dass der Kurs in 5 Tagen steigen wird.")
        else:
            st.error(f"### 🛑 KI-Empfehlung: ABWARTEN")
            st.write(f"Die Gewinnchance wird nur auf **{proba[1]:.0%}** geschätzt. Das Risiko ist zu hoch.")
            
    # Feature Importance (vereinfacht)
    st.divider()
    with st.expander("Warum entscheidet die KI so? (Experten-Ansicht)"):
        imp_df = pd.DataFrame({'Faktor': feature_names, 'Wichtigkeit': model.feature_importances_}).sort_values('Wichtigkeit', ascending=True)
        # Namen eindeutschen für die Grafik
        name_mapping = {
            'Volatility': 'Nervosität', 'Dist_MA20': 'Abstand zum Trend', 
            'MA20': 'Kurzfrist-Trend', 'MA50': 'Langfrist-Trend', 
            'Close': 'Preis', 'Change_Percent': 'Änderung %'
        }
        imp_df['Faktor'] = imp_df['Faktor'].map(name_mapping).fillna(imp_df['Faktor'])
        
        fig_imp = px.bar(imp_df, x='Wichtigkeit', y='Faktor', orientation='h', title="Einflussfaktoren auf die Entscheidung")
        st.plotly_chart(fig_imp)