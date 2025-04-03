import streamlit as st
import pandas as pd
import requests
from io import StringIO
import plotly.express as px
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
from folium.plugins import FloatImage
import pydeck as pdk
from sklearn.linear_model import LinearRegression
import numpy as np


# === PAGINA CONFIGURATIE ===
st.set_page_config(page_title="Schiphol Vluchtdata Dashboard", layout="wide")

# === SIDEBAR NAVIGATIE ===
st.sidebar.title("Navigatie")
pagina = st.sidebar.radio("Ga naar:", [
    "Home",
    "Vluchten per Continent",
    "Drukte per Uur",
    "Pier Analyse",
    "Bestemmingen",
    "Routekaart vanuit Schiphol",
    "Schiphol-kaart (live piers)"
])

# === SIDEBAR: DATUMSELECTIE ===
vandaag = datetime.today().date()
geselecteerde_datum = st.sidebar.date_input(
    "Kies een datum", 
    value=vandaag,
    min_value=vandaag - timedelta(days=7),
    max_value=vandaag + timedelta(days=2)
)

richting = st.sidebar.radio("Vlucht richting", options=["Aankomst", "Vertrek"])
richting_code = "A" if richting == "Aankomst" else "D"

# === SIDEBAR: BRONNEN ===
st.sidebar.markdown("---")
st.sidebar.markdown("### Bronnen")
st.sidebar.markdown("""
- [Schiphol Open API](https://developer.schiphol.nl/)
- [Wikipedia Bestemmingen](https://nl.wikipedia.org/wiki/Lijst_van_luchtvaartbestemmingen)
- [GeoNames.org](https://www.geonames.org/)
- Interne GeoCoördinaten dictionary
""")


# === SCHIPHOL API CONFIGURATIE ===
app_id = "271dfe19"
app_key = "4b73f597bf2b67cec2630572385e354a"
url = "https://api.schiphol.nl/public-flights/flights"
headers = {
    "Accept": "application/json",
    "app_id": app_id,
    "app_key": app_key,
    "ResourceVersion": "v4"
}

# === FUNCTIE: VLUCHTDATA OPHALEN ===
@st.cache_data(ttl=600)
def laad_data(datum, richting):
    all_flights = []
    max_pages = 50
    for page in range(max_pages):
        params = {
            "flightDirection": richting,
            "scheduleDate": datum.strftime("%Y-%m-%d"),
            "page": page
        }
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            flights = response.json().get("flights", [])
            if not flights:
                break
            all_flights.extend(flights)
        else:
            st.error(f"Fout bij ophalen pagina {page}: status {response.status_code}")
            break

    if all_flights:
        return pd.json_normalize(all_flights)
    else:
        return pd.DataFrame()

# === DATA OPHALEN ===
df = laad_data(geselecteerde_datum, richting_code)

# === FEEDBACK & EERSTE WEERGAVE ===
if df.empty:
    st.warning(f"Geen vluchtdata gevonden voor {geselecteerde_datum} ({richting.lower()})")
    st.stop()
else:
    st.success(f"{len(df)} vluchten gevonden op {geselecteerde_datum} ({richting.lower()})")

# === ACTUELE KOL TOMEN CHECKS ===
kolomlijst = ['flightName', 'scheduleDate', 'scheduleTime', 'terminal', 'gate']
zichtbare_kolommen = [k for k in kolomlijst if k in df.columns]

# === FUNCTIE: CONTROLEER EN VUL MISSENDE KOLOMMEN ===
def check_and_fill_missing_columns(dataframe, expected_columns, fill_value=None):
    for column in expected_columns:
        if column not in dataframe.columns:
            dataframe[column] = fill_value
            # st.warning(f"Kolom '{column}' ontbreekt in de data en is gevuld met {fill_value}.")
    return dataframe


# List of expected columns
expected_columns = ['actualLandingTime', 'estimatedLandingTime', 'scheduleTime']

# Check and fill missing columns
df = check_and_fill_missing_columns(df, expected_columns, fill_value=None)

# === CONVERSIES VOOR AANKOMST ===
if richting_code == "A":
    df['actualLandingTime'] = pd.to_datetime(df['actualLandingTime'], errors='coerce')
    df['estimatedLandingTime'] = pd.to_datetime(df['estimatedLandingTime'], errors='coerce')
    df['landingDelay'] = (df['actualLandingTime'] - df['estimatedLandingTime']).dt.total_seconds()
else:
    df['landingDelay'] = None

# === DATUMTIJD EN DELAY CALCULATIES ===
df['actualLandingTime'] = pd.to_datetime(df['actualLandingTime'], errors='coerce')
df['estimatedLandingTime'] = pd.to_datetime(df['estimatedLandingTime'], errors='coerce')
df['landingDelay'] = (df['actualLandingTime'] - df['estimatedLandingTime']).dt.total_seconds()

# === VERRIJK MET IATA-INFORMATIE VIA WIKIPEDIA ===
wikipedia_url = "https://nl.wikipedia.org/wiki/Vliegvelden_gesorteerd_naar_IATA-code"
wiki_headers = {'User-Agent': 'Mozilla/5.0'}
html = requests.get(wikipedia_url, headers=wiki_headers).text
tables = pd.read_html(StringIO(html))
vliegvelden = tables[1].drop_duplicates(subset='IATA', keep='last')
mapping = vliegvelden.set_index('IATA')['Luchthaven']
mapping2 = vliegvelden.set_index('IATA')['Stad']
mapping3 = vliegvelden.set_index('IATA')['Land']

df['route.destinations'] = df['route.destinations'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
df['Luchthaven'] = df['route.destinations'].map(mapping)
df['Stad'] = df['route.destinations'].map(mapping2)
df['Land'] = df['route.destinations'].map(mapping3)

# === IN-CODE GEOCOORDINATEN DICTIONARY (hernoemd) ===
geo_coords = {
    "AMS": {"latitude": 52.3105, "longitude": 4.7683},
    "JFK": {"latitude": 40.6413, "longitude": -73.7781},
    "LHR": {"latitude": 51.4700, "longitude": -0.4543},
    "CDG": {"latitude": 49.0097, "longitude": 2.5479},
    "FRA": {"latitude": 50.0379, "longitude": 8.5622},
    "DXB": {"latitude": 25.2532, "longitude": 55.3657},
    "SIN": {"latitude": 1.3644, "longitude": 103.9915},
    "DEL": {"latitude": 28.5562, "longitude": 77.1000},
    "HKG": {"latitude": 22.3080, "longitude": 113.9185},
    "ICN": {"latitude": 37.4602, "longitude": 126.4407},
    "MAD": {"latitude": 40.4983, "longitude": -3.5676},
    "BCN": {"latitude": 41.2974, "longitude": 2.0833},
    "IST": {"latitude": 41.2753, "longitude": 28.7519},
    "YYZ": {"latitude": 43.6777, "longitude": -79.6248},
    "SFO": {"latitude": 37.6213, "longitude": -122.3790},
    "ORD": {"latitude": 41.9742, "longitude": -87.9073}
    # Voeg hier zelf nog meer codes toe indien nodig
}

def add_coordinates(row):
    code = row['route.destinations']
    if code in geo_coords:
        return pd.Series(geo_coords[code])
    return pd.Series({"latitude": None, "longitude": None})

df[['latitude', 'longitude']] = df.apply(add_coordinates, axis=1)
df = df.dropna(subset=['longitude'])

# === DATUMTIJD EN DELAY CALCULATIES ===
df['actualLandingTime'] = pd.to_datetime(df['actualLandingTime'], errors='coerce')
df['estimatedLandingTime'] = pd.to_datetime(df['estimatedLandingTime'], errors='coerce')
df['landingDelay'] = (df['actualLandingTime'] - df['estimatedLandingTime']).dt.total_seconds()

# === CONTINENTENMAPPING (IN-CODE) ===
continenten = {
    "Europa": ["Nederland", "Frankrijk", "Duitsland", "Spanje", "Verenigd Koninkrijk", "Turkije"],
    "Azië": ["India", "Singapore", "China", "Zuid-Korea", "Japan", "Verenigde Arabische Emiraten"],
    "Noord-Amerika": ["Verenigde Staten", "Canada"],
    "Afrika": ["Egypte", "Zuid-Afrika", "Marokko"],
    "Zuid-Amerika": ["Brazilië"],
    "Oceanië": ["Australië", "Nieuw-Zeeland"]
}

# === NEDERLANDS → ENGELS LANDNAMEN VOOR KAART ===
nederlandse_naar_engelse_landen = {
    "Duitsland": "Germany",
    "Verenigd Koninkrijk": "United Kingdom",
    "Frankrijk": "France",
    "Spanje": "Spain",
    "Italië": "Italy",
    "Verenigde Staten": "United States of America",
    "Canada": "Canada",
    "Nederland": "Netherlands",
    "België": "Belgium",
    "Zwitserland": "Switzerland",
    "Zweden": "Sweden",
    "Noorwegen": "Norway",
    "Denemarken": "Denmark",
    "China": "China",
    "Japan": "Japan",
    "Australië": "Australia",
    "Turkije": "Turkey",
    "Griekenland": "Greece",
    "Portugal": "Portugal",
    "Polen": "Poland",
    "Tsjechië": "Czech Republic"
    # Voeg meer vertalingen toe indien nodig
}

# === GeoJSON laden ===
geojson_url = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json"
response = requests.get(geojson_url)
landen_geojson = response.json()

def land_naar_continent(land):
    for cont, landen in continenten.items():
        if land in landen:
            return cont
    return "Onbekend"

df['Continent'] = df['Land'].apply(land_naar_continent)

# === FUNCTIE: KLEURTOEWIJZING VOOR DE KAART (volgens aantal vluchten) ===
def get_color(count):
    if count >= 75:
        return "darkred"      # zeer druk
    elif count > 50:
        return "red"          # druk
    elif count > 25:
        return "orange"       # matig
    elif count > 10:
        return "yellowgreen"  # minder druk
    else:
        return "green"        # rustig

# === PAGINA'S ===

# --- HOMEPAGINA ---
if pagina == "Home":
    st.title("✈️ Schiphol Vluchtdata Dashboard")
    st.write("Gebruik het menu links om analyses te bekijken over vluchten, vertragingen en geografische verdeling vanuit Schiphol.")
    st.dataframe(df[['flightName', 'Luchthaven', 'Land', 'Continent', 'landingDelay']].head())
    st.dataframe(df[zichtbare_kolommen].head(15))


# --- VLUCHTEN PER CONTINENT ---
elif pagina == "Vluchten per Continent":
    st.header("🌍 Vluchten per Continent")
    geselecteerd = st.sidebar.selectbox("Kies continent:", ["Alle"] + sorted(df['Continent'].unique()))
    subset = df if geselecteerd == "Alle" else df[df['Continent'] == geselecteerd]
    fig = px.histogram(subset, x='Land', title=f"Aantal vluchten uit {geselecteerd}",
                       color='Land', color_discrete_sequence=px.colors.qualitative.Set1)
    st.plotly_chart(fig)

# --- PIE CHART PER PIER ---
elif pagina == "Pier Analyse":
    st.header("Vluchten per Pier (Vertrekkende vluchten)")

    # Data ophalen alleen voor vertrek
    df_vertrek = laad_data(geselecteerde_datum, richting="D")
    df_vertrek = check_and_fill_missing_columns(df_vertrek, ['pier', 'route.destinations'], fill_value=None)

    # Verrijk met landinformatie
    df_vertrek['route.destinations'] = df_vertrek['route.destinations'].apply(
        lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None
    )
    df_vertrek['Land'] = df_vertrek['route.destinations'].map(mapping3)

    # Landfilter toevoegen
    beschikbare_landen = sorted(df_vertrek['Land'].dropna().unique())
    gekozen_land = st.selectbox("Filter op bestemming (land):", options=["Alle"] + beschikbare_landen)

    # 🔍 Filter toepassen indien nodig
    if gekozen_land != "Alle":
        df_vertrek = df_vertrek[df_vertrek['Land'] == gekozen_land]

    # Analyse als er nog data is
    if 'pier' in df_vertrek.columns and not df_vertrek.empty:
        df_pier = df_vertrek['pier'].value_counts().reset_index()
        df_pier.columns = ['Pier', 'Aantal Vluchten']

        fig = px.pie(
            df_pier,
            names='Pier',
            values='Aantal Vluchten',
            title=f"Verdeling Vertrekkende Vluchten per Pier ({gekozen_land})",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig)
    else:
        st.info("Geen pier-gegevens beschikbaar voor deze selectie.")


# --- BESTEMMINGEN ANALYSE ---
elif pagina == "Bestemmingen":
    st.header("📍 Top Bestemmingen")
    df_dest = df['Luchthaven'].value_counts().reset_index()
    df_dest.columns = ['Bestemming', 'Aantal Vluchten']
    # Kleurenschaal: hogere aantallen (druk) worden rood en lagere aantallen (rustig) groen
    fig = px.bar(df_dest, x='Bestemming', y='Aantal Vluchten', color='Aantal Vluchten',
                 title="Top Bestemmingen", color_continuous_scale="RdYlGn_r")
    st.plotly_chart(fig)

# --- DRUKTE PER UUR ---
elif pagina == "Drukte per Uur":
    st.header("Drukte-analyse per Uur")

    # Filter alleen vluchten met geldige scheduleTime
    df_uur = df.dropna(subset=['scheduleTime']).copy()

    # Zet tijd om naar datetime (met seconden-formaat!)
    df_uur['Uur'] = pd.to_datetime(df_uur['scheduleTime'], format="%H:%M:%S", errors='coerce').dt.hour

    # Groepeer per uur en tel aantal vluchten
    vluchten_per_uur = df_uur['Uur'].value_counts().sort_index()
    uren_labels = [f"{uur:02d}:00" for uur in vluchten_per_uur.index]

    # Plot: aantal vluchten per uur
    fig = px.bar(
        x=uren_labels,
        y=vluchten_per_uur.values,
        labels={'x': 'Uur van de dag', 'y': 'Aantal vluchten'},
        title=f"Vluchten per uur op {geselecteerde_datum}",
        color=vluchten_per_uur.values,
        color_continuous_scale='YlOrRd'
    )
    st.plotly_chart(fig)

    # === Statistische analyse: lineaire regressie + correlatie ===
    from sklearn.linear_model import LinearRegression
    from scipy.stats import pearsonr
    import numpy as np

    # Bereid data voor regressie
    uur_data = pd.DataFrame({
        'uur': vluchten_per_uur.index,
        'aantal_vluchten': vluchten_per_uur.values
    })
    X = uur_data[['uur']]
    y = uur_data['aantal_vluchten']

    # Regressiemodel trainen
    model = LinearRegression()
    model.fit(X, y)
    voorspelling = model.predict(X)

    # Correlatiecoëfficiënt en R²
    r, p = pearsonr(X['uur'], y)
    r_squared = model.score(X, y)

    # Toon resultaten
    st.markdown(f"**Correlatiecoëfficiënt (r)** tussen uur en aantal vluchten: `{r:.2f}`")
    st.markdown(f"**R² van het regressiemodel**: `{r_squared:.2f}`")

    if abs(r) > 0.7:
        st.info("Sterke correlatie: het aantal vluchten is duidelijk afhankelijk van het uur van de dag.")
    elif abs(r) > 0.4:
        st.info("Matige correlatie: er lijkt een patroon te zijn, maar het is niet perfect.")
    else:
        st.info("Zwakke correlatie: het uur van de dag heeft weinig voorspellende waarde.")

    # Optioneel: plot regressielijn bovenop originele data
    import plotly.graph_objects as go
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=X['uur'], y=y, mode='markers', name='Aantal vluchten'))
    fig2.add_trace(go.Scatter(x=X['uur'], y=voorspelling, mode='lines', name='Regressielijn'))
    fig2.update_layout(
        title="Lineaire regressie: aantal vluchten per uur",
        xaxis_title="Uur van de dag",
        yaxis_title="Aantal vluchten"
    )
    st.plotly_chart(fig2)


# --- KAART MET GROEIENDE CIRKELS PER GATE ---
elif pagina == "Schiphol-kaart (live piers)":
    st.title("Schiphol Piers op echte kaart")

    # Alleen vertrekdata gebruiken
    df_vertrek = laad_data(geselecteerde_datum, richting="D")
    df_vertrek = check_and_fill_missing_columns(df_vertrek, ['pier'], fill_value=None)
    df_pier = df_vertrek[df_vertrek["pier"].notna()]
    pier_counts = df_pier["pier"].value_counts().reset_index()
    pier_counts.columns = ["pier", "aantal"]

    # 🔁 Betere coördinaten per pier (handmatig verbeterd)
    pier_coords = {
        "B": (52.3094, 4.7611),
        "C": (52.3096, 4.7600),
        "D": (52.3105, 4.7595),
        "E": (52.3109, 4.7620),
        "F": (52.3109, 4.7635),
        "G": (52.3113, 4.7650),
        "H": (52.3095, 4.7645),
        "M": (52.3080, 4.7585),
    }

    kleuren_lijst = px.colors.qualitative.Set3
    pier_kleuren = {pier: kleuren_lijst[i % len(kleuren_lijst)] for i, pier in enumerate(pier_coords.keys())}

    m = folium.Map(location=[52.31, 4.761], zoom_start=17, tiles="CartoDB positron")

    for pier, coords in pier_coords.items():
        aantal = pier_counts.loc[pier_counts['pier'] == pier, 'aantal'].sum()
        if aantal > 0:
            kleur = pier_kleuren.get(pier, "gray")

            folium.CircleMarker(
                location=coords,
                radius=7 + aantal / 10,
                color=kleur,
                fill=True,
                fill_color=kleur,
                fill_opacity=0.7,
                popup=f"Pier {pier} - {aantal} vluchten",
                tooltip=f"{pier}: {aantal} vluchten"
            ).add_to(m)

            folium.map.Marker(
                coords,
                icon=folium.DivIcon(
                    html=f'''
                        <div style="font-size: 11px; font-weight: bold; color: black;
                                    background-color: rgba(255,255,255,0.85); padding: 3px 6px;
                                    border-radius: 5px; border: 1px solid #888;">
                            Pier {pier}<br>{aantal} vluchten
                        </div>
                    '''
                )
            ).add_to(m)

    # Legenda links: kleuren per pier
    legenda_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; width: 180px; height: auto;
        border:2px solid grey; z-index:9999; font-size:14px; background-color:white;
        padding: 10px;">
        <b>Legenda Piers</b><br>
    '''
    for pier, kleur in pier_kleuren.items():
        legenda_html += f'&nbsp; Pier {pier} <i class="fa fa-circle" style="color:{kleur}"></i><br>'
    legenda_html += '</div>'
    m.get_root().html.add_child(folium.Element(legenda_html))

    # Legenda rechts: cirkelgrootte = aantal vluchten
    intensiteit_legenda_html = '''
    <div style="position: fixed; bottom: 50px; right: 50px; width: 200px; height: auto;
        border:2px solid grey; z-index:9999; font-size:14px; background-color:white;
        padding: 10px;">
        <b>📈 Cirkelgrootte = Drukte</b><br>
        <svg width="160" height="60">
            <circle cx="30" cy="20" r="7" fill="gray" fill-opacity="0.6" stroke="black" stroke-width="0.5"/>
            <text x="50" y="25" font-size="12">~5 vluchten</text>

            <circle cx="30" cy="45" r="15" fill="gray" fill-opacity="0.6" stroke="black" stroke-width="0.5"/>
            <text x="50" y="50" font-size="12">~50 vluchten</text>
        </svg>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(intensiteit_legenda_html))

    st_folium(m, width=1000, height=620)




#=============
# ROUTE
# ======---------
elif pagina == "Routekaart vanuit Schiphol":
    st.header("🌐 Routekaart vanuit Schiphol (ingekleurde landen - alleen vertrekvluchten)")

    # 🚀 Alleen vertrekvluchten ophalen (richting = D)
    df_vertrek = laad_data(geselecteerde_datum, richting="D")
    df_vertrek = df_vertrek.dropna(subset=["route.destinations"])
    df_vertrek['route.destinations'] = df_vertrek['route.destinations'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)

    # Wikipedia-verrijking opnieuw toepassen op vertrekdata
    df_vertrek['Land'] = df_vertrek['route.destinations'].map(mapping3)
    df_vertrek['Land_EN'] = df_vertrek['Land'].map(nederlandse_naar_engelse_landen)

    landen_in_data_en = df_vertrek['Land_EN'].dropna().unique()

    # 🎨 Dynamische kleuren per EN land
    land_kleuren = {
        land: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
        for i, land in enumerate(sorted(landen_in_data_en))
    }

    # 🌍 Sidebar-filter (NL labels tonen, intern EN gebruiken)
    land_opties_nl = [nl for nl in df_vertrek['Land'].dropna().unique() if nl in nederlandse_naar_engelse_landen]
    geselecteerd_land_nl = st.sidebar.selectbox("Filter op land (inkleuring):", options=["Alle"] + sorted(land_opties_nl))
    geselecteerd_land_en = nederlandse_naar_engelse_landen.get(geselecteerd_land_nl) if geselecteerd_land_nl != "Alle" else "Alle"

    # 🌐 GeoJSON landenkaart ophalen
    geojson_url = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json"
    response = requests.get(geojson_url)
    landen_geojson = response.json()

    # Voeg kleuren toe aan GeoJSON
    for feature in landen_geojson["features"]:
        land_naam_en = feature["properties"]["name"]
        if land_naam_en in land_kleuren:
            feature["properties"]["kleur"] = px.colors.hex_to_rgb(land_kleuren[land_naam_en])
        else:
            feature["properties"]["kleur"] = [220, 220, 220]

    # Layer 1: alle landen
    geojson_layer = pdk.Layer(
        "GeoJsonLayer",
        data=landen_geojson,
        get_fill_color="properties.kleur",
        get_line_color=[80, 80, 80],
        opacity=0.5,
        stroked=True,
        pickable=True,
        auto_highlight=True
    )

    layers = [geojson_layer]

    # Layer 2: geselecteerd land highlighten
    if geselecteerd_land_en != "Alle":
        geselecteerd_feature = [f for f in landen_geojson["features"]
                                if f["properties"]["name"] == geselecteerd_land_en]
        if geselecteerd_feature:
            single_layer = pdk.Layer(
                "GeoJsonLayer",
                data={"type": "FeatureCollection", "features": geselecteerd_feature},
                get_fill_color=px.colors.hex_to_rgb(land_kleuren[geselecteerd_land_en]),
                opacity=0.8,
                stroked=True,
                get_line_color=[10, 10, 10],
            )
            layers.append(single_layer)

    # View instellen
    view_state = pdk.ViewState(latitude=30, longitude=10, zoom=1.5)

    # Pydeck-kaart weergeven
    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=view_state,
        layers=layers,
        tooltip={"text": "{name}"}
    ), use_container_width=True)

    # 🖌️ Legenda genereren op basis van gebruikte kleuren
    omgekeerde_mapping = {v: k for k, v in nederlandse_naar_engelse_landen.items()}

    st.markdown(
        """
        <style>
        .legend {
            position: absolute;
            bottom: 80px;
            left: 60px;
            background-color: white;
            padding: 10px;
            border: 2px solid #ccc;
            border-radius: 8px;
            z-index: 9999;
            font-size: 14px;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.15);
        }
        .legend span {
            display: inline-block;
            width: 12px;
            height: 12px;
            margin-right: 6px;
            border-radius: 50%;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    legend_html = '<div class="legend"><b> Legenda</b><br>'
    for land_en in sorted(land_kleuren):
        land_nl = omgekeerde_mapping.get(land_en, land_en)
        kleur = land_kleuren[land_en]
        legend_html += f'<span style="background-color:{kleur}"></span>{land_nl}<br>'
    legend_html += '</div>'

    st.markdown(legend_html, unsafe_allow_html=True)
