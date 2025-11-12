import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import hashlib
from SPARQLWrapper import SPARQLWrapper, JSON
from datetime import datetime

DATA_PATH = 'indian_startup_funding.csv'

USERS = {
    "user1": hashlib.sha256("pass1".encode()).hexdigest(),
    "user2": hashlib.sha256("pass2".encode()).hexdigest(),
}

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data['Starting Year'] = pd.to_numeric(data['Starting Year'], errors='coerce')
    data.dropna(subset=['Starting Year'], inplace=True)
    data['Starting Year'] = data['Starting Year'].astype(int)
    return data

def encode_input(value, categories):
    if value in categories:
        return categories.get_loc(value)
    else:
        return categories.get_loc('unknown')

def show_eda_tabs(data):
    tabs = st.tabs(["Overview", "City Analysis", "Industry Analysis", "Funding Analysis", "Founders Analysis", "Time Trends"])
    with tabs[0]:
        st.header("Dataset Overview")
        st.write(data.head())
        st.write("Summary statistics:")
        st.write(data.describe(include='all'))
    with tabs[1]:
        st.header("City Analysis")
        city_counts = data['City'].value_counts().head(10)
        st.bar_chart(city_counts)
        avg_funding = data.groupby('City')['Funding Amount in $'].mean().sort_values(ascending=False).head(10)
        st.bar_chart(avg_funding)
    with tabs[2]:
        st.header("Industry Analysis")
        industry_funding = data.groupby('Industries')['Funding Amount in $'].sum().sort_values(ascending=False).head(10)
        st.bar_chart(industry_funding)
        median_funding = data['Funding Amount in $'].median()
        data['Success'] = (data['Funding Amount in $'] > median_funding).astype(int)
        success_rate = data.groupby('Industries')['Success'].mean().sort_values(ascending=False).head(10)
        st.bar_chart(success_rate)
    with tabs[3]:
        st.header("Funding Rounds Analysis")
        fig, ax = plt.subplots()
        sns.boxplot(data=data, x='Funding Round', y='Funding Amount in $', ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        st.header("Funding vs Number of Investors")
        fig, ax = plt.subplots()
        sns.scatterplot(data=data, x='No. of Investors', y='Funding Amount in $', ax=ax)
        st.pyplot(fig)
    with tabs[4]:
        st.header("Founders Analysis")
        founders = data['Founders'].dropna().str.split(', ').explode()
        founder_counts = Counter(founders)
        top_founders = pd.Series(founder_counts).sort_values(ascending=False).head(10)
        st.bar_chart(top_founders)
    with tabs[5]:
        st.header("Startup Growth Over Time")
        yearly_count = data['Starting Year'].value_counts().sort_index()
        st.line_chart(yearly_count)

def compare_startups_ui(model, city_cat, ind_cat, emp_cat):
    st.header("Compare Startups")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Startup 1")
        city1 = st.text_input("City (Startup 1)", key='city1')
        year1 = st.number_input("Starting Year (Startup 1)", min_value=1900, max_value=2100, step=1, key='year1')
        ind1 = st.text_input("Industries (Startup 1)", key='ind1')
        emp1 = st.text_input("No. of Employees (Startup 1)", key='emp1')
        fr1 = st.number_input("Funding Round (Startup 1)", min_value=1, step=1, key='fr1')
        inv1 = st.number_input("No. of Investors (Startup 1)", min_value=0, step=1, key='inv1')
        fund1 = st.number_input("Funding Amount (Startup 1)", min_value=0, step=1, key='fund1')
    with col2:
        st.subheader("Startup 2")
        city2 = st.text_input("City (Startup 2)", key='city2')
        year2 = st.number_input("Starting Year (Startup 2)", min_value=1900, max_value=2100, step=1, key='year2')
        ind2 = st.text_input("Industries (Startup 2)", key='ind2')
        emp2 = st.text_input("No. of Employees (Startup 2)", key='emp2')
        fr2 = st.number_input("Funding Round (Startup 2)", min_value=1, step=1, key='fr2')
        inv2 = st.number_input("No. of Investors (Startup 2)", min_value=0, step=1, key='inv2')
        fund2 = st.number_input("Funding Amount (Startup 2)", min_value=0, step=1, key='fund2')
    if st.button("Compare Predictions"):
        s1_features = np.array([[encode_input(city1, city_cat), year1, encode_input(ind1, ind_cat), encode_input(emp1, emp_cat), fr1, inv1, fund1]])
        s2_features = np.array([[encode_input(city2, city_cat), year2, encode_input(ind2, ind_cat), encode_input(emp2, emp_cat), fr2, inv2, fund2]])
        pred1 = model.predict(s1_features)[0]
        pred2 = model.predict(s2_features)[0]
        results = {
            'Feature': ['City', 'Starting Year', 'Industries', 'No. of Employees', 'Funding Round', 'No. of Investors', 'Funding Amount'],
            'Startup 1': [city1, year1, ind1, emp1, fr1, inv1, fund1],
            'Startup 1 Prediction': ['Success' if pred1 == 1 else 'Failure']*7,
            'Startup 2': [city2, year2, ind2, emp2, fr2, inv2, fund2],
            'Startup 2 Prediction': ['Success' if pred2 == 1 else 'Failure']*7
        }
        st.table(results)

def show_saved_profiles():
    if 'saved_startups' not in st.session_state:
        st.session_state.saved_startups = []
    st.header("Saved Startup Profiles")
    if len(st.session_state.saved_startups) == 0:
        st.write("No saved startups yet.")
        return
    for i, entry in enumerate(st.session_state.saved_startups):
        st.subheader(f"Startup {i+1}")
        for key, value in entry.items():
            st.write(f"**{key}:** {value}")
        if st.button(f"Delete Startup {i+1}", key=f"delete_{i}"):
            st.session_state.saved_startups.pop(i)
            st.experimental_rerun()

def login():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if st.session_state.logged_in:
        st.sidebar.write("Logged in!")
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
        return True
    st.sidebar.header("Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if username in USERS and USERS[username] == hash_password(password):
            st.session_state.logged_in = True
            st.sidebar.success("Successfully logged in. Please refresh the page manually if needed.")
        else:
            st.sidebar.error("Invalid username or password")
    return st.session_state.logged_in

# ---- Competitor Analysis Functions ----
industry_mapping = {
    "edutech": ["education industry", "educational technology"],
    "education": ["education industry", "educational technology"],
    "software": ["software industry", "computer software industry"],
    "fintech": ["financial technology industry"],
    # Extend as needed
}

def query_wikidata_startups_for_industry_label(industry_label, limit=20):
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    query = f"""
    SELECT ?company ?companyLabel ?inception ?headquartersLabel ?website WHERE {{
      ?company wdt:P31 wd:Q4830453;
               wdt:P452 ?industry;
               wdt:P17 wd:Q668.
      OPTIONAL {{ ?company wdt:P571 ?inception. }}
      OPTIONAL {{ ?company wdt:P159 ?headquarters. }}
      OPTIONAL {{ ?company wdt:P856 ?website. }}
      ?industry rdfs:label ?industryLabel.
      FILTER(CONTAINS(LCASE(?industryLabel), "{industry_label.lower()}"))
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
    }}
    LIMIT {limit}
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    startups = {}
    for result in results["results"]["bindings"]:
        name = result["companyLabel"]["value"]
        if name not in startups:
            inception = result.get("inception", {}).get("value", None)
            headquarters = result.get("headquartersLabel", {}).get("value", "N/A")
            website = result.get("website", {}).get("value", "N/A")
            # Remove 'T00:00:00Z' from founded date if present
            if inception and "T" in inception:
                inception = inception.split("T")[0]
            startups[name] = {
                "founded": inception,
                "headquarters": headquarters,
                "website": website
            }
    return startups

def query_wikidata_startups_with_mapping(user_input):
    mapped_industries = industry_mapping.get(user_input.lower(), [user_input])
    all_startups = {}
    for industry_label in mapped_industries:
        startups = query_wikidata_startups_for_industry_label(industry_label)
        all_startups.update(startups)
    return all_startups

def analyze_competitors(startups):
    founded_years = []
    for s in startups.values():
        try:
            founded_years.append(int(s['founded'][:4]))
        except:
            pass
    analysis = {
        "total_competitors": len(startups),
        "earliest_founded": min(founded_years) if founded_years else "N/A",
        "average_founding_year": int(sum(founded_years) / len(founded_years)) if founded_years else "N/A"
    }
    return analysis

# ---- Main app with tabs and competitor analysis ----
def main():
    data = load_data()
    model = joblib.load('startup_success_model.pkl')
    city_cat = joblib.load('city_categories.pkl')
    ind_cat = joblib.load('industries_categories.pkl')
    emp_cat = joblib.load('employees_categories.pkl')
    logged_in = login()
    if not logged_in:
        return

    st.title("Indian Startup Success Prediction & Analysis Dashboard")

    # --- Sidebar prediction ---
    st.sidebar.header("Predict & Save Startup")
    city = st.sidebar.text_input('City')
    starting_year = st.sidebar.number_input('Starting Year', min_value=1900, max_value=2100, step=1)
    industries = st.sidebar.text_input('Industries (comma separated)')
    no_of_employees = st.sidebar.text_input('No. of Employees (range category)')
    funding_round = st.sidebar.number_input('Funding Round', min_value=1, step=1)
    no_of_investors = st.sidebar.number_input('Number of Investors', min_value=0, step=1)
    funding_amount = st.sidebar.number_input('Funding Amount in $', min_value=0, step=1)
    if st.sidebar.button('Predict & Save'):
        city_enc = encode_input(city, city_cat)
        ind_enc = encode_input(industries, ind_cat)
        emp_enc = encode_input(no_of_employees, emp_cat)
        inp = np.array([[city_enc, starting_year, ind_enc, emp_enc, funding_round, no_of_investors, funding_amount]])
        pred = model.predict(inp)[0]
        prediction = "Success 🚀" if pred == 1 else "Failure 😞"
        st.sidebar.success(f"Prediction: Likely to {prediction}")
        entry = {
            'City': city,
            'Starting Year': starting_year,
            'Industries': industries,
            'No. of Employees': no_of_employees,
            'Funding Round': funding_round,
            'No. of Investors': no_of_investors,
            'Funding Amount': funding_amount,
            'Prediction': prediction
        }
        if 'saved_startups' not in st.session_state:
            st.session_state.saved_startups = []
        st.session_state.saved_startups.append(entry)

    # --- Feature Analysis ---
    show_eda_tabs(data)
    st.markdown("---")

    # --- Competitor Analysis ---
    st.header("Wikidata Indian Competitor Analysis")
    industry_input = st.text_input("Enter your startup's industry for competitor analysis")
    if st.button("Fetch Competitors"):
        if industry_input.strip() != "":
            with st.spinner("Fetching competitor startups..."):
                startups = query_wikidata_startups_with_mapping(industry_input)
                analysis = analyze_competitors(startups)
            if analysis['total_competitors'] > 0:
                st.write(f"Found **{analysis['total_competitors']}** Indian competitors in industry **{industry_input}**")
                for name, details in startups.items():
                    st.markdown(f"- **{name}**  \n Founded: {details['founded'] or 'N/A'}  \n Headquarters: {details['headquarters']}  \n Website: [Link]({details['website']})" if details['website'] != 'N/A' else f"- **{name}**  \n Founded: {details['founded'] or 'N/A'}  \n Headquarters: {details['headquarters']}")
                st.markdown("### Competitor Summary")
                st.write(f"**Total Competitors:** {analysis['total_competitors']}")
                st.write(f"**Earliest Founded Year:** {analysis['earliest_founded']}")
                st.write(f"**Average Founding Year:** {analysis['average_founding_year']}")
            else:
                st.warning("No competitor startups found for the provided industry.")
        else:
            st.error("Please enter a valid industry name.")

    st.markdown("---")
    compare_startups_ui(model, city_cat, ind_cat, emp_cat)
    st.markdown("---")
    show_saved_profiles()

if __name__ == '__main__':
    main()
