from SPARQLWrapper import SPARQLWrapper, JSON
from datetime import datetime

industry_mapping = {
    "edutech": ["education industry", "educational technology"],
    "education": ["education industry", "educational technology"],
    "software": ["software industry", "computer software industry"],
    "fintech": ["financial technology industry"],
    # Add more mappings as necessary
}

def query_wikidata_startups_for_industry_label(industry_label, limit=20):
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    query = f"""
    SELECT ?company ?companyLabel ?inception ?headquartersLabel ?website WHERE {{
      ?company wdt:P31 wd:Q4830453;
               wdt:P452 ?industry;
               wdt:P17 wd:Q668.         # country India
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
            founded_years.append(datetime.strptime(s['founded'][:4], "%Y").year)
        except:
            pass
    
    analysis = {
        "total_competitors": len(startups),
        "earliest_founded": min(founded_years) if founded_years else "N/A",
        "average_founding_year": int(sum(founded_years) / len(founded_years)) if founded_years else "N/A"
    }
    return analysis

def main():
    industry = input("Enter your startup's industry: ").strip()
    startups = query_wikidata_startups_with_mapping(industry)
    analysis = analyze_competitors(startups)
    
    print(f"\nFound {analysis['total_competitors']} Indian competitors in industry '{industry}':\n")
    for name, details in startups.items():
        print(f"- {name}")
        print(f"  Founded: {details['founded'] or 'N/A'}")
        print(f"  Headquarters: {details['headquarters']}")
        print(f"  Website: {details['website']}\n")
    
    print("Competitor Analysis Summary:")
    print(f" - Total Competitors: {analysis['total_competitors']}")
    print(f" - Earliest Founded Year: {analysis['earliest_founded']}")
    print(f" - Average Founding Year: {analysis['average_founding_year']}")

if __name__ == "__main__":
    main()
