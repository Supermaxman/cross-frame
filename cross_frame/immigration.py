import random
import pandas as pd
from typing import Optional


problems = {
    "Economic": "Financial implications of an issue.",
    "Capacity & Resources": "The availability or lack of time, physical, human, or financial resources.",
    "Morality & Ethics": "Perspectives compelled by religion or secular sense of ethics or social responsibility.",
    "Fairness & Equality": "The (in)equality with which laws, punishments, rewards, and resources are distributed.",
    "Legality, Constitutionality & Jurisdiction": "Court cases and existing laws that regulate policies; constitutional interpretation; legal processes such as seeking asylum or obtaining citizenship; jurisdiction.",
    "Crime & Punishment": "The violation of policies in practice and the consequences of those violations.",
    "Security & Defense": "Any threat to a person, group, or nation and defenses taken to avoid that threat.",
    "Health & Safety": "Health and safety outcomes of a policy issue, discussions of health care.",
    "Quality of Life": "Effects on people’s wealth, mobility, daily routines, community life, happiness, etc.",
    "Cultural Identity": "Social norms, trends, values, and customs; integration/assimilation efforts.",
    "Public Sentiment": "General social attitudes, protests, polling, interest groups, public passage of laws.",
    "Political Factors & Implications": "Focus on politicians, political parties, governing bodies, political campaigns and debates; discussions of elections and voting.",
    "Policy Prescription & Evaluation": "Discussions of existing or proposed policies and their effectiveness.",
    "External Regulation & Reputation": "Relations between nations or states/provinces; agreements between governments; perceptions of one nation/state by another.",
    "Victim: Global Economy": "Immigrants are victims of global poverty, underdevelopment, and inequality.",
    "Victim: Humanitarian": "Immigrants experience economic, social, and political suffering and hardships.",
    "Victim: War": "Focus on war and violent conflict as reasons for immigration.",
    "Victim: Discrimination": "Immigrants are victims of racism, xenophobia, and religion-based discrimination.",
    "Hero: Cultural Diversity": "Highlights positive aspects of differences that immigrants bring to society.",
    "Hero: Integration": "Immigrants successfully adapt and fit into their host society.",
    "Hero: Worker": "Immigrants contribute to economic prosperity and are an important source of labor.",
    "Threat: Jobs": "Immigrants take nonimmigrants’ jobs or lower their wages.",
    "Threat: Public Order": "Immigrants threaten public safety by breaking the law or spreading disease.",
    "Threat: Fiscal": "Immigrants abuse social service programs and are a burden on resources.",
    "Threat: National Cohesion": "Immigrants’ cultural differences are a threat to national unity and social harmony.",
    "Episodic": "Message provides concrete information about specific people, places, or events.",
    "Thematic": "Message is more abstract, placing stories in broader political and social contexts.",
}


def load_immigration(data_path: str):
    df = pd.read_excel(data_path)
    data = []
    f_columns = df.columns[3:-1]
    for idx, row in df.iterrows():
        total_problems = sum([int(row[f_column]) for f_column in f_columns])
        if total_problems <= 0:
            continue
        data.append({"id": str(row["id"]), "text": row["text"]})
    return data, problems
