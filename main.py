import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

sb.set_theme() 

jobData = pd.read_csv('job_placement.csv')
uniRankings = pd.read_csv('uni_ranking.csv')

jobData.head()
uniRankings.head()


# Markdown 1: Cleaning of Data 
jobData = jobData.drop("name", axis=1) # dropping column "name"
jobData = jobData.replace("--", "" ,regex = True)

collegeName = jobData["college_name"].unique() # find out how many unique college
uniRanking = pd.DataFrame(uniRankings[['2024 RANK', 'Institution Name']]).iloc[1:] # extra only '2024 RANK' and 'Instituition Name' from row 1 onwards
cleanedUniRankings = uniRanking.set_index('2024 RANK') 
uniRankingDict = cleanedUniRankings.to_dict('dict') # convert to dictionary for comparison later on
uniRankingDict= uniRankingDict['Institution Name'] # accessing the dictionary 


def find_best_match(college_name, uniRankingDict):
    """Finds the best match and returns the corresponding ranking"""
    choices = list(uniRankingDict.values())  # Use values as choices
    best_match = process.extractOne(college_name, choices, scorer=fuzz.ratio)
    if best_match[1] >= 80: # ratio of 80 and above 
        for ranking, uni_name in uniRankingDict.items():
            if uni_name == best_match[0]:
                return ranking
    else:
        return None  # Or a default value for no match

# Create new column 
jobData['ranking'] = jobData['college_name'].apply(lambda x: find_best_match(x, uniRankingDict))

jobData.head()

courses = ['Computer Science', 'Information Technology'] 

jobData = jobData[jobData['stream'].isin(courses)] # extra only computer science and infromation technology
print(len(jobData))

