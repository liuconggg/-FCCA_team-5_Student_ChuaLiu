import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from sklearn import preprocessing


sb.set_theme() 

jobData = pd.read_csv('job_placement.csv')
uniRankings = pd.read_csv('uni_ranking.csv')

jobData.head()
uniRankings.head()


# Description of jobData 
jobData.describe()



# Markdown 1: Cleaning of Data 
jobData.isnull().sum() # finding null values for each data
jobData[jobData['years_of_experience'].isnull()] # seeing the null row
jobData=jobData.dropna() # dropping the null row

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

le = preprocessing.LabelEncoder() 
jobData = jobData.replace("=", "" ,regex = True) 

jobData.drop(columns=['degree', 'name'], inplace=True)
jobData['gender'] = le.fit_transform(jobData['gender']) # encoding gender, 0 for female, 1 for male
jobData['stream'] = le.fit_transform(jobData['stream']) # encoding stream, 0 for computer science, 1 for information technology
jobData['college_name'] = le.fit_transform(jobData['college_name']) # encoding college_name, 0-23 represents the different schools
jobData['placement_status'] = le.fit_transform(jobData['placement_status']) # encoding placement_status, 0 for not placed, 1 for placed.

encoded_gender_counts = jobData['gender'].value_counts()

print("Number of 0s:", encoded_gender_counts[0])
print("Number of 1s:", encoded_gender_counts[1])

# Heat Map 
plt.figure(figsize=(8, 8))
sb.heatmap(jobData.corr(), annot=True)
plt.show()

# Histogram for distribution of gender
plt.figure(figsize=(10, 5))
ax = sb.countplot(x='gender', data=jobData)

for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'), 
                 (p.get_x() + p.get_width() / 2., p.get_height()), 
                 ha = 'center', va = 'center', 
                 xytext = (0, 5), 
                 textcoords = 'offset points')
    
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Distribution of Gender')
plt.xticks(ticks=[0, 1], labels=['Female', 'Male'])  # If 0 represents male and 1 represents female
plt.show()