import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn import preprocessing
from collections import defaultdict


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
jobData = jobData.replace("-", "" ,regex = True) 

collegeName = jobData["college_name"].unique() # find out how many unique college


uniRanking = pd.DataFrame(uniRankings[['2024 RANK', 'Institution Name', 'Country']]).iloc[1:] # extra only '2024 RANK' and 'Instituition Name' from row 1 onwards
# inUs = uniRanking['Country'].isin(["United States"])

# cleanedUniRankings = uniRanking[inUs].set_index('2024 RANK') 
cleanedUniRankings = uniRanking.set_index('2024 RANK') 
cleanedUniRankings['Institution Name'] = cleanedUniRankings['Institution Name'].str.replace(r'\s*\([^()]*\)', '', regex=True)
cleanedUniRankings['Institution Name'] = cleanedUniRankings['Institution Name'].str.replace(",", "" ,regex = True) 
cleanedUniRankings['Institution Name'] = cleanedUniRankings['Institution Name'].str.replace("-", "" ,regex = True) 
cleanedUniRankings['Institution Name'] = cleanedUniRankings['Institution Name'].str.replace(" at ", "" ,regex = True) 
cleanedUniRankings['Institution Name'] = cleanedUniRankings['Institution Name'].str.replace("The ", "" ,regex = True) 



# Create a defaultdict with lists as values
uniRankingDict = defaultdict(list)

# Iterate over your DataFrame rows and populate the dictionary
for index, row in cleanedUniRankings.iterrows():
    ranking = index  # Using the index as the ranking
    college_name = row['Institution Name']
    
    # Append the college name to the list corresponding to the ranking key
    uniRankingDict[ranking].append(college_name)


def remove_whitespace(text):
    return ''.join(text.split())

def find_best_match(college_name, uniRankingDict):
    """Finds the best match and returns the corresponding ranking"""
    # Preprocess college_name and university names to remove whitespace
    
    college_name = remove_whitespace(college_name)
    cleaned_uni_names = {ranking.strip('='): [remove_whitespace(name) for name in names] 
                         for ranking, names in uniRankingDict.items()}
    # Find the best match
    for ranking, uni_names in cleaned_uni_names.items():
        ranking = ''.join(filter(lambda x: x.isdigit() or x == '-', ranking))
        ranking = ranking.replace('+', '')  # Remove '+' character
        if '-' in ranking:
            start_rank, _ = map(int, ranking.split('-'))
        else:
            start_rank = int(ranking)
        
        # If college_name is found in uni_names, return the starting rank
        if college_name in uni_names:
            return start_rank

    # If no match is found, return the college_name that does not exist in the dataset
    return college_name


# Create new column 
jobData['ranking'] = jobData['college_name'].apply(lambda x: find_best_match(x, uniRankingDict))
isSanFrans  = jobData['college_name'].isin(['University of CaliforniaSan Francisco'])
jobData = jobData[~isSanFrans] # filter out the college name that is in job data but not in university_ranking  which is "University of CaliforniaSan Francisco"

courses = ['Computer Science', 'Information Technology'] 

jobData = jobData[jobData['stream'].isin(courses)] # extra only computer science and infromation technology

le = preprocessing.LabelEncoder() 
jobData = jobData.replace("=", "" ,regex = True) 

jobData.drop(columns=['degree', 'name'], inplace=True)


encoding_mappings = {}

# Encode each variable and store mappings
encoded_variables = ['gender', 'stream', 'college_name', 'placement_status']
for variable in encoded_variables:
    jobData[f'{variable}'] = le.fit_transform(jobData[variable])
    encoding_mappings[variable] = dict(zip(le.classes_, le.transform(le.classes_)))

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


# Insights

variables = ['age', 'gpa', 'years_of_experience']
plt.figure(figsize=(18, 9))

placement_status_labels = {v: k for k, v in encoding_mappings['placement_status'].items()}

# Loop with adjusted plotting
variables = ['age', 'gpa', 'years_of_experience']
plt.figure(figsize=(18, 9))

for i, variable in enumerate(variables, 1):
    plt.subplot(2, 3, i)
    sb.kdeplot(data=jobData, x=variable, hue=jobData['placement_status'].map(placement_status_labels), fill=True, alpha=0.5, linewidth=0)
    plt.xlabel(variable.capitalize())
    plt.ylabel('Density')
    plt.title(f'Density Plot of {variable.capitalize()} by Placement Status')

plt.tight_layout()
plt.show()


