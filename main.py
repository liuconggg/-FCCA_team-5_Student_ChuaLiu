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
inUs = uniRanking['Country'].isin(["United States"])

cleanedUniRankings = uniRanking[inUs].set_index('2024 RANK') 
# cleanedUniRankings = uniRanking.set_index('2024 RANK') 
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
plt.figure(figsize=(16, 10))
heatmap = sb.heatmap(jobData.corr(), annot=True)
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=0)

plt.show()

# Histogram for distribution of gender
plt.figure(figsize=(10, 5))
sb.set_palette("pastel")
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

placement_status_labels = {v: k for k, v in encoding_mappings['placement_status'].items()}

# Loop with adjusted plotting
variables = ['gpa', 'years_of_experience']
plt.figure(figsize=(18, 9))

for i, variable in enumerate(variables, 1):
    plt.subplot(2, 3, i)
    sb.kdeplot(data=jobData, x=variable, hue=jobData['placement_status'].map(placement_status_labels), fill=True, alpha=0.5, linewidth=0)
    plt.xlabel(variable.capitalize())
    plt.ylabel('Density')
    plt.title(f'Density Plot of {variable.capitalize()} by Placement Status')

plt.tight_layout()
plt.show()


college_counts = jobData['college_name'].value_counts()

college_percentages = (college_counts / college_counts.sum()) * 100

# Group colleges with less than a threshold (e.g., 3%) into "Others" category
threshold = 3
mask = (college_percentages < threshold)
college_counts_grouped = college_counts[~mask]
college_counts_grouped['Others'] = college_counts[mask].sum()

def map_college_name(index):
    # Check if the index is 'Others', return it directly
    if index == 'Others':
        return 'Others'
    else:
        # Get the corresponding college name from the encoding mappings
        return list(encoding_mappings['college_name'].keys())[index]

# Map the index values (college names) to their corresponding labels
mapped_labels = college_counts_grouped.index.map(map_college_name) 

def autopct_format(values):
    def my_format(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))
        return f'{val} ({pct:.1f}%)'
    return my_format

colors = plt.cm.tab20c.colors[:len(college_counts_grouped)]

# Plot the pie chart with the mapped labels
plt.figure(figsize=(20, 10))
plt.pie(college_counts_grouped, autopct = autopct_format(college_counts_grouped), startangle=90, colors=colors, textprops={'fontsize': 12}, pctdistance=0.875, labeldistance=1.1)
plt.title(f'Distribution of Students in Different Schools ({college_counts.sum()})')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Adding legend
plt.legend(labels = mapped_labels, title="Colleges", loc="center left", bbox_to_anchor=(0.8, 0, 0.5, 1))
plt.show()


#Catplot for Placement 
plt.figure(figsize=(15, 10))
sb.set_palette("pastel")
graph = sb.catplot(y="placement_status", hue="gender", data=jobData, kind="count", height=8, hue_order=range(len(encoding_mappings['gender'])), legend=False)
ax = graph.axes[0, 0]

# Add count labels to the bars
for p in ax.patches:
    ax.annotate(format(p.get_width(), '.0f'), 
                 (p.get_x() + p.get_width(), p.get_y() + p.get_height() / 2.), 
                 ha = 'center', va = 'center', 
                 xytext = (12, 0), 
                 textcoords = 'offset points',
                 fontsize=12)

ax.set_ylabel('') 
ax.set_xlabel("Count", labelpad=10)  
ax.set_title("Placement Status for graduates", pad=15)
plt.yticks(ticks=range(len(encoding_mappings['placement_status'])), labels=list(encoding_mappings['placement_status'].keys()))

legend_labels = [f"{k}" for k, v in encoding_mappings['gender'].items()]
plt.legend(labels=legend_labels, title="Gender", loc="upper right")
plt.show()

