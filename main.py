import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn import preprocessing
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
import graphviz
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error

jobData = pd.read_csv('job_placement.csv')
uniRankings = pd.read_csv('uni_ranking.csv')

jobData.head()
jobData.describe()


uniRankings.head()
uniRankings.describe()

# Cleaning of Data 
jobData.isnull().sum() # finding null values for each data
jobData[jobData['years_of_experience'].isnull()] # seeing the null row
jobData=jobData.dropna() # dropping the null row
jobData = jobData.replace("--", "" ,regex = True) 
jobData = jobData.replace("-", "" ,regex = True) 

collegeName = jobData["college_name"].unique() # find out how many unique college
uniRanking = pd.DataFrame(uniRankings[['2024 RANK', 'Institution Name', 'Country']]).iloc[1:] # extra only '2024 RANK' and 'Instituition Name' from row 1 onwards
inUs = uniRanking['Country'].isin(["United States"])

cleanedUniRankings = uniRanking[inUs].set_index('2024 RANK') 
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
isSanFrans  = jobData['college_name'].isin(['University of CaliforniaSan Francisco']) # name of university not in ranking dataset
jobData = jobData[~isSanFrans] # filter out the college name that is in job data but not in university_ranking  which is "University of CaliforniaSan Francisco"

# bar chart for distribution of students in each stream 
plt.figure(figsize=(15, 10))
ax = sb.countplot(data=jobData, x='stream')

# Annotate each bar with its count
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 10), 
                   textcoords = 'offset points')

ax.set_ylabel('Count', labelpad=10) 
ax.set_xlabel("Stream", labelpad=10)  
plt.title('Distribution of Student by Stream')
plt.show()

# Further data cleaning and preparation by encoding categoricable variables and focusing on computer science and information technology
courses = ['Computer Science', 'Information Technology'] 
jobData = jobData[jobData['stream'].isin(courses)] # extract only computer science and infromation technology

le = preprocessing.LabelEncoder() 
jobData = jobData.replace("=", "" ,regex = True) 
jobData.drop(columns=['degree', 'name'], inplace=True)
encoding_mappings = {}

# Encode each variable and store mappings
encoded_variables = ['gender', 'stream', 'college_name', 'placement_status']
for variable in encoded_variables:
    jobData[f'{variable}'] = le.fit_transform(jobData[variable])
    encoding_mappings[variable] = dict(zip(le.classes_, le.transform(le.classes_)))

jobData.head()
jobData.describe()

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

#Pie chart for distribution of students in different colleges
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

#Catplot for Placement by Gender
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

# Heat Map to show correlation between variables
plt.figure(figsize=(16, 10))
heatmap = sb.heatmap(jobData.corr(), annot=True)
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=0)
plt.title('Heat Map')
plt.show()

#KDE plot GPA and Years of Experience
placement_status_labels = {v: k for k, v in encoding_mappings['placement_status'].items()}
variables = ["gpa", "years_of_experience"]
plt.figure(figsize=(18, 9))

for i, variable in enumerate(variables, 1):
    plt.subplot(2, 2, i)
    sb.kdeplot(data=jobData, x=variable, hue=jobData['placement_status'].map(placement_status_labels), fill=True, alpha=0.5, linewidth=0.5)
    plt.xlabel(variable.capitalize())
    plt.ylabel('Density')
    plt.title(f'Density Plot for {variable.capitalize()} by Placement Status')
plt.tight_layout()
plt.show()

# Classification
y = pd.DataFrame(jobData["placement_status"])
X = pd.DataFrame(jobData[variables])

# Split the Dataset into Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

trainDF = pd.concat([y_train, X_train], axis = 1).reindex(y_train.index)

dectree = DecisionTreeClassifier(max_depth = 3)  # create the decision tree object
dectree.fit(X_train, y_train)  

f = plt.figure(figsize=(12,12))
plot_tree(dectree, filled=True, rounded=True, 
          feature_names=X_train.columns.tolist(),
          class_names=["not place","placed"])        

# Predict Legendary values corresponding to Total
y_train_pred = dectree.predict(X_train)
y_test_pred = dectree.predict(X_test)

# Check the Goodness of Fit (on Train Data)
print("Goodness of Fit of Model \tTrain Dataset")
print("Classification Accuracy \t:", dectree.score(X_train, y_train))
print()

# Check the Goodness of Fit (on Test Data)
print("Goodness of Fit of Model \tTest Dataset")
print("Classification Accuracy \t:", dectree.score(X_test, y_test))
print()

f, axes = plt.subplots(1, 2, figsize=(12, 4))

# Confusion matrix for the training set
train_cm = confusion_matrix(y_train, y_train_pred)
sb.heatmap(train_cm, annot=True, fmt=".0f", annot_kws={"size": 18}, ax=axes[0])
axes[0].set_title('Train Confusion Matrix', fontsize=12)  # Set the title

# Confusion matrix for the test set
test_cm = confusion_matrix(y_test, y_test_pred)
sb.heatmap(test_cm, annot=True, fmt=".0f", annot_kws={"size": 18}, ax=axes[1])
axes[1].set_title('Test Confusion Matrix', fontsize=12)  # Set the title

# Inserting a title in the middle
f.suptitle('Classification Confusion Matrix for GPA and Years of Experience with respect to Placement Status', fontsize=14, y=1.05)
plt.tight_layout()  # Adjust layout
plt.show()

# Random Forest Classification
y = pd.DataFrame(jobData["placement_status"])
X = pd.DataFrame(jobData[variables])

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

# 100 decision trees and use random forest cross-validation and no limit for processor
classifier_rf = RandomForestClassifier(n_jobs=-1, max_depth=5,
                                       n_estimators=100, oob_score=True)

classifier_rf.fit(X_train, y_train)

y_pred = classifier_rf.predict(X_test)

# Calculate Model Accuracy
print("Random Forest Classifier Accuracy:", accuracy_score(y_test, y_pred))

# show only the first three decision tree
for i in range(3):
    tree = classifier_rf.estimators_[i]
    dot_data = export_graphviz(tree,
                               feature_names=X_train.columns,  
                               filled=True,  
                               max_depth=2, 
                               impurity=False, 
                               proportion=True)
    graph = graphviz.Source(dot_data)

# display(graph)
y_train_pred = classifier_rf.predict(X_train)
y_test_pred = classifier_rf.predict(X_test)

# Plotting
f, axes = plt.subplots(1, 2, figsize=(12, 4))

# Confusion matrix for the training set
train_cm = confusion_matrix(y_train, y_train_pred)
sb.heatmap(train_cm, annot=True, fmt=".0f", annot_kws={"size": 18}, ax=axes[0])
axes[0].set_title('Train Confusion Matrix', fontsize=12)  # Set the title

# Confusion matrix for the test set
test_cm = confusion_matrix(y_test, y_test_pred)
sb.heatmap(test_cm, annot=True, fmt=".0f", annot_kws={"size": 18}, ax=axes[1])
axes[1].set_title('Test Confusion Matrix', fontsize=12)  # Set the title

# Inserting a title in the middle
f.suptitle('Random Forest Classification Matrix for GPA and Years of Experience with respect to Placement Status', fontsize=14, y=1.05)

plt.tight_layout()  # Adjust layout
plt.show()

#Linear Regression (GPA) 
jobDataWithSalary = pd.DataFrame(jobData[jobData['salary'] > 0])
jobDataWithSalary.head()

x = pd.DataFrame(jobDataWithSalary['gpa']) #Predictor: GPA 
x.head()
y = pd.DataFrame(jobDataWithSalary['salary']) #Response: Salary
y.head()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)
y_train.describe()
x_train.describe()

linreg = LinearRegression()
linreg.fit(x_train, y_train)

y_train_pred = linreg.predict(x_train)
y_test_pred = linreg.predict(x_test)

# Create a figure and subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Plot for train data
axes[0].scatter(x_train, y_train, color="blue", label="True values")
axes[0].plot(x_train, y_train_pred, color='red', label="Predicted values")
axes[0].set_title("Train Data (Linear Regression)")
axes[0].set_xlabel("GPA")
axes[0].set_ylabel("Salary")
axes[0].legend()

# Plot for test data
axes[1].scatter(x_test, y_test, color="blue", label="True values")
axes[1].plot(x_test, y_test_pred, color='red', label="Predicted values")
axes[1].set_title("Test Data (Linear Regression)")
axes[1].set_xlabel("GPA")
axes[1].set_ylabel("Salary")
axes[1].legend()

# Show plots
plt.tight_layout()
plt.show()

# Check the Goodness of Fit (on Train Data)
print("Goodness of Fit of Model \tTrain Dataset (GPA)")
print("Explained Variance (R^2) \t:", linreg.score(x_train, y_train))
print("Mean Squared Error (MSE) \t:", mean_squared_error(y_train, y_train_pred))
print()

# Check the Goodness of Fit (on Test Data)
print("Goodness of Fit of Model \tTest Dataset (GPA)")
print("Explained Variance (R^2) \t:", linreg.score(x_test, y_test))
print("Mean Squared Error (MSE) \t:", mean_squared_error(y_test, y_test_pred))
print()


#Linear Regression (Years of Experience)
x = pd.DataFrame(jobDataWithSalary['years_of_experience']) #Predictor: Years of experience
x.head()
y = pd.DataFrame(jobDataWithSalary['salary']) #Response: salary
y.head()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)
y_train.describe()
x_train.describe()

linreg = LinearRegression()
linreg.fit(x_train, y_train)

y_train_pred = linreg.predict(x_train)
y_test_pred = linreg.predict(x_test)

# Create a figure and subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Plot for train data
axes[0].scatter(x_train, y_train, color="blue", label="True values")
axes[0].plot(x_train, y_train_pred, color='red', label="Predicted values")
axes[0].set_title("Train Data (Linear Regression)")
axes[0].set_xlabel("Years of Experience")
axes[0].set_ylabel("Salary")
axes[0].legend()

# Plot for test data
axes[1].scatter(x_test, y_test, color="blue", label="True values")
axes[1].plot(x_test, y_test_pred, color='black', label="Predicted values")
axes[1].set_title("Test Data (Linear Regression)")
axes[1].set_xlabel("Years of Experience")
axes[1].set_ylabel("Salary")
axes[1].legend()

# Show plots
plt.tight_layout()
plt.show()

# Check the Goodness of Fit (on Train Data)
print("Goodness of Fit of Model \tTrain Dataset (Years of Experience)")
print("Explained Variance (R^2) \t:", linreg.score(x_train, y_train))
print("Mean Squared Error (MSE) \t:", mean_squared_error(y_train, y_train_pred))
print()

# Check the Goodness of Fit (on Test Data)
print("Goodness of Fit of Model \tTest Dataset (Years of Experience)")
print("Explained Variance (R^2) \t:", linreg.score(x_test, y_test))
print("Mean Squared Error (MSE) \t:", mean_squared_error(y_test, y_test_pred))
print()
