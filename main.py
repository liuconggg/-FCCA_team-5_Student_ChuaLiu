import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt


sb.set_theme() 

jobData = pd.read_csv('job_placement.csv')
uniRankings = pd.read_csv('university_rankings.csv')

new_header = uniRankings.iloc[0] #grab the first row for the header
uniRankings = uniRankings[1:] #take the data less the header row
uniRankings.columns = new_header #set the header row as the df header

jobData.head()

uniRankings.head(10)

uniRankings.drop("2023", axis=1)



# Markdown 1: Cleaning of Data 
jobData.drop("name", axis=1) # dropping column "name"
collegeName = jobData["college_name"].unique()
print((collegeName))


