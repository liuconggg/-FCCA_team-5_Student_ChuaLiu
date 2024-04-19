<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>

<h3 align="center">Predicting employability of graduates with Bachelor's in US</h3>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
The primary goal of this project is to predict the employability of graduates with Bachelor's in US, specifically in Computer Science & Information Technology. We have divided into following section to achieve the primary goal. 

Datasets:
Source: Kaggle
Dataset 1: Job placement dataset - This dataset contains This dataset contains information about Bachelor's degree graduates from various universities in the USA and their placement status. 
Dataset 2: QS World University Rankings 2024 - This dataset contains the 20th edition of the QS World University Rankings features 1,500 institutions across 104 locations.

Data Preparation:
1) Removal of null values from dataset 1.
2) Standardizing the college names from both dataset by removing specific characters and blanks.
3) Generating a new column "ranking" by merging the two dataset based on the college names.
4) Filtering out streams such that it contains only graduates from CS & IT.
5) Dropping irrelevant variables such as "degree" and "name" from dataset 1.
6) Encoding of categorical variables from dataset 1. One example for Gender. Female - 0, Male - 1.

Note:
At this stage, our merged dataset is cleaned.

Visual Representations:
(Raw dataset):
1) Bar chart - distribution of students in each stream

(Cleaned dataset):
1) Pie chart - distribution of students in different colleges.
2) Histogram - distribution of gender in both CS & IT.
3) Catplot - distribution of gender by streams (CS & IT) respectively.
4) Catplot - distribution of placement by gender.
5) Heatmap - correlation between variables.
6) KDE plot - placement_status vs gpa & placement_status vs years of experience.
7) Decision tree - gini index of our classification for placement with respect to the variables gpa & years of experience. 
8) Confusion matrix - the accuracy of our classification for placement with respect to the variables gpa & years of experience.
9) Scatter plot - prediction of salary. salary vs gpa & salary vs years of experience.
10) 3D scatter plot - displaying salary with respect to gpa & years of experience.

Machine Learning Models Used:
1) Decision Tree Classifier - a classification model to predict the employability.
2) Random Forest Classifier - a better classification model to predict the employability.
3) Linear Regression - a model to redict salary using linear regression.

Conclusion: 
We were able to achieve a good accuracy score of predicting the employability of students in US from CS & IT with the random forest classifier model. 
However, we were not able to find a optimal regression model for the salary due to the insufficient data from our cleaned dataset as we only have less than 300 data.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Chua Yong Tai Anthony - CHUA1091@e.ntu.edu.sg, Liu Cong - CLIU034@e.ntu.edu.sg

Project Link: [https://github.com/github_username/repo_name](https://github.com/github_username/repo_name)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

[Job Placement Dataset]
[QS World University Rankings 2024]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
[Job Placement Dataset]: https://www.kaggle.com/datasets/mahad049/job-placement-dataset/data
[QS World University Rankings 2024]: https://www.kaggle.com/datasets/joebeachcapital/qs-world-university-rankings-2024
