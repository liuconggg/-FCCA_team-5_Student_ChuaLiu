<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>

<h3 align="center">Predicting employability of graduates with Bachelor's in US</h3>
<p align="center">
    <a href="https://www.youtube.com/watch?v=aNG5O4Qzyfw">Link to Presentation Video</a>
  </p>
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
The primary goal of this project is to predict the employability of graduates with Bachelor's in the US, specifically in Computer Science & Information Technology. We have divided into the following sections to achieve the primary goal. 

Data Selection:
<br>
Source: Kaggle
<br>
Dataset 1: Job placement dataset - This dataset contains information about Bachelor's degree graduates from various universities in the USA and their placement status. 
<br>
Dataset 2: QS World University Rankings 2024 - This dataset contains the 20th edition of the QS World University Rankings featuring 1,500 institutions across 104 locations.

Data Preparation:
1) Removal of null values from dataset 1.
2) Standardizing the naming convention for college names from both datasets by removing specific characters and blanks.
3) Generating a new column "ranking" by merging the two datasets based on the college names.
4) Filtering out streams such that the dataset only contains graduates from CS & IT.
5) Dropping irrelevant variables such as "degree" and "name" from dataset 1.
6) Encoding of categorical variables from dataset 1. One example for Gender: Female - 0, Male - 1.

Note:
At this stage, our merged dataset is cleaned.

Visual Representations:
<br>
(Raw dataset):
1) Bar chart - distribution of students according to streams

![1](https://github.com/liuconggg/FCS4_Team-5-/assets/128717856/07f38486-8e0f-4c9b-9108-6e988805d5cd)


(Cleaned dataset):
1) Pie chart - distribution of graduates according to colleges

![2](https://github.com/liuconggg/FCS4_Team-5-/assets/128717856/6a87ff0d-b9f3-4415-a8fe-0f1b0d21fe26)

2) Histogram - distribution of graduates by gender in both CS & IT

![13](https://github.com/liuconggg/FCS4_Team-5-/assets/128717856/6321d7f5-8f71-4e8a-9233-31d63865b37b)

3) Catplot - distribution of graduates by gender according to their streams (CS & IT) respectively

![3](https://github.com/liuconggg/FCS4_Team-5-/assets/128717856/24422810-730f-47ed-b187-6342f834dd4c)

4) Catplot - distribution of placement by gender

![4](https://github.com/liuconggg/FCS4_Team-5-/assets/128717856/90c79852-d86a-4301-997c-d82dca097b04)

5) Heatmap - correlation between variables

![5](https://github.com/liuconggg/FCS4_Team-5-/assets/128717856/11d83e03-6ffa-4c8a-8c89-f5a97ce4f594)

6) KDE plot - placement_status using gpa & placement_status using years of experience.

![6](https://github.com/liuconggg/FCS4_Team-5-/assets/128717856/3b20be9f-1b48-4da8-9264-115b44a2bc95)

7) Decision tree - gini index of our classification for placement with respect to the variables GPA & years of experience. (Decision Tree Classifier)

![7](https://github.com/liuconggg/FCS4_Team-5-/assets/128717856/55147a37-105f-4444-88ac-6b3f2cd0a528)

8) Confusion matrix - the accuracy of our classification model for placement with respect to the variables GPA & years of experience. (Decision Tree Classifier)

![8](https://github.com/liuconggg/FCS4_Team-5-/assets/128717856/6036cf3a-738e-4b0b-ae20-6bcc59faca0a)

9) Decision tree - gini index of our classification for placement with respect to the variables GPA & years of experience. (Random Forest Classifier)

![9](https://github.com/liuconggg/FCS4_Team-5-/assets/128717856/16208b9e-367b-44fc-8d64-9a819e928569)

10) Confusion matrix - the accuracy of our classification model for placement with respect to the variables GPA & years of experience. (Random Forest Classifier)

![10](https://github.com/liuconggg/FCS4_Team-5-/assets/128717856/7b896df6-3a34-4de4-aa7c-50c04eeb58c0)

11) Scatter plot - prediction of salary using GPA

![11](https://github.com/liuconggg/FCS4_Team-5-/assets/128717856/43cd3ec2-8c26-456a-929a-27393ba3cc1d)

12) Scatter plot - prediction of salary using years of experience

![12](https://github.com/liuconggg/FCS4_Team-5-/assets/128717856/cd4480bf-cbbe-4444-a9b8-a6530af87344)

13) 3D scatter plot - displaying salary with respect to GPA & years of experience

![3D Scatterplot Gif](https://github.com/liuconggg/FCS4_Team-5-/assets/128717856/8028451c-f3cc-4940-8cf6-bcfc79b06343)


Machine Learning Models Used:
1) Decision Tree Classifier - a classification model to predict the employability of graduates
2) Random Forest Classifier - a better classification model to predict the employability of graduates
3) Linear Regression - a model to predict salary using linear regression

Findings:
1. College name and its world ranking, gender and age do not have much impact on the placement status of fresh graduates in US
2. GPA and years of experience can influence the placement status and the salary one can expect 

Conclusion: 
We were able to achieve a good accuracy score in predicting the employability of students in US from CS & IT with the random forest classifier model. 
However, we were not able to find a optimal regression model for the salary due to the insufficient data from our cleaned dataset as we only have less than 300 data.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Chua Yong Tai Anthony - CHUA1091@e.ntu.edu.sg, 
Liu Cong - CLIU034@e.ntu.edu.sg

Project Link: [https://github.com/github_username/repo_name](https://github.com/github_username/repo_name)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contributions
Data Preparation: Chua Yong Tai Anthony & Liu Cong
Visual Representations: Chua Yong Tai Anthony & Liu Cong
Machine Learning Models Used:
1) Decision Tree Classifier - Liu Cong
2) Random Forest Classifier - Chua Yong Tai Anthony & Liu Cong
3) Linear Regression - Chua Yong Tai Anthony

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

[Job Placement Dataset]
[QS World University Rankings 2024]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
[Job Placement Dataset]: https://www.kaggle.com/datasets/mahad049/job-placement-dataset/data
[QS World University Rankings 2024]: https://www.kaggle.com/datasets/joebeachcapital/qs-world-university-rankings-2024
