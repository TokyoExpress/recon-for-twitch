# Project S Developer Log

Presented by <strong>TokyoExpress</strong>

***Abstract***

This project is aimed at the following questions:
<ol>
   <li>What are the overarching similarities and key features of successful Kickstarter projects?</li>
   <li>How accurately can one predict project success or failure, given the characteristics of a new project?</li>
</ol>
As follows, the purpose of this project is to gain insight on what truly makes up a successul Kickstarter campaign, and whether or not that success can be replicated. The importance of having the regression analysis in addition to the machine learning model is to highlight the reasons and concepts behind an accurate prediction. Just as essential as the ability to have a ML program predict project success is the understanding of the underlying forces and factors that guide the prediction, and as such we have made it a priority of this project to draw real, conversable conclusions for data scientists and casual readers alike.<br /><br />

On the technical side, the project is done in **R** and **Python**, as is standard. Notable R packages include **dplyr**, **ggplot**, and **Shiny**, while the bulk of the Python work is done with the assistance of the essential machine learning library **sklearn**. An honorable mention to **Microsoft Excel** for many of the graphs and visuals.<br /><br />

***Premise and Motivation***

Kickstarter has been one of the premiere crowdfunding platforms since its launch in 2009. It is now home to over 445,000 successfully backed projects. While crowdfunding remains a great resource for any aspiring product designers and entrepreneurs, taking care of a campaign still requires a decently significant amount of time and money, with no guarantee or indicator of success. This project attempts to provide potential campaign managers with information and insight that can be used to maximize the probability of success for a certain project, as well as provide areas of interest that can later be researched further by the project manager.<br /><br />

***Table of Contents***

* [Prepping Data](#prepping-data)
* [Exploratory Insight](#exploratory-insight)
* [Statistical Models and Analysis](#statistical-models-and-analysis)
* [Machine Learning](#machine-learning)
* [Relevant Conclusions and Applications](#relevant-conclusions-and-applications)
* [Next Steps](#next-steps)<br /><br />

## Prepping Data

The raw data for this project comes from Kaggle: https://www.kaggle.com/yashkantharia/kickstarter-campaigns/data. It's a 32 MB dataset with 170731 unique projects, along with the following variables:

<ul>
  <li>ID</li>
  <li>Name</li>
