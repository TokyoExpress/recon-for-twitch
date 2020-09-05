# Project S Developer Log

Presented by <strong>TokyoExpress</strong><br /><br />


***Abstract***

This project began with a simple ambition:
<ul>
   <li>Can a neural network identify the game you're playing from a screenshot?</li>
</ul>
As I began development, I soon discovered that the answer was "yes, absolutely". But I ran into a few more overarching questions:
<ul>
   <li>What types of games should Project S be able to classify?</li>
   <li>Should Project S be able to identify correctly even when the image provided is not gameplay (e.g. loading screens or menus)?</li>
   <li>At what accuracy should optimization stop? 90%? 95%? What is the maximum capability of a model like this?</li>
   <li>What are the practical applications and extensions of Project S?</li>
</ul>

Language: **Python**

Libraries: **OpenCV**, **TensorFlow**, **Keras**, **SKLearn**<br /><br /><br /><br />

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
