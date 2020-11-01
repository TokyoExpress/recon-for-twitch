<img width=500px src="https://i.imgur.com/50XEGn7.png"></img>
#

Presented by <strong>TokyoExpress</strong><br /><br />

***Abstract***

Recon for Twitch is a Google Chrome extension that allows you to sort Twitch channels by what state of the game they are in (e.g. Champion Select or In-Game for League of Legends).<br /><br />



***Premise***

Recon is a set of neural net models built around the data format of video game screenshots.

The project began with a simple ambition:
<ul>
   <li>Can a neural network identify the game you're playing from a screenshot?</li>
</ul>

As I began development, I soon discovered that the answer was "yes, absolutely". But I ran into a few more overarching questions:

<ul>
   <li>What types of games should Recon be able to classify?</li>
   <li>Should Recon be able to identify correctly even when the image provided is not gameplay (e.g. loading screens or menus)?</li>
   <li>At what accuracy should optimization stop? 90%? 95%? What is the maximum capability of a model like this?</li>
   <li>What are the practical applications and extensions of Recon?</li>
</ul>

Language: **Python**

Libraries: **OpenCV**, **TensorFlow**, **Keras**, **SKLearn**<br /><br /><br /><br />

***Developer Log***

* [Chapter 0: Preliminary Test of Viability](#chapter-0-preliminary-test-of-viability)
   * [A Little Further In](#a-little-further-in)
   * [Reality Check](#reality-check)
* [Chapter 1: League of Legends](#chapter-1-league-of-legends)
   * [Not Object Detection, Template Matching](#not-object-detection-template-matching)<br /><br /><br /><br />

# Chapter 0: Preliminary Test of Viability

In order for this to have any future, we just have to make sure that a neural net is actually capable of working with screenshots as data. I have confidence in machine learning, but it's always better to be safe and build one of these simple models first.

By scraping thumbnails from Twitch livestreams, I assembled a small dataset of around 100-200 images of the following games:

<ol>
   <li>League of Legends</li>
   <li>VALORANT</li>
   <li>Counter-Strike: Global Offensive</li>
   <li>Overwatch</li>
</ol>

<img width=750px src="https://i.imgur.com/PZJdKGT.png"></img>

Then I just copied the layers and nodes from the architecture of a pretty basic Keras dogs-vs-cats classifier. Because of the low amount of training data, I ran it for 175 epochs: enough times so that the accuracy would substantially improve but not too many that the model would overfit. The results were not bad: 90% accuracy.

<img width=500px src="https://i.imgur.com/ezI34zF.png"></img>

Using another file to test completely new images on the model, I found that the model was actually pretty legitimate, being able to differentiate and classify most pictures with high probability/accuracy.


<img width=500px src="https://i.imgur.com/XR1qeYV.png"></img>

<img width=500px src="https://i.imgur.com/Ufbmce4.png"></img>

<img width=500px src="https://i.imgur.com/tYttHrJ.png"></img>

<img width=500px src="https://i.imgur.com/0nPJUUu.png"></img>

On most of the test images, at least.

<img width=500px src="https://i.imgur.com/J2qZeQ5.png"></img>

But 90% was pretty good! Far better than I expected the model to perform, which was a good enough greenlight for me to move on.

## A Little Further In

So with the preliminary test passed, I proceeded to move on to developing deeper and more practical models. I attempted to fit a model twice as large, containing 8 of the most viewed games on Twitch: 

<ol>
   <li>Apex Legends</li>
   <li>Fall Guys</li>
   <li>Fortnite</li>
   <li>League of Legends</li>
   <li>Call of Duty: Modern Warfare</li>
   <li>VALORANT</li>
   <li>Counter-Strike: Global Offensive</li>
   <li>Overwatch</li>
</ol>

Same deal, I used my web scraper to scrape over 100 thumbnails for each game and sorted them into appropriate directories for my pipeline. I adjusted the neurons to handle the new input format and did some hyperparameter tuning to find a good number of layers and epochs.

The model did pretty well and reached a slightly worse accuracy of around 85%. Still not bad, all things considered. But looking through my test data revealed some wildly inaccurate predictions.

Like usual, many of the new games were able to be predicted with high accuracy:

<img width=300px src="https://i.imgur.com/yatGHjI.png"></img>

<img width=300px src="https://i.imgur.com/AdNX3fh.png"></img>

<img width=300px src="https://i.imgur.com/9w3Ab2Y.png"></img>

However, the new games presented several more possibilities for error, some for games that looked similar, and some that just didn't make sense at all:

<img width=300px src="https://i.imgur.com/LcQpeFF.png"></img>

<img width=300px src="https://i.imgur.com/vrdiedw.png"></img>

<img width=300px src="https://i.imgur.com/77Pet5A.png"></img>

Additionally, this time I did not interfere with the data, and so images of lobbies and loading screens also factored into the model, for better or for worse:

<img width=300px src="https://i.imgur.com/LNZjlOv.png"></img>

<img width=300px src="https://i.imgur.com/BxUzfik.png"></img>

## Reality Check

# Chapter 1: League of Legends

## Not Object Detection, Template Matching
