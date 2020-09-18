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

***Log***

* [Preliminary Test of Viability](#preliminary-test-of-viability)
* [Reality Check](#reality-check)<br /><br /><br /><br />

## Preliminary Test of Viability

In order for this to have any future, we just have to make sure that a neural net is actually capable of working with screenshots as data. I have confidence in machine learning, but it's always better to be safe and build one of these simple models first.

By scraping thumbnails from Twitch livestreams, I assembled a small dataset of around 100-200 images of the following games:

<ol>
   <li>League of Legends</li>
   <li>VALORANT</li>
   <li>Counter-Strike: Global Offensive</li>
   <li>Overwatch</li>
</ol>

<img width=750px src="https://i.imgur.com/WUyjPFc.png"></img>

<img width=750px src="https://i.imgur.com/PZJdKGT.png"></img>

Then I just copied the layers and nodes from the architecture of a pretty basic Keras dogs-vs-cats classifier. Because of the low amount of training data, I ran it for 175 epochs: enough times so that the accuracy would substantially improve but not too many that the model would overfit. The results were not bad: 90% accuracy.

<img width=500px src="https://i.imgur.com/ezI34zF.png"></img>

Using another file to test completely new images on the model, I found that the model was actually pretty legitimate, being able to differentiate most pictures with high probability.


<img width=500px src="https://i.imgur.com/XR1qeYV.png"></img>

<img width=500px src="https://i.imgur.com/Ufbmce4.png"></img>

<img width=500px src="https://i.imgur.com/tYttHrJ.png"></img>

<img width=500px src="https://i.imgur.com/0nPJUUu.png"></img>

On most of the test images, at least.

<img width=500px src="https://i.imgur.com/J2qZeQ5.png"></img>
