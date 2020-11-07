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
   * [The Name of the Game](#the-name-of-the-game)
   * [Pyke Detection](#pyke-detection)
   * [Deus Ex Machina](#deus-ex-machina)<br /><br /><br /><br />

# Chapter 0: Preliminary Test of Viability

In order for this to have any future, we just have to make sure that a neural net is actually capable of working with screenshots as data. I have confidence in machine learning, but it's always better to be safe and build one of these simple models first.

By scraping thumbnails from Twitch livestreams, I assembled a small dataset of around 100-200 images of the following games:

<ul>
   <li>League of Legends</li>
   <li>VALORANT</li>
   <li>Counter-Strike: Global Offensive</li>
   <li>Overwatch</li>
</ul>

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

<ul>
   <li>Apex Legends</li>
   <li>Fall Guys</li>
   <li>Fortnite</li>
   <li>League of Legends</li>
   <li>Call of Duty: Modern Warfare</li>
   <li>VALORANT</li>
   <li>Counter-Strike: Global Offensive</li>
   <li>Overwatch</li>
</ul>

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

Needless to say, this was a good time to reevalute the scope and ambition of the project.

## Reality Check

So we were able to fit a model with pretty high accuracy, and 85% of the time you can give it a video game screenshot and it'll tell you what game you're playing.

What now?

The original idea was to have the model sit and watch Twitch live thumbnails, thus being able to sort streamers into their respective game categories without them having to select the game they were playing. But if we actually take a step back and think about it, there presents a bunch of issues, both logistical and practical:

<ol>
   <li>The model gets increasingly inaccurate the more games it has to account for. If it gets 15% wrong with only 8 games to choose from, how much more often will the softmax function pick the wrong game when there are 1,000 games to choose from?</li><br />
   
   
   
   <li>In order for the supervised model to be relatively accurate, it would need at least 100 instances of each game, including singleplayer games like God of War or Super Mario Odyssey of which there are no constant streamer or viewer bases for. It would be incredibly hard to efficiently find footage that would well represent these games.</li><br />
   
   
   
   <li>85% is not a bad accuracy, but the 15% of times where streamers would have to manually go back into their Twitch settings just to change an incorrect prediction would outweigh the benefits of such an extension. Which brings us to our final point:</li><br />
   
   
   
   <li>This does not really solve an issue. No one is complaining about the 5 seconds it takes to input the name of the game they're playing, and although there are some people who forget to change it and end up in the wrong category, this specific model and its purposes are not worth the effort it will take to build it.
</ol>

But this isn't the end of the project! This was just the beginning, a test to make sure that we were able to actually get things done in this context and space. The lab is open, and we have work to do.

# Chapter 1: League of Legends

I'm writing this up instead of developing, so that means unfortunately I haven't been progressing as smoothly as I'd hoped. We'll get to that part soon, though, so let's just jump right in for now.

League of Legends has been and is still the most viewed game categories on Twitch. It's also one of my favorite games and IPs. Now we pose the following challenge:

- Can we detect what champion a streamer is playing from the livestream preview alone?

There is currently a tag function for the League of Legends category that allows a streamer to manually add a tag to their stream, but nobody really uses it. After all, if you're streaming for 6-7 hours on end, it may become annoying to have to go back to your stream settings every game and change the champion tag. As you can see on the front page of LoL on Twitch, the tag is virtually obsolete:

<img src="https://i.imgur.com/XHdKuzs.png"></img>

But what if you didn't have to type in that tag, and Recon would just automatically be able to tell what champion you were playing just from the thumbnail? Anyone with the extension would be able to find you if they wanted to watch a specific champion.

Anyways, that's what we're working with. Technical stuff in the next section.

## The Name of the Game

In the prologue, we used image classification to separate game screenshots. This, however, is a pretty different task. Take for example, these two stream previews of League:

<img src="https://i.imgur.com/prhQ8E4.jpg"></img>

<img src="https://i.imgur.com/ZHRnQMT.jpg"></img>

If you have a good eye and you've played a good amount of League, you can probably recognize the characters being played as Alistar and Twitch, respectively. However, when we feed thousands and thousands of champion-labeled League images into a classifier, it won't be accurate. There is far too much noise in these screenshots for the neural net to be able to learn what actually gives away what champion is being played.

The answer to that lies in the HUD. In a MOBA like League, the player's screen can be anywhere on the map at any given time. The player character might not even visible. But the one thing that doesn't move as long as the player is in a game is the heads up display in the bottom center of the screen, shown below with annotations:

<img src="https://i.imgur.com/z8v5hFh.png"></img>

Some of these attributes are useful, some of these are not. Items can vary across the board, and most of the stat icons are shared by almost all the champions. Even the champion portrait actually depends on what skin the player is using. However, all of these assets are fixed images and can be used as a reference in one way or another:

<img src="https://i.imgur.com/eMmAAyS.png"></img>

These images and sprites are all available on Riot's dataset Data Dragon (ddragon). Our goal and attempt at a solution here is to try to find these images in the preview thumbnail image. Once we do that, we can find the champion the player is playing (e.g. if we find the sprite image for Dr. Mundo's W ability, Burning Agony, in the image, we can conclude that that the player is playing Dr. Mundo).

Normally the typical approach for "find an image within an image" would be **object detection**, a computer vision technique that is capable of things like finding a cat within a photograph. This requires training, just like an image classifier, and is a whole other field in itself. Luckily for us, we have a constraint in our problem that is going to be extremely helpful:

- The image we are looking for will never be in a different form.

What does this mean? Well, in a normal object classification problem like the cat one mentioned, the model has to learn from many, many cat images so that it can learn what a cat looks like, regardless of what direction it's facing and the size and type of cat it is. Whereas in our case, if we're checking if a thumbnail contains Pyke's W ability, Ghostwater Dive, we can count on it to look exactly the same every time, because it's a static image icon. Instead of training a model to detect different variations of an object, we just have to find the existence of one image inside of the other image.

This is called **template matching**, and it's the key technique we'll be using in this chapter.

<img src="https://i.imgur.com/L3S5hbR.png"></img>

We'll start work in the lab in the next section.

## Pyke Detection

## Deus Ex Machina
