<img width=500px src="https://i.imgur.com/50XEGn7.png"></img>
#

Presented by <strong>TokyoExpress</strong><br /><br />

This project is on hold, see github link for its successor.

***Abstract***

Recon for Twitch is a set of experiments and potential applications revolving around computer vision of video game screenshots.<br /><br />

***Premise***

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
   * [Deus Ex Machina](#deus-ex-machina)
   * [Bounding Boxes](#bounding-boxes)<br /><br /><br /><br />

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

Alright, I don't want to get too behind in the logs, so let's get this section over with.

For our preliminary test, we're going to see if we can detect a single character as either being played or not being played. I'm using my current main champion Pyke. Pyke has 4 skins, 4 abilities, and 1 passive, for a total of 9 images that can be used to recognize Pyke as the character being played in a screenshot. There are some constraints for us, though. When any champion uses an ability, it goes on *cooldown*, and the icon is hidden behind a timer until the spell is able to be used again. Additionally, when a player dies, the champion portait turns grayscale and has a similar countdown timer on it until the player respawns.

Which leaves the passive icon as the only thing that can be relied on to actually be fully visible in a screenshot at any point in the game. Now, some champions have passives that trigger and have cooldowns as well, but luckily Pyke is not one of them. So we'll be using template matching on the passive icon to do our testing.

I set up a simple Python script to do iterative template matching over a series of sizes for the smaller image. Basically, template matching requires you to specify the size of the image that you'll be looking for, which isn't ideal because we don't actually know how big the streamer's HUD is going to be, but we can work around that by template matching multiple times with different icon sizes. And it worked! The script was able to find the passive icon in the screenshot of Pyke (outlined in red):

<img src="https://i.imgur.com/LiGigbR.png"></img>

Not bad, but also not incredibly impressive. The way template matching works is that it computes a score for every possible location of the smaller image in the larger image, so it can't actually directly tell us if the passive icon *isn't* in the picture. For that, we have to use thresholding: essentially we have to determine a value that the template match has to "pass", otherwise we consider it not found.

This is where it got a little tricky. Because the resolutions of the thumbnails we scraped from Twitch were so low (440 x 248 pixels), the difference between a successful match and an unsuccessful match was not a fine line at all. We're talking 8 x 8 images for a total of 64 pixels, where 53 matching pixels could be a true positive in one image, while 56 matching pixels could be a false positive in another. I computed the bare minimum value for passing all of my sample Pyke images, and used that as a threshold to test the template matching on one of my small League thumbnail sets. The results were not so hot (all of the following are false positives):

<img src="https://i.imgur.com/FhI8xuU.png"></img>
<img src="https://i.imgur.com/lfid0Qq.png"></img>
<img src="https://i.imgur.com/37W8lTT.png"></img>
<img src="https://i.imgur.com/qZ49RpD.png"></img>
<img src="https://i.imgur.com/1USWXT2.png"></img>

I've only included five of these, but I ran into about 100 when I was iterating through my dataset (which was only around 150 images). So this clearly didn't work. The resolution was just too small and the noise in the downscaled pixels varied too much for us to be able to find an accurate threshold. It seemed that we would either have to find a different approach or just abandon the method altogether, because if we weren't able to detect Pyke, there was no way we were going to be able to detect anything else.

Of course, we did find a solution to the problem (and it was really simple). And now that we're caught up, I can start developing again! See you in the next section.

## Deus Ex Machina

I'm sitting here downloading every image passive in the game into a folder, so I might as well give an update. Working on solo projects is a little difficult, especially when you have no idea what you're doing and have to somehow motivate yourself to continue.

A while back, I figured out that you could modify the URL of a Twitch thumbnail image to make it whatever resolution you wanted it to be. So I've begun downloading my samples as 1980 x 1116 pixels, which should give the template matching ample room to determine whether or not a given image actually exists in the screenshot or not. However, doing this requires a good amount of threshold testing that I have not gotten around to, simply because I haven't really found a good way to go about it. Also, I've been looking into other methods of doing this (of which there are actually a lot) and I want to see if I can solve this problem using bounding boxes, edge detection, and simple euclidean distance minimization.

So, template matching is on hold for now, which means we're also moving forward from Pyke Detection to Every Champion Detection. I'll show you the workflow in the next section, and apologies for the poor organization and structure of this documentation. I am, after all, just writing this as I go.

## Bounding Boxes

We've gotten to the point where I'd rather actually work on the project than write these up, which is pretty good. It means that I actually think that I can get stuff done and advance the project further. However, developer documentation is indeed half the fun sometimes, so I cannot neglect it.

I've built something that can correctly identify the champion being played in the screenshot around, let's say, 75% of the time. Maybe 70%. Here's basically how it works:

We know that if the character is not dead and is not one of the 10-something champions with passive cooldowns, then the passive will for sure be visible as a box in the bottom left corner of the screen. I went into my own game and tested out the maximum and minimum HUD size in order to obtain the coordinates for a rectangle that contained all the possible locations of the passive box.

I then extract this region from the image and apply a sharpen kernel on it, which accentuates the contrast between the box and the background:

<img src="https://i.imgur.com/m0eQDHj.png"></img>

Next, simple Canny edge detection and morphological operations. Then we simply select the shape that most resembles a box and is of appropriate size:

<img src="https://i.imgur.com/XwzTL90.png"></img>

We can then extract the box from the image and check the Euclidean distance against a collection of all passive images. The minimum distance should be the most similar image, and hopefully:

<img src="https://i.imgur.com/rhnU6EU.png"></img>
<img src="https://i.imgur.com/Lsm6VtZ.png"></img>

the champion is then found. Rammus' passive is recognized in the above example.

Of course, this isn't a perfect solution. It doesn't work if the passive is on cooldown or obscured:

<img src="https://i.imgur.com/pPX0awf.png"></img>

or if you are playing Aphelios, who doesn't even have any boxes:

<img src="https://i.imgur.com/A5jKRZZ.png"></img>

So we've still got a ways to go. I'm thinking that we can possibly use the coordinates of the passive box to obtain the coordinates of the other abilities and thus have more opportunities to check for images (an idea generously provided by my friend Kitnips). We can also go back to template matching and see if thresholding would be a more smooth solution. Anyways, this thing works for a good chunk of Twitch thumbnails, so I now have to think about how to wrap around these edge cases. Something about time and ignoring unclassifiable images and defaulting to the previously seen champion or something.

See you around.
