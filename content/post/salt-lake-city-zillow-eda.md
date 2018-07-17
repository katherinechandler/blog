---
date: 2018-04-01T10:58:08-04:00
description: "An exploratory analysis of Salt Lake City's real estate data to address the eternal question"
title: "Living small in the city or large in the 'burbs?"
author: 'Katherine Chandler'
featured_image: '/images/houses.png'
---

I am a proponent of smaller homes, and recent dinner discussions have made me curious about the relationship between the size of a home and its relative appreciation in value. My vocal family members maintain that 'You're better off owning a tiny home downtown than a mansion in the 'burbs!!!'. Indeed, current hipster cannon asserts that bigger is not better when it comes to housing. While there are many non-economic merits of a modestly sized home, I was curious about how the 'small-home' movement holds up to economic reality. Is a small urban home really a better investment than a large suburban home? I recently moved back to my hometown of Salt Lake City, and in an entirely self-serving selection process I elected to address this question using real estate data for the Salt Lake City metro area.

### Questions for Exploration

This question was addressed as three discrete sub-questions:

* Are small homes more valuable (per square foot) than large homes?
* Is a small home downtown really more valuable than a large home in the 'burbs?
* Has the relative value of small urban homes increased more over time than that of large suburban homes?

### My Data

I used two different data sets in my exploration of these questions: one for active real estate listings and one for historical real estate trends. I generated the active listing data using a modified script ([```scrapehero's zillow.py```](https://gist.github.com/scrapehero/5f51f344d68cf2c022eb2d23a2f1cf95 "scraperhero")) to scrape Zillow listing data for Salt Lake City zip codes (see the full code in my [GitHub repo](https://github.com/katherinechandler/Zillow_Analysis_Salt_Lake_City 'GitHub')). The historical data is a timecourse series (April 1996 to February 2018) of Zillow Home Value Index (ZHVI, [ZHVI methodology here](https://www.zillow.com/research/zhvi-methodology-6032/ "ZHVI methodology")) published in the [Zillow Research Data sets](https://www.zillow.com/research/data/ "Zillow Research Data").


### Now, small house or large house? Which is it?!?

After an initial organization and exploration of the data, I dug into the small versus large comparisons.

#### Question 1. Are small homes more valuable (per square foot) than large homes?

This was a pretty straighforward question to answer. I defined three home size 'types' - small (less than 1200 sqft), medium (1200 - 2400 sqft), and large (more than 2400 sqft) - and calcuated the mean price per square foot for each home size type. 

<figure>
  <img src="/images/zillow_post_images/Project2_figures/Post1_Fig1.png" alt="Fig1" style='width:60%'>
  <figcaption>Figure 1. Price per Square Foot by Home Size.</figcaption>
</figure>

The mean price per square foot for a small home is $249, for a 'medium' sized home is $195, and for a 'large' home is $208. **Relative to their size, small homes are more valuable than large homes**. That was easy! Next I addressed geographic features in the small vs. large comparisons.

#### Question 2: Is a small home downtown really more valuable than a large home in the 'burbs?

To address this question I needed to create some geographic categories in the data. Each zip code was assigned to a neighborhood and one of four gegraphic regions: Central, East, South, and North-West. The regions category is a helpful way to understand the data if you're not familiar with Salt Lake City's neighborhoods. In this analysis the Central region is considered 'urban' and the other three regions are considered 'suburban'. See the [GitHub repo](https://github.com/katherinechandler/Zillow_Analysis_Salt_Lake_City 'GitHub')) for full explaination of the geographic classifications.

After categorizing the listings, the price per square foot for home size categories (small, medium, and large) were plotted for each neighborhood and region.

<figure>
  <img src="/images/zillow_post_images/Project2_figures/Post1_Fig2a.png" alt="Fig2" style='width:200%'>
  <figcaption>Figure 2a. Price per Square Foot by Size Category and Neighborhood.</figcaption>
  <img src="/images/zillow_post_images/Project2_figures/Post1_Fig2b.png" alt="Fig2" style='width:200%'>
  <figcaption>Figure 2b. Price per Square Foot by Size Category and Geographic Region.</figcaption>
</figure>


To strictly answer my question, **yes, a small home downtown is more valuable (relative to size) than a large house in the suburbs.** The median price per square foot for a small home in Central Salt Lake is $323 while large homes in East, North-West, and South Salt Lake City are priced at $222, $128, and $148, respectively. 

The price per square foot by neighborhood plot surprised me, however, in that a small home in Central Salt Lake is *not* more valuable (per square foot) than a small home in the fancy Eastern suburb of Cottonwood. In fact, the most valuable asset type in this data set is a small home in Cottonwood, with a median price per square foot of $364. This doesn't hold true for the other East-SLC neighborhoods nor for neighborhoods in the North-West and South suburbs.


#### Question 3: Has the relative value of small urban homes increased more over time than that of large suburban homes?

Exploration for this question made use of the Zillow Home Value Index (ZHVI) in the historic pricing timeseries data. This data set doesn't contain a direct measure of home size (i.e. sqft) but it does contain aggregate information about estimated value for a given bedroom count. I calculated a simple Pearson correlation (&rho; = 0.575) to confirm that the number of bedrooms and the size of the listing are indeed positively correlated. I determined that one and two bedroom properties usually fall below the 'small' category (median size is 759 and 1202 sqft, respectively), and homes with four or more bedrooms usually fall within the 'large' category cutoff of 2400+ sqft (median size is 2696 sqft). Thus, for this analysis 'small' homes will be classified as 1-2 bedrooms and 'large' will be classified as 4 or more bedrooms.

To compare price as ZHVI values for different classes of homes (small-downtown, large-suburban, etc), I needed to normalize the data to a contstant value. The absolute ZHVI of a small home downtown may be lower than that of a large suburban home, but I'm interested in how properties have appreciated over time. I normalized the ZHVI at each timestamp to the earliest ZHVI for that given 'home type' (number of bedrooms + zip code). The historic price data was processed using code in the `Zillow_SLC_Historic_Data` notebook in the [GitHub repo](https://github.com/katherinechandler/Zillow_Analysis_Salt_Lake_City 'GitHub').

The normalized Zillow Home Value Index (ZHVI) was plotted for small homes downtown versus large homes in each of the three suburban regions.

<figure>
  <img src="/images/zillow_post_images/Project2_figures/fig8.png" alt="Fig3" style='width:100%'>
  <figcaption>Figure 3. Historical Prices for Small Urban Homes and Large Suburban Homes.</figcaption>
</figure>

The relative value of small urban homes **has** appreciated more than that of large suburban homes, particularly in less desirable neighborhoods like North-West and South Salt Lake City. In fact, the 2018 data shows small homes at their highest valuation since 1996, while large suburban homes are appreciated to roughly the same valuation as during the 2006-2008 bubble. It is interesting to note that while all geographic regions and home size types exhibited rapid appreciation during the 'bubble', the magnitude of depreciation for each region/home type differed. To explore this observation, I created a 'volitility-index' to measure the percentage drop in price from the maximum price during the boom to the minimum price during the recession. 

<figure>
  <img src="/images/zillow_post_images/Project2_figures/depreciation.png" alt="Fig3" style='width:100%'>
  <img src="/images/zillow_post_images/Project2_figures/depreciation2.png" alt="Fig3" style='width:100%'>
  <figcaption>Figure 3. Percent Depreciation during the 2008 Economic 'Crash'.</figcaption>
</figure>

During the 2008-2010 market collapse, the value of small urban homes declined the least (-22.8%) and large suburban homes in the North-West region declined the most (-36.7% for 5+ bedrooms). Homes of all sizes in the Central region maintained value better than in other regions, with an average decline of just 25.0% compared to 29.6% in the East region, 35.7% in the North-West region, and 32.5% in the South region. 

So, is it better to own a small home downtown than a large home in the suburbs? The answer undoubtably depends on the specific suburb and the market trajectory, but in the context of Salt Lake City and the current economic context I will give this question a **'probably' yes**. Small urban homes are rapidly appreciating and are likely to maintain value well in the next downturn.

### Summary

* Are small homes more valuable (per square foot) than large homes?
    
    **Yes.**
    
    
* Is a small home downtown really more valuable than a large home in the 'burbs? 
    
    **Yes, but surprisingly a small home in a fancy suburb is worth more!**


* Has the relative value of small urban homes increased more rapidly than that of larger suburban homes?
    
    **Yep, and small homes are likely to maintain value better as well.**
    

These findings are mixed for the small house people. Small urban homes do hold their value well and in the current market are appreciating quickly, but for prospective buyers small homes come with a high price per square foot. I am satisfied that small urban homes are a stable investment, but I suppose I'll have to keep an open mind about Salt Lake City's eastern suburbs. 
    