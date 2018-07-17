---
date: 2018-05-01T10:58:08-04:00
description: "A statistical approach"
title: "Exploring Salt Lake City's Real Estate Market"
author: 'Katherine Chandler'
featured_image: '/images/slc.png'
---

In the previous project I explored the 'bigger is not better' movement with an initial exploratory analysis of active and historic real estate data in the Salt Lake City metro area. I found that small homes in the urban center are more valuable per square foot than large suburban homes and maintain their value more robustly during times of market contraction. In this project, I used statistical inference and linear regression to validate these initial findings and explore the ways in which various factors impact home pricing.

### Questions for Exploration

-   Are the regional differences in price and size statistically significant?
-   Are comparable size listings in different regions significantly different in listing price?
-   Can we model home pricing based on size and geography characteristics?

### My Data

The data sources for this analysis were explained in detail in the previous post. See my [GitHub repo](https://github.com/katherinechandler/Zillow_Analysis_Salt_Lake_City 'GitHub')) for full code, data processing, and deeper analysis notes.

### Question 1. Are the regional differences in price and size statistically significant?

To address this question pairwise Tukey's Honest Significant Difference test (Tukey's HSD) tests were calculated for a given a set of geographic and numeric parameters. These tests establish a confidence interval to compare the means of all pairs within a given group and establish whether two groups are statistically different (i.e. 'reject' the null hypothesis that the groups are not different). The separation between two group's confidence intervals reflect the extent to which differences are statistically significant (greater separation means more significant).

<figure>
  <img src="/images/zillow_post_images/Project3_figures/fig4.png" alt="Fig1" style='width:75%'>
  <figcaption>Figure 1. Confidence intervals for median price, size, and price per square foot by region</figcaption>
</figure>

**Yes, the price and size distinctions are significant.** Central and East Salt Lake have distinctly different prices than South and North-West Salt Lake, all regions have distinctly different sizes, and all regions have distinctly different price per square foot values.

### Question 2: Are comparable size listings in different regions significantly different in listing price?

This question was addressed using a function to pair comparable listings (by bedroom number) for each combination of regions and perform a two-tailed t-test to identify statistical differences. The analysis output is not shown here (for brevity), but this analysis shows that there are many significant differences in the pricing of comparable homes between different regions. The most significant price difference in comparable listings was for 3-bedroom homes listed in the North-West region (median price $249,900) and the East region (median price $445,000.00), with a p-value of 3.316e-09.

This data provides a definitive **yes, many types of comparable listings in different regions are listed at significantly different prices.**

### Question 3: Can we model home pricing based on size and geography characteristics?

From the statistical analyses above we can infer that there is a relationship between listing price, home size, and home region. The correlation heatmaps below illustrate these relationships graphically. As expected, there is a positive relationship between price and size. There is also a positive relationship between price and being in the East of Central regions, and a negative relationship between price and being in North-West region.

<figure>
  <img src="/images/zillow_post_images/Project3_figures/fig5.png" alt="Fig1" style='width:200%'>
  <figcaption>Figure 2. Heatmap representations of the relationship between price and size and price and region</figcaption>
</figure>

Can we use these relationships to build a linear regression model for the home listing price? After performing feature selection I fit a model for price using the following parameters: baths, sqft, region\_Central, region\_East, region\_North\_West, and region\_South. Note that region was converted to a dummy variable for this analysis. This model for listing price using 'region' as a dummy variable had an *R*<sup>2</sup> of 0.633 and 7 variables (6 had p-values greater than 0.05; number of bathrooms was not significant).

<figure>
  <img src="/images/zillow_post_images/Project3_figures/fig6.png" alt="Fig1" style='width:200%'>
  <figcaption>Figure 3. Variable coefficients of linear regression model for price</figcaption>
</figure>

All other variables being equal, this model predicts that an additional bathroom in a home adds $20,580 to list price, each additional square foot adds $272 to list price, being in the Central region adds $180,100 to list price, being in the East region subtracts -$82,220 from list price, being in the North\_West region subtracts -$180,700 from list price, and being in the South region subtracts -$83,010 from list price. Based on previous exploration of the data, we know that being in the East region should have a net positive - not net negative- effect on home price. This indicates the limitation of a simple linear regression model for predicting home price with the given data, and suggests that either more robust data or a different model should be explored.

Additionally, diagnostic plots of the regression model (not shown) indicate the presence of outliers in the data as well as a non-normal distribution. The regression model using region as a variable did not perform significantly better than a simple model using only size, bedroom count, and bathroom count (R^2 is 0.638, data not shown). Despite the limitations in the model, however, it does provide some insight into relationships in the data and price predictions using this model were performed.

Summary
-------

-   Are the regional differences in price and size statistically significant?

    **Yes, the distinctions are significant and indicate distinct home 'types' in different geographic regions.**

-   Are comparable size listings in different regions significantly different in listing price?

    **Yes, size-comparable listings in different regions are listed at significantly different prices.**

-   Can we model home pricing based on size and geography characteristics?

    **Kind of... a linear regression model using size and geography variables isn't a great fit for the data, but it does provide some meaningful insights.**

These findings illustrate that there are significant difference in home type and price between Salt Lake City's different geographic regions, but that these variables alone are not sufficient to predict listing price. are mixed for the small house people. Small urban homes do hold their value well and in the current market are appreciating quickly, but for prospective buyers small homes come with a high price per square foot. I am satisfied that small urban homes are a stable investment, but I suppose I'll have to keep an open mind about Salt Lake City's eastern suburbs.

Limitations and Further Exploration
-----------------------------------

There are many limitations in the modeling portion of this exploration. The only model considered was a linear regression model, and it may be the case that a linear model simply doesn't describe real estate pricing well. Many difficult-to-quantify home features (like style and finish quality) affect home price, and the present data set does not reflect these metrics. The regression diagnostic plots reveal a non-normal distribution in the data, and other models types may provide better estimates of price. Additionally, these diagnostic plots reveal a number of outliers that were not explored in depth. Addressing these outliers may yield more robust linear models.
