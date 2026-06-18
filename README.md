# N-Elicit

## Overview

This folder contains code that creates a poll for expert elicitation of the parameters of the [Drake equation](https://en.wikipedia.org/wiki/Drake_equation) using distributions of estimates (not one-point estimates), obtaining a distribution (not a point estimate), and obtaining a distribution of the possible number of detectable alien civilizations in our Galaxy.

Check out [here](https://github.com/ferdesmello/drake-metaculus) a precursor version of this project that uses [Metaculus](https://www.metaculus.com/home/) data.

## Repository structure

<pre>
n-elicit/
├── index.html
├── intro.html
├── consent.html
├── survey_question_page.html
├── results.html
├── questions.json
├── README.md  
├── i18n/
│   ├── en.json
│   ├── es.json
│   └── pt.json
├── data/
└── backend/
    ├── main.py
    ├── requirements.txt
    ├── README.md
    └── database.py
</pre>

## What the code does


## The Drake equation

The original Drake equation had many variations over time. Carl Sagan's version changes the first (star formation rate) and last (how long civilizations release detectable signal) parameters, and it is the one we use here.

$N = N_{*}\cdot f_{p}\cdot n_{e}\cdot f_{l}\cdot f_{i}\cdot f_{c}\cdot f_{L}$

where

$N$ = the number of civilizations in the Milky Way galaxy with which communication might be possible (i.e., which are on the current past light cone);

and

$N_{∗}$ = the number of stars in our Galaxy.

$f_{p}$ = the fraction of those stars that have planets.

$n_{e}$ = the average number of planets that can potentially support life per star that has planets.

$f_{l}$ = the fraction of planets that could support life that actually develop life at some point.

$f_{i}$ = the fraction of planets with life that go on to develop intelligent life (civilizations).

$f_{c}$ = the fraction of civilizations that develop a technology that releases detectable signs of their existence into space.

$f_{L}$ = the fraction of a star's lifetime for which such civilizations release detectable signals into space.

The Drake equation is usually computed using point estimates for every parameter. However, we are not certain of the value of many parameters, with estimates varying by many orders of magnitude. Following Sandberg, Drexler, and Toby Ord's [Dissolving the Fermi Paradox](https://arxiv.org/abs/1806.02404), it seems better to use probability density functions and get a distribution for the possible values of $N$.

## Instructions

<img width="580" height="338" alt="demonstration" src="https://github.com/ferdesmello/n-elicit/blob/main/data/slider_demo.gif" />
