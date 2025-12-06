# Brown Dwarfs, How do They Work?

Benjamin Pennell

November 27th, 2025

Max Planck Institut für Astronomie, Heidelberg

---

I want to learn about brown dwarfs (BDs). We have an abundance of data from *Gaia* and the angle here is to explore beyond the *full orbit solutions* by concocting a mixture model which incorporates both single stars and binaries at once, and learn the mass and period distributions by computing the likelihood of a particular model to recover the correct solution types for every object.

We will take every object between 100 and 400pc, trimmed to objects less than $0.4M_\odot$ with luminous companions cut out by seeing them in the XP spectra to only have companions $q<0.5$

We will precompute the solution type for every possible binary, marginalised over the parameters I don't care about (sky positions, proper motions, orbit angles).

We will use this for quick look-up at inference while running markov chains

The plan will be to bin the secondary masses into say, 5 bins. Additionally, the period into 5 bins. 

I will generate three different cubes for different eccentricity models: circular, thermal, turnover. The hope is that the results will be very similar across distributions so that it's robust in that sense.