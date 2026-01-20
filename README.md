# Brown Dwarfs, How do They Work?

Benjamin Pennell

November 27th, 2025

Max Planck Institut für Astronomie, Heidelberg

---

I want to learn about brown dwarfs (BDs). We have an abundance of data from *Gaia* and the angle here is to explore beyond the *full orbit solutions* by concocting a mixture model which incorporates both single stars and binaries at once, and learn the mass and period distributions by computing the likelihood of a particular model to recover the correct solution types for every object.

We will take every object between 100 and 200pc (`./setup/CatalogueCreation.ipynb`), trimmed to objects less than $0.4M_\odot$ with luminous companions cut out by seeing them in the XP spectra to only have companions $q<0.5$

We will precompute the solution type for every possible binary (`./setup/SetupCube.ipynb`, and running it through my seperate `GridGenerator` code), marginalised over the parameters I don't care about (sky positions, proper motions, orbit angles). We will use this for quick look-up at inference.

The plan will be to bin the secondary masses into ten equally sized mass bins.

I will generate three different cubes for different eccentricity models: circular, thermal, turnover.

We will generate synthetic datasets (`./GenereateSyntheticDatasets.ipynb`) to demonstrate the efficacy of the method. We will test the dependence on period, ecentricity, mass ratio, \# of objects, and the mass-binarity relation (`./DependencePeriod.ipynb, ./DependenceEccentricity.ipynb, ./DependenceMassRatio.ipynb, ./DependenceCounts.ipynb, ./DependenceBinarity.ipynb`)

Armed with this, we will apply many different models to the real dataset (`./RealData.ipynb`) to show the mass-binarity relation. Then, we will compare how with different maximum period cutoffs and mass ratio models, you can change the resulting total binarity (`./FractionMatching.ipynb`)


Scripts:
- `Sampler.py` takes in an input catalogue and model and can spit out the likelihood of different binary fractions. It can take the catalogue and convert between $q$ and $\lambda$ space for the inference.
- `SyntheticData.py` is a parallelised script to use `Gaiaimock` to generate synthetic data. The path to `Gaiamock` is to be configured in `config.json`
- `style.py` is used to set `Matplotlib.rcparams` to make a uniform plot style for the paper

Folders:
- `./data/` stores data for the sample and the precomputed grids
- `./plot_data/` is the storage location for the plot scripts to save for plotting later
- `./massive_data/` is for synthetic generated data. These files are too big for github, so a `.gitignore` ignores it