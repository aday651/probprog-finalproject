# Final Project Repository

## Summary of project

The goal of the project is to try and perform accurate uncertainity quantification for 
detecting whether users are trustworthy or not on a Bitcoin OTC platform.

## Data source

The data comes from ratings between users on a Bitcoin OTC platform, and was obtained from 
the following [webpage](https://cs.stanford.edu/%7Esrijan/rev2/), which contains all
the ratings between users in the network, and also some ground truth data. A full description
is given in the project writeup, with a shorter description available on the 
[SNAP website](http://snap.stanford.edu/data/soc-sign-bitcoin-otc.html).

## Requirements file

Note: I use Conda for handling packages. I therefore ran the following commands to obtain
requirements files:

```
conda list -e > requirements_conda.txt
conda env export > ppp.yml
pip freeze > requirements_pip.txt
```

All three are listed in the root of the directory. Please let me know if you get into
any problems.