REQUIREMENTS
------------
python >3.5 (used 3.7)
numpy
scipy
pandas
sklearn


INSTALLATION / ENVIRONMENT
--------------------------
I wrote and ran this code in PyCharm, which has a few quirks.

In terms of setting up dependencies, I had a better time with
py -m pip install --upgrade 
rather than
pip install
because my python setup is a bit garbage at the moment.

I had to run my code through PyCharm, rather than via terminal, but 
that's just one of those quirks; the code shouldn't rely on an IDE.

RUNNING -- D/R
--------------
The four dimensionality reduction algorithms can be run by running
the following files:

 - pca.py
 - ica.py
 - rp.py
 - lda.py

Clustering algorithms are run on all data sets using:

 - kmeans.py
 - em.py

The neural network training is done in:

 - NeuralNetwork.py



CODE
----
https://github.com/katharinebrinker/cs7641-p3


