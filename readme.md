## Installation:

- Clone the adapted MOA repository from [here](https://github.com/Waikato/moa)

- Build on windows: The easiest way is to use an IDE such as IntelliJ

- Build on linux:

  ```
  1. cd execmoa
  2. mkidr moa， mkdir moa/classifiers， mkdir moa/classifiers/trees
  3. javac -cp moa.jar HoeffdingTree.java
  4. mv *.class moa/classifiers/trees
  5. jar -uf moa.jar moa/classifiers/trees/*
  ```

  The code in execmoa comes from this [GitHub](https://github.com/chaitanya-m/kdd2018) 

## Datasets

### Synthetic data

Synthetic data is all generated using the API proposed by MOA.

**SEA**: '-i 5000000 -f 10000 -s (generators.SEAGenerator -b)',

**LED**:  '-i 1000000 -f 1000 -s (ConceptDriftStream -s (generators.LEDGeneratorDrift -d 1)   -d (ConceptDriftStream -s (generators.LEDGeneratorDrift -d 3) -d (ConceptDriftStream -s (generators.LEDGeneratorDrift -d 5)  -d (generators.LEDGeneratorDrift -d 7) -w 50000 -p 250000 ) -w 50000 -p 250000 ) -w 50000 -p 250000)'

**AGR**:  '-i 10000000 -f 10000 -s (ConceptDriftStream -s (generators.AgrawalGenerator -f 1 -b) -d (ConceptDriftStream -s (generators.AgrawalGenerator -f 2 -b) -d (ConceptDriftStream -s (generators.AgrawalGenerator -b)   -d (generators.AgrawalGenerator -f 4 -b) -w 500 -p 2500000) -w 500 -p 2500000 ) -w 500 -p 2500000)',

**RBF**: '-i 2000000 -f 10000 -s (generators.RandomRBFGenerator -c 15 -a 200 -n 100) ',

**RTG**: '-i 10000000 -f 10000 -s (generators.RandomTreeGenerator -c 25 -o 200 -u 200)',

### Real-word data

**[Covertype.](https://moa.cms.waikato.ac.nz/datasets/)** The forest covertype data set represents forest cover type for 30 x 30 meter cells obtained from the US Forest Service Region 2 Resource Information System (RIS) data.

**[KDD99](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html).** KDD99 dataset corresponds to a cyber-attack de- tection problem (i.e., attack or common access), an inherent streaming scenario since instances are sequentially presented as a time series.

**[MNIST8M.]()** MNIST8M is the augmentation of original MNIST database by using pseudo- random deformations and translations.

## Code

We have added the IMAC method to the code provided by this [github](https://github.com/ICDM2018Submission/VFDT-split-time-prediction).

There are four algorithms:

- [VFDT](https://homes.cs.washington.edu/~pedrod/papers/kdd00.pdf) algorithm, which periodic executes split-attempts. 
- [OSM](https://www.techfak.uni-bielefeld.de/~hwersing/LosingHammerWersing_ICDM2018.pdf) algorithm, which predicts the interval of split-attempts in VFDT by assuming that heuristic measure of the second-best attribute does not change and the  the heuristic measure of  best attribute increases as much as possible as the data arrives.
- [CGD](https://pdfs.semanticscholar.org/96a4/3c8607a4311a3ef37d48cab5d5396f50d9d3.pdf) algorithm,  which predicts the interval of split-attempts in VFDT by assuming that the difference of best attribute and second-best attribute does not change.
- IMAC algorithm, which determines the potential split timing in VFDT with incremental information.

```
trees/HoeffdingTree.java        // Clean implementation
trees/HoeffdingTreeMeasure.java // can evaluate split-delay and split-attempts

run.py // evaluate VFDT, OSM, CGD and IMAC
run_measure.py // evaluate split-delay and split-attempts of VFDT, OSM, CGD and IMAC
```

