/*
 *    VFADTMeasure.java
 *    Copyright (C) 2007 University of Waikato, Hamilton, New Zealand
 *    @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 */
package moa.classifiers.trees;

import com.github.javacliparser.*;
import com.yahoo.labs.samoa.instances.Instance;
import moa.AbstractMOAObject;
import moa.capabilities.CapabilitiesHandler;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.bayes.NaiveBayes;
import moa.classifiers.core.AttributeSplitSuggestion;
import moa.classifiers.core.attributeclassobservers.AttributeClassObserver;
import moa.classifiers.core.attributeclassobservers.DiscreteAttributeClassObserver;
import moa.classifiers.core.attributeclassobservers.NullAttributeClassObserver;
import moa.classifiers.core.attributeclassobservers.NumericAttributeClassObserver;
import moa.classifiers.core.conditionaltests.InstanceConditionalTest;
import moa.classifiers.core.conditionaltests.NumericAttributeBinaryTest;
import moa.classifiers.core.splitcriteria.GiniSplitCriterion;
import moa.classifiers.core.splitcriteria.InfoGainSplitCriterion;
import moa.classifiers.core.splitcriteria.SplitCriterion;
import moa.core.StringUtils;
import moa.core.*;
import moa.options.ClassOption;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintWriter;
import java.util.*;

/**
 * Hoeffding Tree or VFDT.
 *
 * A Hoeffding tree is an incremental, anytime decision tree induction algorithm
 * that is capable of learning from massive data streams, assuming that the
 * distribution generating examples does not change over time. Hoeffding trees
 * exploit the fact that a small sample can often be enough to choose an optimal
 * splitting attribute. This idea is supported mathematically by the Hoeffding
 * bound, which quantiﬁes the number of observations (in our case, examples)
 * needed to estimate some statistics within a prescribed precision (in our
 * case, the goodness of an attribute).</p> <p>A theoretically appealing feature
 * of Hoeffding Trees not shared by other incremental decision tree learners is
 * that it has sound guarantees of performance. Using the Hoeffding bound one
 * can show that its output is asymptotically nearly identical to that of a
 * non-incremental learner using inﬁnitely many examples. See for details:</p>
 *
 * <p>G. Hulten, L. Spencer, and P. Domingos. Mining time-changing data streams.
 * In KDD’01, pages 97–106, San Francisco, CA, 2001. ACM Press.</p>
 *
 * <p>Parameters:</p> <ul> <li> -m : Maximum memory consumed by the tree</li>
 * <li> -n : Numeric estimator to use : <ul> <li>Gaussian approximation
 * evaluating 10 splitpoints</li> <li>Gaussian approximation evaluating 100
 * splitpoints</li> <li>Greenwald-Khanna quantile summary with 10 tuples</li>
 * <li>Greenwald-Khanna quantile summary with 100 tuples</li>
 * <li>Greenwald-Khanna quantile summary with 1000 tuples</li> <li>VFML method
 * with 10 bins</li> <li>VFML method with 100 bins</li> <li>VFML method with
 * 1000 bins</li> <li>Exhaustive binary tree</li> </ul> </li> <li> -e : How many
 * instances between memory consumption checks</li> <li> -g : The number of
 * instances a leaf should observe between split attempts</li> <li> -s : Split
 * criterion to use. Example : InfoGainSplitCriterion</li> <li> -c : The
 * allowable error in split decision, values closer to 0 will take longer to
 * decide</li> <li> -t : Threshold below which a split will be forced to break
 * ties</li> <li> -b : Only allow binary splits</li> <li> -z : Stop growing as
 * soon as memory limit is hit</li> <li> -r : Disable poor attributes</li> <li>
 * -p : Disable pre-pruning</li>
 *  <li> -l : Leaf prediction to use: MajorityClass (MC), Naive Bayes (NB) or NaiveBayes
 * adaptive (NBAdaptive).</li>
 *  <li> -q : The number of instances a leaf should observe before
 * permitting Naive Bayes</li>
 * </ul>
 *
 * @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 * @version $Revision: 7 $
 */


public class HoeffdingTreeMeasure extends AbstractClassifier implements MultiClassClassifier,
        CapabilitiesHandler {

    private static final long serialVersionUID = 1L;
    private static final int MAX_STEPS = 100000;
    private static final int MEASURE_GRACE_PERIOD = 1;
    private static final int INITIAL_GRACE_PERIOD = 100;
    private static int brentTotalIterations = 0;
    private static int brentSearches = 0;

    @Override
    public String getPurposeString() {
        return "VFADTMeasure.";
    }

    public IntOption maxByteSizeOption = new IntOption("maxByteSize", 'm',
            "Maximum memory consumed by the tree.", 33554432, 0,
            Integer.MAX_VALUE);

    /*
     * public MultiChoiceOption numericEstimatorOption = new MultiChoiceOption(
     * "numericEstimator", 'n', "Numeric estimator to use.", new String[]{
     * "GAUSS10", "GAUSS100", "GK10", "GK100", "GK1000", "VFML10", "VFML100",
     * "VFML1000", "BINTREE"}, new String[]{ "Gaussian approximation evaluating
     * 10 splitpoints", "Gaussian approximation evaluating 100 splitpoints",
     * "Greenwald-Khanna quantile summary with 10 tuples", "Greenwald-Khanna
     * quantile summary with 100 tuples", "Greenwald-Khanna quantile summary
     * with 1000 tuples", "VFML method with 10 bins", "VFML method with 100
     * bins", "VFML method with 1000 bins", "Exhaustive binary tree"}, 0);
     */
    public ClassOption numericEstimatorOption = new ClassOption("numericEstimator",
            'n', "Numeric estimator to use.", NumericAttributeClassObserver.class,
            "GaussianNumericAttributeClassObserver");

    public ClassOption nominalEstimatorOption = new ClassOption("nominalEstimator",
            'd', "Nominal estimator to use.", DiscreteAttributeClassObserver.class,
            "NominalAttributeClassObserver");

    public IntOption memoryEstimatePeriodOption = new IntOption(
            "memoryEstimatePeriod", 'e',
            "How many instances between memory consumption checks.", 1000000,
            0, Integer.MAX_VALUE);

    /** the parameter grace period will have two meanings here:
     1. If the parameter is used in IMA, the minVal should be set 100, maxVal should be set 1000
     The candidate attribute set is not very accurate with a samll grace period.
     2. If the parameter is used in VFDT/OSM/CGD, the minVal should be set 0, maxVal should be set Integer.MAX_VALUE
     **/
    public IntOption gracePeriodOption = new IntOption(
            "gracePeriod",
            'g',
            "The number of instances a leaf should observe between split attempts " +
                    "or the minimum number of instances a leaf shound observe before splits attempts",
            200, 0, Integer.MAX_VALUE);

    public FlagOption measureSplitOffsetOption = new FlagOption("measureOffset", 'M',
            "Measure split offset.");

    public StringOption savePathOption = new StringOption("savePathPrefix", 'o',
            "savePathPrefix.", "");

    public ClassOption splitCriterionOption = new ClassOption("splitCriterion",
            's', "Split criterion to use.", SplitCriterion.class,
            "InfoGainSplitCriterion");

    public FloatOption splitConfidenceOption = new FloatOption(
            "splitConfidence",
            'c',
            "The allowable error in split decision, values closer to 0 will take longer to decide.",
            0.0000001, 0.0, 1.0);

    public FloatOption tieThresholdOption = new FloatOption("tieThreshold",
            't', "Threshold below which a split will be forced to break ties.",
            0.05, 0.0, 1.0);

    public FlagOption binarySplitsOption = new FlagOption("binarySplits", 'b',
            "Only allow binary splits.");

    public FlagOption stopMemManagementOption = new FlagOption(
            "stopMemManagement", 'z',
            "Stop growing as soon as memory limit is hit.");

    public FlagOption removePoorAttsOption = new FlagOption("removePoorAtts",
            'r', "Disable poor attributes.");

    public FlagOption noPrePruneOption = new FlagOption("noPrePrune", 'p',
            "Disable pre-pruning.");

    public double trainOnInstanceTime = 0.;
    public double attemptToSplitTime = 0;
    public double voteOnInstanceTime = 0.;
    public double topAttributeAttemptTime = 0;
    public int attempts = 0;
    public int boundSplits = 0;
    public int maxSplits = 0;
    public int trainStepCount = 0;
    public DoubleVector boundSplitErrors = new DoubleVector();
    public DoubleVector boundSplitNumSamples = new DoubleVector();
    public DoubleVector tieSplitNumSamples = new DoubleVector();
    public DoubleVector maxSplitErrors = new DoubleVector();
    private Random rand = new Random();

    public double incrementAttemptTime = 0;
    public double SplitNodeMoreTime = 0;
    public double ActiveNodeMoreTime = 0;
    public double InactiveNodeMoreTime = 0;
    public double updateAttributeMoreTime = 0;
    public double calAttributeMoreTime = 0;
    public double setAttributeMoreTime = 0;
    public double trainFitTime = 0;
    public double trainMainTime = 0;
    public double filterLeafTime = 0;
    public double estimateModelTime = 0;
    public double enforeModelTime = 0;
    public double limitAttempt = 0;
    public double hoeffdingJudge = 0;
    public double learnMoreTime = 0;
    public double checkNumeric = 0;
    public double checkOrder = 0;
    public double checkOrderRight = 0;
    public static class FoundNode {

        public Node node;

        public SplitNode parent;

        public int parentBranch;

        public FoundNode(Node node, SplitNode parent, int parentBranch) {
            this.node = node;
            this.parent = parent;
            this.parentBranch = parentBranch;
        }
    }

    public static class Node extends AbstractMOAObject {

        private static final long serialVersionUID = 1L;

        protected DoubleVector observedClassDistribution;

        public Node(double[] classObservations) {
            //this.observedClassDistribution = new DoubleVector(classObservations);
            this.observedClassDistribution = new DoubleVector();
        }

        public int calcByteSize() {
            return (int) (SizeOf.sizeOf(this) + SizeOf.fullSizeOf(this.observedClassDistribution));
        }

        public int calcByteSizeIncludingSubtree() {
            return calcByteSize();
        }

        public boolean isLeaf() {
            return true;
        }

        public FoundNode filterInstanceToLeaf(Instance inst, SplitNode parent,
                                              int parentBranch) {
            return new FoundNode(this, parent, parentBranch);
        }

        public double[] getObservedClassDistribution() {
            return this.observedClassDistribution.getArrayCopy();
        }

        public double[] getClassVotes(Instance inst, HoeffdingTreeMeasure ht) {
            return this.observedClassDistribution.getArrayCopy();
        }

        public boolean observedClassDistributionIsPure() {
            return this.observedClassDistribution.numNonZeroEntries() < 2;
        }

        public void describeSubtree(HoeffdingTreeMeasure ht, StringBuilder out,
                                    int indent) {
            StringUtils.appendIndented(out, indent, "Leaf ");
            out.append(ht.getClassNameString());
            out.append(" = ");
            out.append(ht.getClassLabelString(this.observedClassDistribution.maxIndex()));
            out.append(" weights: ");
            this.observedClassDistribution.getSingleLineDescription(out,
                    ht.treeRoot.observedClassDistribution.numValues());
            StringUtils.appendNewline(out);
        }

        public int subtreeDepth() {
            return 0;
        }

        public double calculatePromise() {
            double totalSeen = this.observedClassDistribution.sumOfValues();
            return totalSeen > 0.0 ? (totalSeen - this.observedClassDistribution.getValue(this.observedClassDistribution.maxIndex()))
                    : 0.0;
        }

        @Override
        public void getDescription(StringBuilder sb, int indent) {
            describeSubtree(null, sb, indent);
        }
    }

    public static class CandidateAttributeSplitSuggestion extends AbstractMOAObject implements
            Comparable<CandidateAttributeSplitSuggestion>{
        public SplitCriterion splitCriterion;
        public InstanceConditionalTest splitTest;
        public double merit;
        public AutoExpandVector<DoubleVector>  resultClassDistributeionsVector;
        public double totalWeight;

        public int numSplits() {
            return this.resultClassDistributeionsVector.size();
        }

        public CandidateAttributeSplitSuggestion(AttributeSplitSuggestion suggestion, SplitCriterion criterion, double newMerit) {
            this.splitTest = suggestion.splitTest;
            this.merit = newMerit;
            this.splitCriterion = criterion;
            this.resultClassDistributeionsVector = new AutoExpandVector<DoubleVector>();
            this.totalWeight = 0;
            for (int i = 0; i < suggestion.resultingClassDistributions.length; i++) {
                this.resultClassDistributeionsVector.set(i, new DoubleVector(suggestion.resultingClassDistributions[i].clone()));
                for (int j = 0; j < suggestion.resultingClassDistributions[i].length; j++) {
                    this.totalWeight += suggestion.resultingClassDistributions[i][j];
                }
            }
        }

        public CandidateAttributeSplitSuggestion(AttributeSplitSuggestion suggestion, SplitCriterion criterion) {
            this.splitTest = suggestion.splitTest;
            this.merit = suggestion.merit;
            this.splitCriterion = criterion;
            this.resultClassDistributeionsVector = new AutoExpandVector<DoubleVector>();

            for (int i = 0; i < suggestion.resultingClassDistributions.length; i++) {
                this.resultClassDistributeionsVector.set(i, new DoubleVector(suggestion.resultingClassDistributions[i].clone()));
                for (int j = 0; j < suggestion.resultingClassDistributions[i].length; j++) {
                    this.totalWeight += suggestion.resultingClassDistributions[i][j];
                }
            }
        }

        public void updateCandidateAttributeInfo(Instance ins, HoeffdingTreeMeasure ht) {
            double startTime = System.currentTimeMillis();

            int childIndex = 0;
            if (this.splitTest != null) {
                childIndex = this.splitTest.branchForInstance(ins);
            }

            if (childIndex < 0)
                return;

            if (this.resultClassDistributeionsVector.get(childIndex) == null) {
                this.resultClassDistributeionsVector.add(childIndex, new DoubleVector());
            }
            int classValue = (int) ins.classValue();

            int interval = ht.incrementPeriodOption.getValue();
            double weight = ins.weight() * interval;
            double n = this.totalWeight;

            double n_jm = Math.max(1, this.resultClassDistributeionsVector.get(childIndex).getValue(classValue));
            double n_j = Math.max(1, this.resultClassDistributeionsVector.get(childIndex).sumOfValues());

            double add = 0;
            if (this.splitCriterion instanceof InfoGainSplitCriterion) {
                add = n_j * Utils.log2(n_j / (n_j + weight)) + n_jm * Utils.log2((n_jm + weight) / n_jm)
                        + weight * Utils.log2((n_jm + weight) / (n_j + weight));
            }
            else if (this.splitCriterion instanceof GiniSplitCriterion) {
                double n_j_power = 0.0;
                for (double element : this.resultClassDistributeionsVector.get(childIndex).getArrayRef()) {
                    n_j_power -= weight*(element * element);
                }
                add = n_j_power / ((n_j + weight) * n_j) + (2 *weight*n_jm + weight*weight) / (n_j + weight);
            }
            this.merit = this.merit * (n / (n + weight)) + add / (n + weight);

            this.resultClassDistributeionsVector.get(childIndex).addToValue(classValue, weight);
            this.totalWeight += weight;

            ht.updateAttributeMoreTime += System.currentTimeMillis() - startTime;
        }

        @Override
        public int compareTo(CandidateAttributeSplitSuggestion comp) {
            return Double.compare(this.merit, comp.merit);
        }

        @Override
        public void getDescription(StringBuilder sb, int indent) {
            // TODO Auto-generated method stub
        }
    }

    public static class Tuple implements
            Comparable<Tuple> {
        public int attr_idx;
        public int pos;
        public double merit;

        Tuple(int attr_idx, int pos, double merit) {
            this.attr_idx = attr_idx;
            this.pos = pos;
            this.merit = merit;
        }

        @Override
        public int compareTo(Tuple o) {
            return Double.compare(this.merit, o.merit);
        }
    }

    public static class LearningNode extends Node{
        private static final long serialVersionUID = 1L;

        private static final int MIN_CANDIDATE_ATTRIBUTE = 5;
        private static final int MAX_CANDIDATE_ATTRIBUTE = 10;
        private static final double MAX_CANDIDATE_ATTRIBUTE_RATE = 0.1;

        protected double weightSeenAtLastSplitEvaluation;
        protected AutoExpandVector<AttributeClassObserver> attributeObservers = new AutoExpandVector<AttributeClassObserver>();
        protected boolean isInitialized;

        // The following three parameters are used in evaluation mode
        public double measureSplitPostion = -1; // the split point of VFDT
        public boolean isMeasureFinish = false; // Whether the node is evaluated
        public int measureSplitResult = 0; // split type

        // just for debug
        public AttributeSplitSuggestion[] preSuggestion;
        public double measureSplitDiffTimes;
        // just for debug

        protected CandidateAttributeSplitSuggestion[] candidateAttributeSplitSuggestions;
        protected PriorityQueue<Tuple> potentialAttribute = new PriorityQueue<>(new Comparator<Tuple>() {
            @Override
            public int compare(Tuple o1, Tuple o2) {
                return Double.compare(o2.merit, o1.merit);
            }
        });
        protected  List<Integer> illegalAttribute = new ArrayList<>();
        protected double lastIncrementTime = 0;
        public double updateNumericalTime = 0;
        public boolean updateCandidateAttribute = false;

        public LearningNode(double[] initialClassObservations) {
            super(initialClassObservations);
            this.measureSplitPostion = -1;
            this.isMeasureFinish = false;
            this.measureSplitResult = 0;
            this.isInitialized = false;
        }

        public LearningNode(double[] initialClassObservations, CandidateAttributeSplitSuggestion[] candidateAttributeSplitSuggestions,
                            double measureSplitPostion, boolean isMeasureFinish, int measureSplitResult) {
            super(initialClassObservations);
            this.candidateAttributeSplitSuggestions = candidateAttributeSplitSuggestions;
            this.measureSplitPostion = measureSplitPostion;
            this.isMeasureFinish = isMeasureFinish;
            this.measureSplitResult = measureSplitResult;
            this.weightSeenAtLastSplitEvaluation = getWeightSeen();
        }

        public void clearTopAttributeInfo() {
            this.candidateAttributeSplitSuggestions = null;
            this.isMeasureFinish = true;
        }

        // --just for debug
        public double getMeritOfSplit(double[][] postSplitDists) {
            double totalWeight = 0.0;
            double[] distWeights = new double[postSplitDists.length];
            for (int i = 0; i < postSplitDists.length; i++) {
                distWeights[i] = Utils.sum(postSplitDists[i]);
                totalWeight += distWeights[i];
            }
            totalWeight = Math.max(totalWeight, 1);
            double gini = 0.0;
            for (int i = 0; i < postSplitDists.length; i++) {
                gini += (distWeights[i] / totalWeight)
                        * GiniSplitCriterion.computeGini(postSplitDists[i], distWeights[i]);
            }
            return 1.0 - gini;
        }
        // --just for debug

        public void increamentForTopAttribute(Instance inst, HoeffdingTreeMeasure ht) {
            double startTime = System.currentTimeMillis();

            if (this.isIncreaseMode()) {
                int interval = ht.incrementPeriodOption.getValue();
                if (getWeightSeen() - this.lastIncrementTime < interval)
                    return;
                this.lastIncrementTime = getWeightSeen();
                this.updateCandidateAttribute = true;

                for (int i = 0; i < this.candidateAttributeSplitSuggestions.length; i++) {
                    if (this.candidateAttributeSplitSuggestions[i] != null)
                        this.candidateAttributeSplitSuggestions[i].updateCandidateAttributeInfo(inst, ht);
                }

                //  --- just for debug
                /**
                 double[][] best = new double[candidateAttributeSplitSuggestions[0].resultClassDistributeionsVector.size()][];
                 double best_idx = candidateAttributeSplitSuggestions[0].splitTest.getAttsTestDependsOn()[0];
                 double second_idx = candidateAttributeSplitSuggestions[1].splitTest.getAttsTestDependsOn()[0];
                 for (int i=0; i<best.length; i++) {
                 best[i] = candidateAttributeSplitSuggestions[0].resultClassDistributeionsVector.get(i).getArrayRef();
                 }

                 double[][] second = new double[candidateAttributeSplitSuggestions[1].resultClassDistributeionsVector.size()][];
                 for (int i=0; i<second.length; i++) {
                 second[i] = candidateAttributeSplitSuggestions[1].resultClassDistributeionsVector.get(i).getArrayRef();
                 }

                 double top2Merit = candidateAttributeSplitSuggestions[0].merit - candidateAttributeSplitSuggestions[1].merit;
                 double bset_meirt = getMeritOfSplit(best);
                 double second_merit = getMeritOfSplit(second);
                 SplitCriterion splitCriterion = (SplitCriterion) ht.getPreparedClassOption(ht.splitCriterionOption);
                 AttributeSplitSuggestion[] bestSplitSuggestions = ((ActiveLearningNode)this).getBestSplitSuggestions(splitCriterion, ht);

                 int len = bestSplitSuggestions.length;
                 double actual_best = 0, actual_second = 0;
                 for (int i=0; i<len; i++) {
                 if (bestSplitSuggestions[i].splitTest == null) continue;
                 if (bestSplitSuggestions[i].splitTest.getAttsTestDependsOn()[0] == best_idx) {
                 actual_best = bestSplitSuggestions[i].merit;
                 }
                 else if (bestSplitSuggestions[i].splitTest.getAttsTestDependsOn()[0] == second_idx) {
                 actual_second = bestSplitSuggestions[i].merit;
                 }
                 }

                 double actual = actual_best - actual_second;
                 System.out.println("actual: " + actual + " predict: "+ top2Merit + " ==> " + (bset_meirt-second_merit));
                 System.out.println();
                 **/
                // --- just for debug

                if (this.measureSplitPostion != -1) { // If VFDT has been split in the evaluation mode, perform a split attempt in the node of IMA
                    if (ht.attemptToSplitByTop2Attribute(this) > 0) {
                        ht.attempts ++;
                        if (this.measureSplitResult == 1){
                            //just for debug
                            //if (this.getWeightSeen() - measureSplitPostion > 1000) {
                            //    System.err.println("stop");
                            //}
                            //just for debug
                            ht.boundSplitErrors.addToValue(ht.boundSplitErrors.numValues(),Math.max(0., this.getWeightSeen() - measureSplitPostion));
                        }
                        else if (measureSplitResult == 2) {
                            ht.maxSplitErrors.addToValue(ht.maxSplitErrors.numValues(), Math.max(0., this.getWeightSeen() - measureSplitPostion));
                        }
                        this.measureSplitPostion = -1;
                        this.clearTopAttributeInfo();
                    }
                }
            }

            ht.incrementAttemptTime += (System.currentTimeMillis() - startTime);
        }

        public void updateAttributeInfo(Instance inst, HoeffdingTreeMeasure ht) {
            if (this.isInitialized == false) {
                this.attributeObservers = new AutoExpandVector<AttributeClassObserver>(inst.numAttributes());
                this.isInitialized = true;
            }

            for (int i = 0; i < inst.numAttributes() - 1; i++) {
                int instAttIndex = modelAttIndexToInstanceAttIndex(i, inst);
                AttributeClassObserver obs = this.attributeObservers.get(i);
                if (obs == null) {
                    obs = inst.attribute(instAttIndex).isNominal() ? ht.newNominalClassObserver() : ht.newNumericClassObserver();
                    this.attributeObservers.set(i, obs);
                }
                obs.observeAttributeClass(inst.value(instAttIndex), (int) inst.classValue(), inst.weight());
            }
        }

        public void learnFromInstance(Instance inst, HoeffdingTreeMeasure ht) {
            double startTime = System.currentTimeMillis();
            this.updateAttributeInfo(inst, ht);
            this.observedClassDistribution.addToValue((int) inst.classValue(), inst.weight());

            if (!this.isMeasureFinish) {
                this.increamentForTopAttribute(inst, ht);
            }
            ht.learnMoreTime += System.currentTimeMillis() - startTime;
        }

        public void setBestSplitTest(AttributeSplitSuggestion[] suggestions, HoeffdingTreeMeasure ht) {
            if (this.preSuggestion != null && this.preSuggestion.length > 0) {
                if (preSuggestion[0].splitTest.getAttsTestDependsOn()[0] != suggestions[0].splitTest.getAttsTestDependsOn()[0]) {
                    this.measureSplitDiffTimes++;
                }
                else if (this.preSuggestion[0].splitTest instanceof NumericAttributeBinaryTest) {
                    double attrv = ((NumericAttributeBinaryTest) this.preSuggestion[0].splitTest).getSplitValue();
                    double attrv2 = ((NumericAttributeBinaryTest) suggestions[0].splitTest).getSplitValue();
                    if (attrv != attrv2) {
                        this.measureSplitDiffTimes ++;
                    }
                }
            }

            this.preSuggestion = new AttributeSplitSuggestion[suggestions.length];
            for (int i=0; i<suggestions.length; i++) {
                double newMerit = -InfoGainSplitCriterion.computeEntropy(suggestions[i].resultingClassDistributions);
                this.preSuggestion[i] = new AttributeSplitSuggestion(suggestions[i].splitTest, suggestions[i].resultingClassDistributions.clone(), newMerit);
            }
        }

        public void setCandidateAttributeSplitSuggestions(AttributeSplitSuggestion[] splitSuggestions,
                                                          SplitCriterion splitCriterion, HoeffdingTreeMeasure ht, double hoeffdingBound, double merit) {
            double startTime = System.currentTimeMillis();

            int len = splitSuggestions.length;
            List<Integer> candidateAttributes = new ArrayList<>();

            this.illegalAttribute.clear();
            this.potentialAttribute.clear();

            //add top2 attributes by default
            candidateAttributes.add(len - 1);
            candidateAttributes.add(len - 2);
            double bestMerit = splitSuggestions[len -1].merit;
            int maxCandidateAttribute = (int)Math.min(Math.max(MIN_CANDIDATE_ATTRIBUTE, MAX_CANDIDATE_ATTRIBUTE_RATE * len), MAX_CANDIDATE_ATTRIBUTE);
            if (maxCandidateAttribute > len / 2) maxCandidateAttribute = 2;

            for (int i=len-3; i >= 0; i--) {
                if (bestMerit - splitSuggestions[i].merit > hoeffdingBound) {
                    if (splitSuggestions[i].splitTest != null)  {
                        this.illegalAttribute.add(splitSuggestions[i].splitTest.getAttsTestDependsOn()[0]);
                    }
                }
                else {
                    if (splitSuggestions[i].splitTest != null) {
                        if (candidateAttributes.size() < maxCandidateAttribute)
                            candidateAttributes.add(i);
                        else {
                            double newMerit = splitSuggestions[i].merit;
                            if (splitCriterion instanceof InfoGainSplitCriterion)
                                newMerit = -InfoGainSplitCriterion.computeEntropy(splitSuggestions[i].resultingClassDistributions);
                            potentialAttribute.add(new Tuple(splitSuggestions[i].splitTest.getAttsTestDependsOn()[0], -1, newMerit));
                        }
                    }
                }
            }

            this.candidateAttributeSplitSuggestions = new CandidateAttributeSplitSuggestion[candidateAttributes.size()];
            for (int i=0; i<candidateAttributes.size(); i++) {
                int idx = candidateAttributes.get(i);
                double newMerit = splitSuggestions[idx].merit;
                if (splitCriterion instanceof InfoGainSplitCriterion)
                    newMerit = -InfoGainSplitCriterion.computeEntropy(splitSuggestions[idx].resultingClassDistributions);

                if (splitSuggestions[idx].merit != splitSuggestions[idx].merit) {
                    double tmp = splitCriterion.getMeritOfSplit(null, new double[][]{this.observedClassDistribution.getArrayRef()});
                    System.out.println(tmp);
                }
                if (splitSuggestions[idx].splitTest == null) {
                    splitSuggestions[idx].resultingClassDistributions = new double[][]{this.observedClassDistribution.getArrayRef()};
                    if (newMerit != newMerit) newMerit = -200;
                }

                this.candidateAttributeSplitSuggestions[i] = new CandidateAttributeSplitSuggestion(splitSuggestions[idx], splitCriterion, newMerit);
            }

            this.lastIncrementTime = this.getWeightSeen();
            this.updateNumericalTime = getWeightSeen();
            ht.setAttributeMoreTime += System.currentTimeMillis() - startTime;
        }

        public CandidateAttributeSplitSuggestion[] getCandidateAttributeSplitSuggestions() {
            return candidateAttributeSplitSuggestions;
        }

        public boolean isIncreaseMode() {
            return this.candidateAttributeSplitSuggestions != null && this.candidateAttributeSplitSuggestions.length > 0;
        }

        public double getTop2CandidateMerit_diff() {
            if (this.candidateAttributeSplitSuggestions != null && this.candidateAttributeSplitSuggestions.length >=2) {
                double bestMerit = Integer.MIN_VALUE;
                double secondMerit = Integer.MIN_VALUE;
                int len = this.candidateAttributeSplitSuggestions.length;
                for (int i=0; i<len; i++) {
                    if (this.candidateAttributeSplitSuggestions[i] == null) continue;
                    if (this.candidateAttributeSplitSuggestions[i].merit >= bestMerit) {
                        secondMerit = bestMerit;
                        bestMerit = this.candidateAttributeSplitSuggestions[i].merit;
                    }
                    else if (this.candidateAttributeSplitSuggestions[i].merit > secondMerit) {
                        secondMerit = this.candidateAttributeSplitSuggestions[i].merit;
                    }
                }
                return bestMerit - secondMerit;
            }
            return 0.0;
        }

        private int checkPeriodic(double hoeffdingBound, HoeffdingTreeMeasure ht) {
            if (getWeightSeen() - this.updateNumericalTime >= ht.checkPeriodOption.getValue()) {

                double bestMerit = Integer.MIN_VALUE;
                double secondMerit = Integer.MIN_VALUE;

                PriorityQueue<Tuple> priorityQueue = new PriorityQueue<>();

                int interval = ht.incrementPeriodOption.getValue();

                for (int i = 0; i < this.candidateAttributeSplitSuggestions.length; i++) {
                    if (this.candidateAttributeSplitSuggestions[i] == null) continue;

                    int attr_idx = this.candidateAttributeSplitSuggestions[i].splitTest.getAttsTestDependsOn()[0];

                    AttributeClassObserver ob = this.attributeObservers.get(attr_idx);

                    if (this.candidateAttributeSplitSuggestions[i].splitTest instanceof NumericAttributeBinaryTest || interval > 1) {
                        SplitCriterion criterion = this.candidateAttributeSplitSuggestions[i].splitCriterion;
                        AttributeSplitSuggestion suggestion = ob.getBestEvaluatedSplitSuggestion(criterion, new double[]{}, attr_idx, ht.binarySplitsOption.isSet());

                        double newMerit = suggestion.merit;

                        this.candidateAttributeSplitSuggestions[i] = new CandidateAttributeSplitSuggestion(suggestion, criterion, newMerit);
                    }

                    double merit = this.candidateAttributeSplitSuggestions[i].merit;
                    priorityQueue.add(new Tuple(attr_idx, i, merit));
                    if (this.candidateAttributeSplitSuggestions[i].merit >= bestMerit) {
                        secondMerit = bestMerit;
                        bestMerit = this.candidateAttributeSplitSuggestions[i].merit;
                    }
                    else if (this.candidateAttributeSplitSuggestions[i].merit > secondMerit) {
                        secondMerit = this.candidateAttributeSplitSuggestions[i].merit;
                    }
                }

                this.updateNumericalTime = getWeightSeen();
                ht.checkNumeric ++;

                if (bestMerit - secondMerit > hoeffdingBound) {
                    ht.hoeffdingJudge++;
                    return 1;
                }

                int newAdd = -1;

                while (priorityQueue.size() > 0) {
                    Tuple cur = priorityQueue.poll();
                    if (this.candidateAttributeSplitSuggestions[cur.pos] == null) continue;

                    if (cur.attr_idx == newAdd) {
                        priorityQueue.add(cur);
                        break;
                    }

                    int next_idx = -1;
                    if (this.potentialAttribute.size() > 0) {
                        next_idx = this.potentialAttribute.poll().attr_idx;
                    }
                    if (next_idx == -1) {
                        priorityQueue.add(cur);
                        break;
                    }

                    ht.checkOrder ++;
                    SplitCriterion criterion = this.candidateAttributeSplitSuggestions[cur.pos].splitCriterion;
                    AttributeClassObserver ob = this.attributeObservers.get(next_idx);
                    AttributeSplitSuggestion suggestion = ob.getBestEvaluatedSplitSuggestion(criterion, new double[]{}, next_idx, ht.binarySplitsOption.isSet());

                    double newMerit = suggestion.merit;

                    if (newMerit > cur.merit) {
                        ht.checkOrderRight ++;
                        this.candidateAttributeSplitSuggestions[cur.pos] = null;

                        if (bestMerit - cur.merit > hoeffdingBound) {
                            this.illegalAttribute.add(cur.attr_idx);
                        }
                        else {
                            this.potentialAttribute.add(new Tuple(cur.attr_idx, -1, cur.merit));
                        }

                        if (bestMerit - newMerit > hoeffdingBound) {
                            this.illegalAttribute.add(next_idx);
                        }
                        else {
                            this.candidateAttributeSplitSuggestions[cur.pos] = new CandidateAttributeSplitSuggestion(suggestion, criterion, newMerit);
                            newAdd = next_idx;
                            priorityQueue.add(new Tuple(next_idx, cur.pos, newMerit));
                            bestMerit = Math.max(bestMerit, newMerit);
                        }
                    }
                    else {
                        if (bestMerit - newMerit > hoeffdingBound) {
                            this.illegalAttribute.add(next_idx);
                        }
                        else {
                            this.potentialAttribute.add(new Tuple(next_idx, -1, newMerit));
                        }
                        break;
                    }
                }
            }
            return 0;
        }

        public int attemptToSplitForCandidateInfo(double hoeffdingBound, HoeffdingTreeMeasure ht){
            if (this.isIncreaseMode() && this.updateCandidateAttribute) {
                this.updateCandidateAttribute = false;

                double top2Merit = getTop2CandidateMerit_diff();
                if (top2Merit > hoeffdingBound) {
                    ht.hoeffdingJudge ++;
                    return 1;
                }
                return checkPeriodic(hoeffdingBound, ht);
            }
            return 0;
        }

        public double getWeightSeen() {
            return this.observedClassDistribution.sumOfValues();
        }

        public double getWeightSeenAtLastSplitEvaluation() {
            return this.weightSeenAtLastSplitEvaluation;
        }

        public void setWeightSeenAtLastSplitEvaluation(double weight) {
            this.weightSeenAtLastSplitEvaluation = weight;
        }

        public void setMeasureSplitPostion(double measureSplitPostion) {
            this.measureSplitPostion = measureSplitPostion;
        }

        public void setMeasureSplitResult(int measureSplitResult) {
            this.measureSplitResult = measureSplitResult;
        }

        public int calcByteSize() {
            return super.calcByteSize() + (int) SizeOf.fullSizeOf(this.candidateAttributeSplitSuggestions)
                    + (int) SizeOf.fullSizeOf(this.attributeObservers) + (int) (SizeOf.fullSizeOf(this.illegalAttribute)) + (int) (SizeOf.fullSizeOf(this.potentialAttribute));

        }
    }

    public static class SplitNode extends LearningNode {

        private static final long serialVersionUID = 1L;

        protected InstanceConditionalTest splitTest;

        protected AutoExpandVector<Node> children; // = new AutoExpandVector<Node>();

        @Override
        public int calcByteSize() {
            return super.calcByteSize()
                    + (int) (SizeOf.sizeOf(this.children) + SizeOf.fullSizeOf(this.splitTest));
        }

        @Override
        public int calcByteSizeIncludingSubtree() {
            int byteSize = calcByteSize();
            for (Node child : this.children) {
                if (child != null) {
                    byteSize += child.calcByteSizeIncludingSubtree();
                }
            }
            return byteSize;
        }

        public SplitNode(InstanceConditionalTest splitTest,
                         double[] classObservations, int size) {
            super(classObservations);
            this.observedClassDistribution = new DoubleVector(classObservations);
            this.splitTest = splitTest;
            this.children = new AutoExpandVector<Node>(size);
        }

        public SplitNode(InstanceConditionalTest splitTest,
                         double[] classObservations) {
            super(classObservations);
            this.observedClassDistribution = new DoubleVector(classObservations);
            this.splitTest = splitTest;
            this.children = new AutoExpandVector<Node>();
        }

        public SplitNode(InstanceConditionalTest splitTest,
                         double[] classObservations, int size, LearningNode node) {
            super(classObservations, node.candidateAttributeSplitSuggestions,
                    node.measureSplitPostion, node.isMeasureFinish, node.measureSplitResult);
            this.observedClassDistribution = new DoubleVector(classObservations);
            this.splitTest = splitTest;
            this.children = new AutoExpandVector<Node>(size);

            // just for debug
            this.updateNumericalTime = node.updateNumericalTime;
            this.preSuggestion = node.preSuggestion;
            this.weightSeenAtLastSplitEvaluation = node.weightSeenAtLastSplitEvaluation;
            this.attributeObservers = node.attributeObservers;
            this.isInitialized = node.isInitialized;

            this.potentialAttribute = node.potentialAttribute;
            this.illegalAttribute = node.illegalAttribute;
            this.lastIncrementTime = node.lastIncrementTime;
            this.updateCandidateAttribute = node.updateCandidateAttribute;
            // just for debug
        }

        public int numChildren() {
            return this.children.size();
        }

        public void setChild(int index, Node child) {
            if ((this.splitTest.maxBranches() >= 0)
                    && (index >= this.splitTest.maxBranches())) {
                throw new IndexOutOfBoundsException();
            }
            this.children.set(index, child);
        }

        public Node getChild(int index) {
            return this.children.get(index);
        }

        public int instanceChildIndex(Instance inst) {
            return this.splitTest.branchForInstance(inst);
        }

        @Override
        public boolean isLeaf() {
            return false;
        }

        @Override
        public FoundNode filterInstanceToLeaf(Instance inst, SplitNode parent,
                                              int parentBranch) {
            int childIndex = instanceChildIndex(inst);
            if (childIndex >= 0) {
                Node child = getChild(childIndex);
                if (child != null) {
                    return child.filterInstanceToLeaf(inst, this, childIndex);
                }
                return new FoundNode(null, this, childIndex);
            }
            return new FoundNode(this, parent, parentBranch);
        }

        @Override
        public void describeSubtree(HoeffdingTreeMeasure ht, StringBuilder out,
                                    int indent) {
            for (int branch = 0; branch < numChildren(); branch++) {
                Node child = getChild(branch);
                if (child != null) {
                    StringUtils.appendIndented(out, indent, "if ");
                    out.append(this.splitTest.describeConditionForBranch(branch,
                            ht.getModelContext()));
                    out.append(": ");
                    StringUtils.appendNewline(out);
                    child.describeSubtree(ht, out, indent + 2);
                }
            }
        }

        @Override
        public int subtreeDepth() {
            int maxChildDepth = 0;
            for (Node child : this.children) {
                if (child != null) {
                    int depth = child.subtreeDepth();
                    if (depth > maxChildDepth) {
                        maxChildDepth = depth;
                    }
                }
            }
            return maxChildDepth + 1;
        }

        //@Override
        public void learnFromInstance(Instance inst, HoeffdingTreeMeasure ht) {
            double startTime = System.currentTimeMillis();

            // If VFDT has been split and IMA is not, we need to continue calculating the delay of IMA.
            if (!this.isMeasureFinish) {
                this.observedClassDistribution.addToValue((int) inst.classValue(), inst.weight());
                this.updateAttributeInfo(inst, ht);
                this.increamentForTopAttribute(inst, ht);
            }

            int childIndex = instanceChildIndex(inst);
            if (childIndex >= 0) {
                Node child = getChild(childIndex);
                if (child != null) {
                    ((LearningNode)child).learnFromInstance(inst, ht);
                }
            }
            ht.SplitNodeMoreTime += System.currentTimeMillis() - startTime;
        }
    }

    public static class InactiveLearningNode extends LearningNode {

        private static final long serialVersionUID = 1L;

        public InactiveLearningNode(double[] initialClassObservations) {
            super(initialClassObservations);
            this.observedClassDistribution = new DoubleVector(initialClassObservations);
        }

        public InactiveLearningNode(double[] initialClassObservations,
                                    CandidateAttributeSplitSuggestion[] candidateAttributeSplitSuggestions,
                                    double measureSplitPostion, boolean isMeasureFinish, int measureSplitResult) {
            super(initialClassObservations, candidateAttributeSplitSuggestions, measureSplitPostion, isMeasureFinish, measureSplitResult);
            this.observedClassDistribution = new DoubleVector(initialClassObservations);
        }

        public InactiveLearningNode(double[] initialClassObservations, LearningNode node) {
            super(initialClassObservations, node.candidateAttributeSplitSuggestions, node.measureSplitPostion, node.isMeasureFinish, node.measureSplitResult);
            this.observedClassDistribution = new DoubleVector(initialClassObservations);

            // just for debug
            this.updateNumericalTime = node.updateNumericalTime;
            this.preSuggestion = node.preSuggestion;
            this.weightSeenAtLastSplitEvaluation = node.weightSeenAtLastSplitEvaluation;
            this.attributeObservers = node.attributeObservers;
            this.isInitialized = node.isInitialized;

            this.potentialAttribute = node.potentialAttribute;
            this.illegalAttribute = node.illegalAttribute;
            this.lastIncrementTime = node.lastIncrementTime;
            this.updateCandidateAttribute = node.updateCandidateAttribute;
            //just for debug
        }

        //@Override
        public void learnFromInstance(Instance inst, HoeffdingTreeMeasure ht) {
            double startTime = System.currentTimeMillis();
            this.observedClassDistribution.addToValue((int) inst.classValue(),
                    inst.weight());

            if (!this.isMeasureFinish) {
                this.updateAttributeInfo(inst, ht);
                this.increamentForTopAttribute(inst, ht);
            }

            ht.InactiveNodeMoreTime += System.currentTimeMillis() - startTime;
        }
    }

    public static class ActiveLearningNode extends LearningNode {

        private static final long serialVersionUID = 1L;

        protected double initialWeight;

        private int gracePeriod = 0;

        public ActiveLearningNode(double[] initialClassObservations, int gracePeriod) {
            this(initialClassObservations);
            this.gracePeriod = gracePeriod;
        }

        public ActiveLearningNode(double[] initialClassObservations) {
            super(initialClassObservations);
            this.initialWeight = getWeightSeen();
            this.isInitialized = false;
        }

        @Override
        public int calcByteSize() {
            return super.calcByteSize();
        }

        @Override
        public void learnFromInstance(Instance inst, HoeffdingTreeMeasure ht) {
            double startTime = System.currentTimeMillis();

            super.learnFromInstance(inst, ht);
            ht.ActiveNodeMoreTime += System.currentTimeMillis() - startTime;
        }

        public void setGracePeriod(int gracePeriod) {
            this.gracePeriod = gracePeriod;
        }

        public int getGracePeriod() {
            return this.gracePeriod;
        }

        public double getInitialWeight() {
            return initialWeight;
        }

        public AttributeSplitSuggestion[] getBestSplitSuggestions(
                SplitCriterion criterion, HoeffdingTreeMeasure ht) {
            List<AttributeSplitSuggestion> bestSuggestions = new LinkedList<AttributeSplitSuggestion>();
            double[] preSplitDist = this.observedClassDistribution.getArrayCopy();
            if (!ht.noPrePruneOption.isSet()) {
                // add null split as an option
                bestSuggestions.add(new AttributeSplitSuggestion(null,
                        new double[0][], criterion.getMeritOfSplit(
                        preSplitDist,
                        new double[][]{preSplitDist})));
            }
            for (int i = 0; i < this.attributeObservers.size(); i++) {
                AttributeClassObserver obs = this.attributeObservers.get(i);
                if (obs != null) {
                    AttributeSplitSuggestion bestSuggestion = obs.getBestEvaluatedSplitSuggestion(criterion,
                            preSplitDist, i, ht.binarySplitsOption.isSet());
                    if (bestSuggestion != null) {
                        bestSuggestions.add(bestSuggestion);
                    }
                }
            }
            return bestSuggestions.toArray(new AttributeSplitSuggestion[bestSuggestions.size()]);
        }

        public void disableAttribute(int attIndex) {
            this.attributeObservers.set(attIndex,
                    new NullAttributeClassObserver());
        }
    }

    protected Node treeRoot;

    protected int decisionNodeCount;

    protected int activeLeafNodeCount;

    protected int inactiveLeafNodeCount;

    protected double inactiveLeafByteSizeEstimate;

    protected double activeLeafByteSizeEstimate;

    protected double byteSizeEstimateOverheadFraction;

    protected boolean growthAllowed;

    public int calcByteSize() {
        int size = (int) SizeOf.sizeOf(this);
        if (this.treeRoot != null) {
            size += this.treeRoot.calcByteSizeIncludingSubtree();
        }
        return size;
    }

    @Override
    public int measureByteSize() {
        String filePatch = splitTimePredictionOption.getChosenLabel();
        filePatch += String.valueOf(incrementPeriodOption.getValue());

        System.err.println(filePatch + " : attemptTime " + this.attemptToSplitTime/1000. +  "s all " + (this.trainOnInstanceTime + this.voteOnInstanceTime)/1000. + "s");
        System.err.println(filePatch + " : attempts " + this.attempts + " splits " + (this.boundSplits + this.maxSplits) + " boundsplits " + this.boundSplits + " maxsplits " + this.maxSplits);
        System.err.println(filePatch + " : increment Attempt Time " + this.incrementAttemptTime / 1000. + "s  top Attribute Attempt Time " + this.topAttributeAttemptTime/1000. + "s" );
        System.err.println(filePatch + " : updateAttributeMoreTime " + this.updateAttributeMoreTime / 1000 + "s");
        System.err.println(filePatch + " : calAttributeMoreTime " + this.calAttributeMoreTime / 1000 + "s");
        System.err.println(filePatch + " : limitAttempt " + this.limitAttempt  + "  hoeffdingAttempt " + this.hoeffdingJudge);
        System.err.println(filePatch + " : numeric " + this.checkNumeric  + "  order " + this.checkOrder + " right " + this.checkOrderRight
                +" rate "+ this.checkOrderRight / this.checkOrder);


        if (this.splitTimePredictionOption.getChosenIndex() == 3)
            System.err.println(filePatch + " : Brent searchs " + brentSearches + " iterations " + brentTotalIterations);

        if (measureSplitOffsetOption.isSet())
            System.err.println(filePatch + " : boundOffset " + Utils.mean(this.boundSplitErrors.getArrayRef()) + " maxOffset " + Utils.mean(this.maxSplitErrors.getArrayRef()));
        if (!savePathOption.getValue().equals("")) {
            Map<String, String> env = System.getenv();
            String dir = savePathOption.getValue();
            if (!dir.endsWith("/"))
                dir = dir + "/";

            try {
                String fileName = dir + "statistics_" + filePatch + ".txt";
                //System.err.println("fileName :" + fileName);
                PrintWriter writer = new PrintWriter(new FileOutputStream(fileName, false));
                writer.println(String.format("attempts \t boundSplits \t maxSplits \t boundOffset \t maxOffset"));
                writer.println(String.format("%d \t %d \t %d \t %.2f \t %.2f",  attempts, boundSplits, maxSplits, Utils.mean(this.boundSplitErrors.getArrayRef()), Utils.mean(this.maxSplitErrors.getArrayRef())));
                writer.println();

                writer.println(String.format("totalTime \t trainTime \t predictTime \t attemptTime"));
                writer.println(String.format("%.2f \t %.2f \t %.2f \t %.2f", (this.trainOnInstanceTime + this.voteOnInstanceTime) / 1000., this.trainOnInstanceTime / 1000., this.voteOnInstanceTime / 1000., this.attemptToSplitTime / 1000.));
                writer.println();

                writer.println(String.format("trainMainTime \t trainFitTime \t estimateModelTime \t enforeModelTime"));
                writer.println(String.format("%.2f \t %.2f \t %.2f \t %.2f", this.trainMainTime / 1000, this.trainFitTime / 1000, this.estimateModelTime/1000, this.enforeModelTime/1000));
                writer.println();

                writer.println(String.format("learnMoreTime \t splitMoreTime \t activeMoreTime \t inactiveMoreTime"));
                writer.println(String.format("%.2f \t %.2f \t %.2f \t %.2f", this.learnMoreTime /1000, this.SplitNodeMoreTime/1000, this.ActiveNodeMoreTime/1000, this.InactiveNodeMoreTime/1000));
                writer.println();

                writer.println(String.format("incrementAttemptTime \t topAttributeAttemptTime \t updateAttributeMoreTime \t setAttributeMoreTime"));
                writer.println(String.format("%.2f \t %.2f \t %.2f \t %.2f ", this.incrementAttemptTime / 1000, this.topAttributeAttemptTime/1000, this.updateAttributeMoreTime / 1000, this.setAttributeMoreTime / 1000));
                writer.println();

                writer.close();

                fileName = dir + "splitNumSamples_" + filePatch + ".txt";
                writer = new PrintWriter(new FileOutputStream(fileName, false));
                writer.println(Utils.arrayToString(this.boundSplitNumSamples.getArrayRef()));
                writer.println(Utils.arrayToString(this.tieSplitNumSamples.getArrayRef()));
                writer.close();

            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
        }
        return calcByteSize();
    }

    @Override
    public void resetLearningImpl() {
        this.treeRoot = null;
        this.decisionNodeCount = 0;
        this.activeLeafNodeCount = 0;
        this.inactiveLeafNodeCount = 0;
        this.inactiveLeafByteSizeEstimate = 0.0;
        this.activeLeafByteSizeEstimate = 0.0;
        this.byteSizeEstimateOverheadFraction = 1.0;
        this.growthAllowed = true;
        if (this.leafpredictionOption.getChosenIndex()>0) {
            this.removePoorAttsOption = null;
        }
    }

    public void singleFit(LearningNode learningNode, SplitNode parent, int parentBranch) {
        if (this.growthAllowed
                && (learningNode instanceof ActiveLearningNode)) {
            ActiveLearningNode activeLearningNode = (ActiveLearningNode) learningNode;
            double weightSeen = activeLearningNode.getWeightSeen();
            int gracePeriod = this.gracePeriodOption.getValue();
            if (this.splitTimePredictionOption.getChosenIndex() > 1 && activeLearningNode.getGracePeriod() > 0) {
                gracePeriod = activeLearningNode.getGracePeriod();
            }

            if (activeLearningNode.isIncreaseMode()) {   // incremental mode
                if (!activeLearningNode.observedClassDistributionIsPure() && attemptToSplitByTop2Attribute(activeLearningNode) > 0) {
                    this.attempts++;
                    int[] splitResult = {0};
                    LearningNode returnNode = attemptToSplit(activeLearningNode, parent, parentBranch, true, splitResult);
                    activeLearningNode.setWeightSeenAtLastSplitEvaluation(weightSeen);
                    if (splitResult[0] > 0) {
                        returnNode.setMeasureSplitResult(splitResult[0]);
                        returnNode.setMeasureSplitPostion(activeLearningNode.getWeightSeen());
                        activeLearningNode.clearTopAttributeInfo();
                        returnNode.clearTopAttributeInfo();
                    }
                }
            }
            else {
                if (weightSeen - activeLearningNode.getWeightSeenAtLastSplitEvaluation() >= gracePeriod) {
                    if (!activeLearningNode.observedClassDistributionIsPure()) {
                        this.attempts++;
                        int[] splitResult = {0};
                        LearningNode returnNode = attemptToSplit(activeLearningNode, parent, parentBranch, true, splitResult);
                        if (splitResult[0] > 0) {
                            activeLearningNode.clearTopAttributeInfo();
                            returnNode.clearTopAttributeInfo();
                        }
                    }
                    activeLearningNode.setWeightSeenAtLastSplitEvaluation(weightSeen);
                }
            }
        }
    }

    // calculate split dealy
    // We use the method of delay measurement of ICDM
    public void singleFitMeasure(LearningNode learningNode, SplitNode parent, int parentBranch){
        if (this.growthAllowed
                && (learningNode instanceof ActiveLearningNode)) {
            ActiveLearningNode activeLearningNode = (ActiveLearningNode) learningNode;
            double weightSeen = activeLearningNode.getWeightSeen();

            int gracePeriod = MEASURE_GRACE_PERIOD;
            boolean adaptGracePeriod = true;
            if (activeLearningNode.getWeightSeenAtLastSplitEvaluation() == activeLearningNode.getInitialWeight()){
                gracePeriod = INITIAL_GRACE_PERIOD;
                //adaptGracePeriod = false;
            }
            if (weightSeen < activeLearningNode.getInitialWeight() + this.gracePeriodOption.getValue()) {
                adaptGracePeriod = false;
            }

            int[] splitResult = {0};
            LearningNode returnNode;
            if (activeLearningNode.isIncreaseMode()) { // IMA algorithm
                if (activeLearningNode.observedClassDistributionIsPure())
                    return;

                adaptGracePeriod = false;
                int res = attemptToSplitByTop2Attribute(activeLearningNode);
                if (res > 0) {
                    this.attempts++;
                    adaptGracePeriod = true;
                }
                returnNode = attemptToSplit(activeLearningNode, parent, parentBranch, adaptGracePeriod, splitResult);

                if (adaptGracePeriod) {
                    if (splitResult[0] > 0) {
                        activeLearningNode.clearTopAttributeInfo();
                        returnNode.clearTopAttributeInfo();
                        if (splitResult[0] == 1) {
                            this.boundSplitErrors.addToValue(this.boundSplitErrors.numValues(), 0);
                        } else if (splitResult[0] == 2) {
                            this.maxSplitErrors.addToValue(this.maxSplitErrors.numValues(), 0);
                        }
                    }
                    activeLearningNode.setWeightSeenAtLastSplitEvaluation(weightSeen);
                }

                if (splitResult[0] > 0 && returnNode != null) {
                    returnNode.setMeasureSplitResult(splitResult[0]);
                    returnNode.setMeasureSplitPostion(activeLearningNode.getWeightSeen());
                }
            }
            else if (weightSeen - activeLearningNode.getWeightSeenAtLastSplitEvaluation() >= gracePeriod) {
                activeLearningNode.setGracePeriod(activeLearningNode.getGracePeriod() - gracePeriod);
                if (!activeLearningNode.observedClassDistributionIsPure()) {
                    if (activeLearningNode.getGracePeriod() <= 0) {
                        returnNode = attemptToSplit(activeLearningNode, parent, parentBranch, adaptGracePeriod, splitResult);
                        this.attempts++;
                    } else {
                        returnNode = attemptToSplit(activeLearningNode, parent, parentBranch, false, splitResult);
                    }

                    if (splitResult[0] > 0) {
                        activeLearningNode.clearTopAttributeInfo();
                        returnNode.clearTopAttributeInfo();
                        returnNode.setMeasureSplitResult(splitResult[0]);
                        returnNode.setMeasureSplitPostion(activeLearningNode.getWeightSeen());
                    }
                    if (splitResult[0] == 1) {
                        this.boundSplitErrors.addToValue(this.boundSplitErrors.numValues(),Math.max(0., activeLearningNode.getGracePeriod()));
                        if (activeLearningNode.getGracePeriod() > 0) this.attempts ++; //This is different from ICDM code, we think we should add a split attempt here
                    }
                    else if (splitResult[0] == 2) {
                        this.maxSplitErrors.addToValue(this.maxSplitErrors.numValues(), Math.max(0., activeLearningNode.getGracePeriod()));
                        if (activeLearningNode.getGracePeriod() > 0) this.attempts ++; //This is different from ICDM code, we think we should add a split attempt here
                    }
                }
                activeLearningNode.setWeightSeenAtLastSplitEvaluation(weightSeen);
            }
        }
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        trainStepCount++;
        long startTime = System.currentTimeMillis();

        if (this.treeRoot == null) {
            this.treeRoot = newLearningNode();
            this.activeLeafNodeCount = 1;
        }
        FoundNode foundNode = this.treeRoot.filterInstanceToLeaf(inst, null, -1);
        Node leafNode = foundNode.node;
        if (leafNode == null) {
            leafNode = newLearningNode();
            foundNode.parent.setChild(foundNode.parentBranch, leafNode);
            this.activeLeafNodeCount++;
        }
        double time1 = System.currentTimeMillis();
        this.filterLeafTime += time1 - startTime;

        if (this.measureSplitOffsetOption.isSet()) // evaluation mode
            ((LearningNode)this.treeRoot).learnFromInstance(inst, this);

        if (leafNode instanceof LearningNode) {
            LearningNode learningNode = (LearningNode) leafNode;
            if (!this.measureSplitOffsetOption.isSet())
                learningNode.learnFromInstance(inst, this);

            // fit time
            double time2 = System.currentTimeMillis();
            if (measureSplitOffsetOption.isSet())
                this.singleFitMeasure(learningNode, foundNode.parent, foundNode.parentBranch);
            else
                this.singleFit(learningNode, foundNode.parent, foundNode.parentBranch);
            this.trainFitTime += System.currentTimeMillis() - time2;
        }
        double time2 = System.currentTimeMillis();
        this.trainMainTime += time2 - time1;

        if (this.trainingWeightSeenByModel
                % this.memoryEstimatePeriodOption.getValue() == 0) {
            estimateModelByteSizes();
        }
        double time3 = System.currentTimeMillis();
        this.estimateModelTime += time3 - time2;
        this.trainOnInstanceTime +=  time3 - startTime;
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        long startTime = System.currentTimeMillis();
        if (this.treeRoot != null) {
            FoundNode foundNode = this.treeRoot.filterInstanceToLeaf(inst,
                    null, -1);
            Node leafNode = foundNode.node;
            if (leafNode == null) {
                leafNode = foundNode.parent;
            }
            this.voteOnInstanceTime += System.currentTimeMillis() - startTime;
            return leafNode.getClassVotes(inst, this);
        } else {
            int numClasses = inst.dataset().numClasses();
            this.voteOnInstanceTime += System.currentTimeMillis() - startTime;
            return new double[numClasses];
        }

    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return new Measurement[]{
                new Measurement("tree size (nodes)", this.decisionNodeCount
                        + this.activeLeafNodeCount + this.inactiveLeafNodeCount),
                new Measurement("tree size (leaves)", this.activeLeafNodeCount
                        + this.inactiveLeafNodeCount),
                new Measurement("active learning leaves",
                        this.activeLeafNodeCount),
                new Measurement("tree depth", measureTreeDepth()),
                new Measurement("active leaf byte size estimate",
                        this.activeLeafByteSizeEstimate),
                new Measurement("inactive leaf byte size estimate",
                        this.inactiveLeafByteSizeEstimate),
                new Measurement("byte size estimate overhead",
                        this.byteSizeEstimateOverheadFraction)};
    }

    public int measureTreeDepth() {
        if (this.treeRoot != null) {
            return this.treeRoot.subtreeDepth();
        }
        return 0;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
        this.treeRoot.describeSubtree(this, out, indent);
    }

    @Override
    public boolean isRandomizable() {
        return false;
    }

    public static double computeHoeffdingBound(double range, double confidence,
                                               double n) {
        return Math.sqrt(((range * range) * Math.log(1.0 / confidence))
                / (2.0 * n));
    }

    //Procedure added for Hoeffding Adaptive Trees (ADWIN)
    protected SplitNode newSplitNode(InstanceConditionalTest splitTest,
                                     double[] classObservations, int size) {
        return new SplitNode(splitTest, classObservations, size);
    }

    protected SplitNode newSplitNode(InstanceConditionalTest splitTest,
                                     double[] classObservations) {
        return new SplitNode(splitTest, classObservations);
    }

    protected SplitNode newSplitNode(InstanceConditionalTest splitTest,
                                     double[] classObservations, int size, ActiveLearningNode node) {
        return new SplitNode(splitTest, classObservations, size, node);
    }

    protected AttributeClassObserver newNominalClassObserver() {
        AttributeClassObserver nominalClassObserver = (AttributeClassObserver) getPreparedClassOption(this.nominalEstimatorOption);
        return (AttributeClassObserver) nominalClassObserver.copy();
    }

    protected AttributeClassObserver newNumericClassObserver() {
        AttributeClassObserver numericClassObserver = (AttributeClassObserver) getPreparedClassOption(this.numericEstimatorOption);
        return (AttributeClassObserver) numericClassObserver.copy();
    }

    protected LearningNode splitNode(ActiveLearningNode node, SplitNode parent, int parentIndex, AttributeSplitSuggestion splitDecision){
        SplitNode newSplit = null;
        if (splitDecision.splitTest == null) {
            // preprune - null wins
            return deactivateLearningNode(node, parent, parentIndex);
        } else {
            newSplit = newSplitNode(splitDecision.splitTest,
                    node.getObservedClassDistribution(),splitDecision.numSplits(), node);
            for (int i = 0; i < splitDecision.numSplits(); i++) {
                Node newChild = newLearningNode(splitDecision.resultingClassDistributionFromSplit(i));
                newSplit.setChild(i, newChild);
            }
            this.activeLeafNodeCount--;
            this.decisionNodeCount++;
            this.activeLeafNodeCount += splitDecision.numSplits();
            if (parent == null) {
                this.treeRoot = newSplit;
            } else {
                parent.setChild(parentIndex, newSplit);
            }
        }
        // manage memory
        enforceTrackerLimit();
        return newSplit;
    }

    public static double getHoeffdingN(double range, double confidence, double delta) {
        return ((range * range) * Math.log(1.0 / confidence)) / (2.0 * Math.pow(delta, 2.));
    }

    static double Brent(double a,  double b, double iEps, double oEps, IOptFunction optFunction, Object...params)
    {
        double  fa, fb, fc, fs, c, c0, c1, c2,temp, d, s;
        int mflag;

        c = a;
        d = c;
        fa = optFunction.func(a, params);
        fb = optFunction.func(b, params);
        fc = fa;

        if (fa*fb >= 0) {
            System.err.println("Error no root between x0 and x1: f(x0) = " + fa + " f(x1)=" + fb);
            return -1.;
        }

        if ( Math.abs(fa) < Math.abs(fb))
        {
            temp = a;
            a = b;
            b = temp;
            temp = fa;
            fa = fb;
            fb = temp;
        } // if

        mflag = 1;
        brentSearches++;
        while ( (Math.abs(fb) > oEps) && ( Math.abs(b-a) > iEps))
        {
            brentTotalIterations++;
            if ( (fa != fc) && fb != fc)
            {
                c0 = a*fb*fc/((fa-fb)*(fa-fc));
                c1 = b*fa*fc/((fb-fa)*(fb-fc));
                c2 = c*fa*fb/((fc-fa)*(fc-fb));

                s = c0 + c1 + c2;
            } // if
            else
                s = b - fb*(b-a)/(fb - fa);

            if ( ( s < (3*(a+b)/4) || s > b) || ( (mflag == 1) &&
                    Math.abs(s-b) >= (Math.abs(b-c)/2) ) ||
                    ( (mflag == 0) && Math.abs(s-b) >= (Math.abs(c-d)/2) ) )
            {
                s = (a+b)/2;
                mflag = 1;
            } //if
            else
                mflag = 0;

            fs = optFunction.func(s, params);
            d = c;
            c = b;
            fc = fb;
            if ( (fa*fs)< 0)
                b = s;
            else
                a = s;
            if ( Math.abs(fa) < Math.abs(fb))
            {
                temp = a;
                a = b;
                b = temp;
                temp = fa;
                fa = fb;
                fb = temp;
            } // if

        } // while

        return b;
    } //brent

    interface IOptFunction {
        double func(double x, Object[] params);
    }

    static class OptFunction implements IOptFunction {
        public double func(double x, Object[] params) {
            double[][] dists = ((double[][])params[0]).clone();
            int minEntropyIdx = (int)params[1];
            int maxClassIdx = (int)params[2];
            double secEntropy = (double)params[3];
            double entropyRange = (double)params[4];
            double splitConfidence = (double)params[5];
            double numTrainSamples = (double)params[6];

            double delta = computeHoeffdingBound(entropyRange, splitConfidence, numTrainSamples + x);
            dists[minEntropyIdx] = dists[minEntropyIdx].clone();
            dists[minEntropyIdx][maxClassIdx] += x;
            double entropy = InfoGainSplitCriterion.computeEntropy(dists);
            double result = secEntropy - entropy - delta;
            return result;
        }
    }

    public static int splitTimePredictionCGD(AttributeSplitSuggestion bestSuggestion, AttributeSplitSuggestion secondBestSuggestion, double entropyRange, double splitConfidence, double numTrainSamples, double tieDelta, int gracePeriod){
        double delta = bestSuggestion.merit - secondBestSuggestion.merit;
        if (delta > 0) {
            double estimate = getHoeffdingN(entropyRange, splitConfidence, delta) - numTrainSamples;
            return (int) Math.ceil(Math.min(estimate, MAX_STEPS));
        } else
            return MAX_STEPS;

    }

    public static int splitTimePredictionOSM(AttributeSplitSuggestion bestSuggestion, AttributeSplitSuggestion secondBestSuggestion, double entropyRange, double splitConfidence, double numTrainSamples, double tieDelta, int gracePeriod){
        double minEntropy = Double.MAX_VALUE;
        int minEntropyIdx = 0;
        for (int i = 0; i < bestSuggestion.resultingClassDistributions.length; i++){
            double weight = Utils.sum(bestSuggestion.resultingClassDistributions[i]);
            double entropy = InfoGainSplitCriterion.computeEntropy(bestSuggestion.resultingClassDistributions[i]);
            if (entropy < minEntropy && weight > 0){
                minEntropyIdx = i;
                minEntropy = entropy;
            }
        }
        double secEntropy = InfoGainSplitCriterion.computeEntropy(secondBestSuggestion.resultingClassDistributions);

        double[][] dists = bestSuggestion.resultingClassDistributions.clone();
        int maxClassIdx = Utils.maxIndex(dists[minEntropyIdx]);
        double delta = computeHoeffdingBound(entropyRange, splitConfidence, numTrainSamples + MAX_STEPS);
        dists[minEntropyIdx] = dists[minEntropyIdx].clone();

        dists[minEntropyIdx][maxClassIdx] += MAX_STEPS;
        double tmpEntropy = InfoGainSplitCriterion.computeEntropy(dists);
        double result = MAX_STEPS;

        if (secEntropy - tmpEntropy > delta){
            result = Brent(0, (double) MAX_STEPS, 0.5, 0.005, new OptFunction(),
                    bestSuggestion.resultingClassDistributions, minEntropyIdx, maxClassIdx, secEntropy, entropyRange, splitConfidence, numTrainSamples);
        }
        result = Math.ceil(Math.min(Math.max(gracePeriod, result), MAX_STEPS));
        return (int)result;
    }

    public int attemptToSplitByTop2Attribute(LearningNode node) {
        double startTime = System.currentTimeMillis();
        SplitCriterion splitCriterion = (SplitCriterion) getPreparedClassOption(this.splitCriterionOption);
        double criterionRange = splitCriterion.getRangeOfMerit(node.getObservedClassDistribution());
        double hoeffdingBound = computeHoeffdingBound(criterionRange,
                this.splitConfidenceOption.getValue(), node.getWeightSeen());

        int attemptToSplitForCandInfo = node.attemptToSplitForCandidateInfo(hoeffdingBound, this);

        this.topAttributeAttemptTime += System.currentTimeMillis() - startTime;
        if (attemptToSplitForCandInfo > 0)
            return attemptToSplitForCandInfo;
        if (hoeffdingBound <= this.tieThresholdOption.getValue())
            return 3;
        return 0;
    }

    protected LearningNode attemptToSplit(ActiveLearningNode node, SplitNode parent, int parentIndex, boolean adaptGracePeriod, int[] splitResult) {
        double startTime = System.currentTimeMillis();
        LearningNode returnNode = null;

        SplitCriterion splitCriterion = (SplitCriterion) getPreparedClassOption(this.splitCriterionOption);
        AttributeSplitSuggestion[] bestSplitSuggestions = node.getBestSplitSuggestions(splitCriterion, this);
        Arrays.sort(bestSplitSuggestions);
        //int splitResult = 0;
        if (bestSplitSuggestions.length == 1) {
            this.boundSplits++;
            returnNode = this.splitNode(node, parent, parentIndex, bestSplitSuggestions[bestSplitSuggestions.length - 1]);
            splitResult[0] = 1;
        } else if (bestSplitSuggestions.length >= 2) {
            double criterionRange = splitCriterion.getRangeOfMerit(node.getObservedClassDistribution());
            double hoeffdingBound = computeHoeffdingBound(criterionRange,
                    this.splitConfidenceOption.getValue(), node.getWeightSeen());
            AttributeSplitSuggestion bestSuggestion = bestSplitSuggestions[bestSplitSuggestions.length - 1];
            AttributeSplitSuggestion secondBestSuggestion = bestSplitSuggestions[bestSplitSuggestions.length - 2];

            // just for debug
            //node.setBestSplitTest(new AttributeSplitSuggestion[]{bestSuggestion, secondBestSuggestion}, this);
            //double pred = node.getTop2CandidateMerit_diff();
            // just for debug

            if (adaptGracePeriod && this.splitTimePredictionOption.getChosenIndex() == 1) {
                node.setCandidateAttributeSplitSuggestions(bestSplitSuggestions, splitCriterion, this, hoeffdingBound, bestSuggestion.merit - secondBestSuggestion.merit);
            }

            if ((bestSuggestion.merit - secondBestSuggestion.merit > hoeffdingBound)
                    || hoeffdingBound < this.tieThresholdOption.getValue()) {
                if (hoeffdingBound < this.tieThresholdOption.getValue()) {
                    this.maxSplits++;
                    splitResult[0] = 2;
                    this.tieSplitNumSamples.addToValue(this.tieSplitNumSamples.numValues(), node.getWeightSeen() - node.initialWeight);
                }
                else {
                    this.boundSplits++;
                    this.boundSplitNumSamples.addToValue(this.boundSplitNumSamples.numValues(), node.getWeightSeen() - node.initialWeight);
                    splitResult[0] = 1;
                }
                returnNode = this.splitNode(node, parent, parentIndex, bestSuggestion);
            } else {
                if (adaptGracePeriod) {
                    double maxN = getHoeffdingN(criterionRange, splitConfidenceOption.getValue(), tieThresholdOption.getValue()) - node.getWeightSeen();
                    double gracePeriod = this.gracePeriodOption.getValue();
                    if (this.splitTimePredictionOption.getChosenIndex() == 2) {
                        gracePeriod = splitTimePredictionCGD(bestSuggestion, secondBestSuggestion, criterionRange, splitConfidenceOption.getValue(), node.getWeightSeen(), tieThresholdOption.getValue(), gracePeriodOption.getValue());
                    } else if (this.splitTimePredictionOption.getChosenIndex() == 3) {
                        gracePeriod = splitTimePredictionOSM(bestSuggestion, secondBestSuggestion, criterionRange, splitConfidenceOption.getValue(), node.getWeightSeen(), tieThresholdOption.getValue(), gracePeriodOption.getValue());
                    } else if (this.splitTimePredictionOption.getChosenIndex() == 4) {
                        gracePeriod = rand.nextInt(MAX_STEPS - this.gracePeriodOption.getValue() + 1) + this.gracePeriodOption.getValue();
                    }
                    node.gracePeriod = (int) Math.ceil(Math.min(maxN, gracePeriod)); //选择两者之间的最小值

                    // just for debug
                    //double real = bestSuggestion.merit - secondBestSuggestion.merit;
                    //System.out.println( real + " ==> " + pred + " ==> " +(real - pred) + " bound " + hoeffdingBound);
                    // just for debug
                }

                if ((this.removePoorAttsOption != null)
                        && this.removePoorAttsOption.isSet()) {
                    Set<Integer> poorAtts = new HashSet<Integer>();
                    // scan 1 - add any poor to set
                    for (int i = 0; i < bestSplitSuggestions.length; i++) {
                        if (bestSplitSuggestions[i].splitTest != null) {
                            int[] splitAtts = bestSplitSuggestions[i].splitTest.getAttsTestDependsOn();
                            if (splitAtts.length == 1) {
                                if (bestSuggestion.merit
                                        - bestSplitSuggestions[i].merit > hoeffdingBound) {
                                    poorAtts.add(new Integer(splitAtts[0]));
                                }
                            }
                        }
                    }
                    // scan 2wwwwe - remove good ones from set
                    for (int i = 0; i < bestSplitSuggestions.length; i++) {
                        if (bestSplitSuggestions[i].splitTest != null) {
                            int[] splitAtts = bestSplitSuggestions[i].splitTest.getAttsTestDependsOn();
                            if (splitAtts.length == 1) {
                                if (bestSuggestion.merit
                                        - bestSplitSuggestions[i].merit < hoeffdingBound) {
                                    poorAtts.remove(new Integer(splitAtts[0]));
                                }
                            }
                        }
                    }
                    for (int poorAtt : poorAtts) {
                        node.disableAttribute(poorAtt);
                    }
                }
            }
        }
        this.attemptToSplitTime += System.currentTimeMillis() - startTime;
        return returnNode;
    }

    public void enforceTrackerLimit() {
        if ((this.inactiveLeafNodeCount > 0)
                || ((this.activeLeafNodeCount * this.activeLeafByteSizeEstimate + this.inactiveLeafNodeCount
                * this.inactiveLeafByteSizeEstimate)
                * this.byteSizeEstimateOverheadFraction > this.maxByteSizeOption.getValue())) {
            if (this.stopMemManagementOption.isSet()) {
                this.growthAllowed = false;
                return;
            }
            FoundNode[] learningNodes = findLearningNodes();
            Arrays.sort(learningNodes, new Comparator<FoundNode>() {

                @Override
                public int compare(FoundNode fn1, FoundNode fn2) {
                    return Double.compare(fn1.node.calculatePromise(), fn2.node.calculatePromise());
                }
            });
            int maxActive = 0;
            while (maxActive < learningNodes.length) {
                maxActive++;
                if ((maxActive * this.activeLeafByteSizeEstimate + (learningNodes.length - maxActive)
                        * this.inactiveLeafByteSizeEstimate)
                        * this.byteSizeEstimateOverheadFraction > this.maxByteSizeOption.getValue()) {
                    maxActive--;
                    break;
                }
            }
            int cutoff = learningNodes.length - maxActive;
            for (int i = 0; i < cutoff; i++) {
                if (learningNodes[i].node instanceof ActiveLearningNode) {
                    deactivateLearningNode(
                            (ActiveLearningNode) learningNodes[i].node,
                            learningNodes[i].parent,
                            learningNodes[i].parentBranch);
                }
            }
            for (int i = cutoff; i < learningNodes.length; i++) {
                if (learningNodes[i].node instanceof InactiveLearningNode) {
                    activateLearningNode(
                            (InactiveLearningNode) learningNodes[i].node,
                            learningNodes[i].parent,
                            learningNodes[i].parentBranch);
                }
            }
        }
    }

    public void estimateModelByteSizes() {
        FoundNode[] learningNodes = findLearningNodes();
        long totalActiveSize = 0;
        long totalInactiveSize = 0;
        double time2 = System.currentTimeMillis();

        for (FoundNode foundNode : learningNodes) {
            if (foundNode.node instanceof ActiveLearningNode) {
                totalActiveSize += SizeOf.fullSizeOf(foundNode.node);
            } else {
                totalInactiveSize += SizeOf.fullSizeOf(foundNode.node);
            }
        }
        double time3 = System.currentTimeMillis();
        this.enforeModelTime += time3 - time2;

        if (totalActiveSize > 0) {
            this.activeLeafByteSizeEstimate = (double) totalActiveSize
                    / this.activeLeafNodeCount;
        }
        if (totalInactiveSize > 0) {
            this.inactiveLeafByteSizeEstimate = (double) totalInactiveSize
                    / this.inactiveLeafNodeCount;
        }

        int actualModelSize = this.measureByteSize();
        double estimatedModelSize = (this.activeLeafNodeCount
                * this.activeLeafByteSizeEstimate + this.inactiveLeafNodeCount
                * this.inactiveLeafByteSizeEstimate);
        this.byteSizeEstimateOverheadFraction = actualModelSize
                / estimatedModelSize;

        if (actualModelSize > this.maxByteSizeOption.getValue()) {
            enforceTrackerLimit();
        }
    }

    public void deactivateAllLeaves() {
        FoundNode[] learningNodes = findLearningNodes();
        for (int i = 0; i < learningNodes.length; i++) {
            if (learningNodes[i].node instanceof ActiveLearningNode) {
                deactivateLearningNode(
                        (ActiveLearningNode) learningNodes[i].node,
                        learningNodes[i].parent, learningNodes[i].parentBranch);
            }
        }
    }

    protected LearningNode deactivateLearningNode(ActiveLearningNode toDeactivate,
                                                  SplitNode parent, int parentBranch) {
        LearningNode newLeaf = new InactiveLearningNode(toDeactivate.getObservedClassDistribution(), toDeactivate);
        if (parent == null) {
            this.treeRoot = newLeaf;
        } else {
            parent.setChild(parentBranch, newLeaf);
        }
        this.activeLeafNodeCount--;
        this.inactiveLeafNodeCount++;
        return newLeaf;
    }

    protected void activateLearningNode(InactiveLearningNode toActivate,
                                        SplitNode parent, int parentBranch) {
        Node newLeaf = newLearningNode(toActivate.getObservedClassDistribution());
        if (parent == null) {
            this.treeRoot = newLeaf;
        } else {
            parent.setChild(parentBranch, newLeaf);
        }
        this.activeLeafNodeCount++;
        this.inactiveLeafNodeCount--;
    }

    protected FoundNode[] findLearningNodes() {
        List<FoundNode> foundList = new LinkedList<FoundNode>();
        findLearningNodes(this.treeRoot, null, -1, foundList);
        return foundList.toArray(new FoundNode[foundList.size()]);
    }

    protected void findLearningNodes(Node node, SplitNode parent,
                                     int parentBranch, List<FoundNode> found) {
        if (node != null) {
            if (node instanceof LearningNode) {
                found.add(new FoundNode(node, parent, parentBranch));
            }
            if (node instanceof SplitNode) {
                SplitNode splitNode = (SplitNode) node;
                for (int i = 0; i < splitNode.numChildren(); i++) {
                    findLearningNodes(splitNode.getChild(i), splitNode, i,
                            found);
                }
            }
        }
    }

    public MultiChoiceOption leafpredictionOption = new MultiChoiceOption(
            "leafprediction", 'l', "Leaf prediction to use.", new String[]{
            "MC", "NB", "NBAdaptive"}, new String[]{
            "Majority class",
            "Naive Bayes",
            "Naive Bayes Adaptive"}, 0);

    public MultiChoiceOption splitTimePredictionOption = new MultiChoiceOption(
            "splitTimePrediction", 'a', "Grace period.", new String[]{
            "None","INC", "CGD", "OSM", "RND"}, new String[]{
            "No prediction",
            "Increment mode",
            "Constant gain difference",
            "One sided minimum",
            "Random"}, 1);

    public IntOption nbThresholdOption = new IntOption(
            "nbThreshold",
            'q',
            "The number of instances a leaf should observe before permitting Naive Bayes.",
            0, 0, Integer.MAX_VALUE);

    public IntOption incrementPeriodOption = new IntOption(
            "incrementPeriod",
            'i',
            "The number of new instances a leaf should observe before updating a candidate attribute infomation incremently.",
            1, 1, 10);

    public IntOption checkPeriodOption = new IntOption(
            "checkPeriod",
            'k',
            "The number of new instances a leaf should observe before checking candidate attributes.",
            200, 50, 1000);

    public static class LearningNodeNB extends ActiveLearningNode {

        private static final long serialVersionUID = 1L;

        public LearningNodeNB(double[] initialClassObservations) {
            super(initialClassObservations);
        }

        public LearningNodeNB(double[] initialClassObservations, int gracePeriod) {
            super(initialClassObservations, gracePeriod);
        }

        @Override
        public double[] getClassVotes(Instance inst, HoeffdingTreeMeasure ht) {
            if (getWeightSeen() >= ht.nbThresholdOption.getValue()) {
                return NaiveBayes.doNaiveBayesPrediction(inst,
                        this.observedClassDistribution,
                        this.attributeObservers);
            }
            return super.getClassVotes(inst, ht);
        }

        @Override
        public void disableAttribute(int attIndex) {
            // should not disable poor atts - they are used in NB calc
        }
    }

    public static class LearningNodeNBAdaptive extends LearningNodeNB {

        private static final long serialVersionUID = 1L;

        protected double mcCorrectWeight = 0.0;

        protected double nbCorrectWeight = 0.0;

        public LearningNodeNBAdaptive(double[] initialClassObservations, int adaptiveGracePeriod) {
            super(initialClassObservations, adaptiveGracePeriod);
        }

        public LearningNodeNBAdaptive(double[] initialClassObservations) {
            super(initialClassObservations);
        }

        @Override
        public void learnFromInstance(Instance inst, HoeffdingTreeMeasure ht) {
            int trueClass = (int) inst.classValue();
            if (this.observedClassDistribution.maxIndex() == trueClass) {
                this.mcCorrectWeight += inst.weight();
            }

            if (Utils.maxIndex(NaiveBayes.doNaiveBayesPrediction(inst,
                    this.observedClassDistribution, this.attributeObservers)) == trueClass) {
                this.nbCorrectWeight += inst.weight();
            }
            super.learnFromInstance(inst, ht);
        }

        @Override
        public double[] getClassVotes(Instance inst, HoeffdingTreeMeasure ht) {
            if (this.mcCorrectWeight > this.nbCorrectWeight) {
                return this.observedClassDistribution.getArrayCopy();
            }
            return NaiveBayes.doNaiveBayesPrediction(inst,
                    this.observedClassDistribution, this.attributeObservers);
        }
    }

    protected LearningNode newLearningNode() {
        return newLearningNode(new double[0]);
    }

    protected LearningNode newLearningNode(double[] initialClassObservations) {
        LearningNode ret;
        int predictionOption = this.leafpredictionOption.getChosenIndex();
        if (predictionOption == 0) { //MC
            ret = new ActiveLearningNode(initialClassObservations, this.gracePeriodOption.getValue());
        } else if (predictionOption == 1) { //NB
            ret = new LearningNodeNB(initialClassObservations, this.gracePeriodOption.getValue());
        } else { //NBAdaptive
            ret = new LearningNodeNBAdaptive(initialClassObservations, this.gracePeriodOption.getValue());
        }
        return ret;
    }
}
