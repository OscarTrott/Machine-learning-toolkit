/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machinelearning;

import java.util.Arrays;
import java.util.Queue;

/**
 * Decision tree classifier, creates a decision tree using the training data, classifies instances based on the path they take through the constructed decision tree
 * @author Owner
 */
public class Classifier_DT extends Classifier{
    Instance[] instances;
    DT_Component decisionTreeRoot;
    Feature[] features;
    boolean[] added;
    double dt_Boundary;
    double bias;

    /**
     * Creates a decision tree classifier with the given parameters
     * @param dt_Boundary_ The decision boundary used within each node to determine which child it should pass the instance on to
     * @param bias_ Determines the bias given to negative values when determining which child is positive and which is negative
     */
    public Classifier_DT(double dt_Boundary_, double bias_) {
        dt_Boundary = dt_Boundary_;
        bias = bias_;
    }
    
    @Override
    public void train(Instance[] instances) {
        this.instances = instances.clone();
        added = new boolean[instances[0].features.length]; //Creates an array of booleans, this array says which features have been added to the decision tree already
        Arrays.fill(added, 0, added.length-1, false);
        binarise(instances); //Converts anything above 0 to one
        
        //Create the root node with the found feature
        decisionTreeRoot = new DT_Node(instances, added, nextFeature(), dt_Boundary, bias);
    }
    
    /**
     * Finds the index of the next feature with the smallest impurity and which has not yet been added to a node previously
     * @return Index of feature with lowest impurity
     */
    public int nextFeature()
    {
        int at = 0; //Holds the index of the feature with the smallest impurity
        double minImpurity = Double.POSITIVE_INFINITY;
        
        //Finds the next feature with the lowest impurity
        for (int i = 0; i < instances[0].features.length; i++)
        {
            //Checks whether the feature being iterated over has already been added
            if (!added[i]){
                minImpurity = impurity(instances, i)<minImpurity?impurity(instances, i):minImpurity;
                at = impurity(instances, i)<minImpurity?i:at;
            }
        }
        
        return at;
    }
    
    /**
     * Calculates the impurity for the given feature across all given instances, the impurity of a feature measures how split the instance classes are across that features values
     * @param instances Instances over which the impurity should be calculated
     * @param feature The feature for which the impurity should be calculated
     * @return A double value representing the impurity of the feature
     */
    public double impurity(Instance[] instances, int feature)
    {
        double class1 = 0; //Counts how many of the given feature in instances equals 1
        double class2 = 0; //Counts how many of the given feature in instances equals 0
        double true1 = 0; //Counts how many instances of the given feature=1 are class 1
        double true2 = 0; //Counts how many instances of the given feature=0 are class 0
                
        double result = 0; //The value which will be updated and returned
        
        //Iterate over instances and increment relevant variables
        for (Instance i : instances)
        {
            if (i.features[feature].getVal()==1) 
            {
                if (i.get_Class() == 1) true1++;
                class1++;
            }
            else 
            {
                if (i.get_Class() == 0) true2++;
                class2++;
            }
        }
        
        //Components of equation are split for debugging purposes
        double t1 = Math.log(true1/(double)class1)/Math.log(2); //Replicates log2(true1/class1)
        double t2 = Math.log(true2/(double)class2)/Math.log(2); //Replicates log2(true2/class2)
        
        double t3 = (class1/instances.length);
        double t4 = -(true1/(class1>0?class1:Double.POSITIVE_INFINITY))*t1; //If class1 == 0 use positive infinity, otherwise dvide by zero
        
        double t5 = (class2/instances.length);
        double t6 = -(true2/(class2>0?class2:Double.POSITIVE_INFINITY))*t2; //If class2 == 0 use positive infinity, otherwise dvide by zero
        
        result = t3 * t4 + t5 * t6; //(class1/n)*(-true1/class1)*log2(true1/class1)+(class2/n)*(-true2/class2)*log2(true2/class2)
        
        return result;
    }
    
    @Override
    public double classify(Instance i) {
        binarise(i);
        return decisionTreeRoot.poll(i.features);
    }
    
    /**
     * Converts the value of all features of the given instance into either 1 or 0 depending on their current value
     * @param instances The instances which should be converted
     */
    private void binarise(Instance instances)
    {
        //Iterate over all features and binarise them each in turn
        for (Feature f : instances.features)
        {
            double val = f.getVal();
            f.value = val>0?1:0;
            val = f.getVal();
        }
    }
    
    /**
     * Converts the value of all features of the given instances into either 1 or 0 depending on their current value
     * @param instances The instances which should be converted
     */
    private void binarise(Instance[] instances)
    {
        //Iterates over each feature in each instances and convert it
        for (Instance i : instances)
        {
            for (Feature f : i.features)
            {
                double val = f.getVal();
                f.value = val>0?1:0;
                val = f.getVal();
            }
        }
    }
}
