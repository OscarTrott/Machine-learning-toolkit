/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machinelearning;

import java.util.Arrays;

/**
 * Acts as an internal node within the decision tree, has a left and a right child which can be either another internal or external node
 * @author Owner
 */
public class DT_Node extends DT_Component{
    private DT_Component left; //Left child
    private DT_Component right; //Right child
    int feature; //Feature this node splits on
    boolean positive; //Determines which of the left or right children the instance to be classified is passed to

    /**
     * Represents an internal node of a decision tree classifier
     * @param instances_ The array of instances from which the node should be constructed
     * @param added_ An array of booleans saying what features have already been used within the 
     * @param feature_ The index of the array which this node will split on
     * @param dt_Boundary The decision boundary for how pure the instances must be to create a hard decision boundary such that ALL test instances go down to one child
     * @param bias The bias for negative class feature value matching
     */
    public DT_Node(Instance[] instances_, boolean[] added_, int feature_, double dt_Boundary, double bias) {
        super(instances_,added_, dt_Boundary);
        feature = feature_;
        int next = nextFeature(); //Finds the next feature with the lowest impurity
        Instance[] leftInstances = getLeftInstances(); //Determines instances which will be passed to the left child
        Instance[] rightInstances = getRightInstances(); //Determines instances which will be passed to the right child
        int true_ = 0;
        int false_ = 0;
        added[feature_] = true;
        
        //If the impurity has been reduced to within reasonable limits or there is only one feature left to be split on, add leaf nodes as both children
        if (impurity(instances, next)<0.1||countNotAdded() == 1){
            add_Left(new DT_Leaf(leftInstances, added, next, dt_Boundary, bias));
            next = nextFeature();
            add_Right(new DT_Leaf(rightInstances, added, next, dt_Boundary, bias));
        }
        else
        {
            //If there are any instances in the left instances array then add a node, otherwise add a leaf
            if (leftInstances.length>0) 
                add_Left(new DT_Node(leftInstances, added, next, dt_Boundary, bias));
            else 
                add_Left(new DT_Leaf(leftInstances, added, next, dt_Boundary, bias));
            
            //If there are any instances in the right instances array then add a node, otherwise add a leaf
            if (rightInstances.length>0) 
                add_Right(new DT_Node(rightInstances, added, next, dt_Boundary, bias));
            else 
                add_Right(new DT_Leaf(rightInstances, added, next, dt_Boundary, bias));
        }
        
        //Iterate over all instances given and count how many of the specified feature is the same as their class label
        for (Instance i : instances)
        {
            if ((i.features[feature_].getVal()==1&&i.get_Class()==1)||(i.features[feature_].getVal()==0&&i.get_Class()==0)) true_++;
            else if ((i.features[feature_].getVal()==1&&i.get_Class()==0)||(i.features[feature_].getVal()==0&&i.get_Class()==1)) false_++;
        }
        positive = true_>bias*false_; //Sets which value should be returned if the features positive
        
        //Sets the position of the boundary over which the specific feature from the test instances will be compared
        if (true_/instances.length>dt_Boundary) 
            boundary = -0.5;
        else if (false_/instances.length<1-dt_Boundary) 
            boundary = 1.5;
        else 
            boundary = 0.5;
    }
    
    /**
     * Add the given decision tree component as the left child
     * @param l The component to be added
     */
    public void add_Left(DT_Component l)
    {
        left = l;
    }
    
    /**
     * Add the given decision tree component as the right child
     * @param r The component to be added
     */
    public void add_Right(DT_Component r)
    {
        right = r;
    }
    
    /**
     * Gets an array of the instances which fulfill the left side requirement, uses the instances given at object creation time
     * @return The array of instances which is a subset of the training instances
     */
    public final Instance[] getLeftInstances()
    {
        int count = 0;
        for (int i = 0; i < instances.length; i++) if (instances[i].features[feature].getVal()>0.5) count++;
        Instance[] result = new Instance[count];
        int pointer = 0;
        for (int i = 0; i < instances.length; i++) if (instances[i].features[feature].getVal()>0.5) result[pointer++] = instances[i];
        return result;//Arrays.copyOfRange(instances, 0, (int)(instances.length/1.5));
    }
    
    /**
     * Gets an array of the instances which fulfill the right side requirement, uses the instances given at object creation time
     * @return The array of instances which is a subset of the training instances
     */
    public final Instance[] getRightInstances()
    {
        int count = 0;
        for (int i = 0; i < instances.length; i++) if (instances[i].features[feature].getVal()<=0.5) count++;
        Instance[] result = new Instance[count];
        int pointer = 0;
        for (int i = 0; i < instances.length; i++) if (instances[i].features[feature].getVal()<=0.5) result[pointer++] = instances[i];
        return result;// Arrays.copyOfRange(instances, instances.length/, instances.length-1);
    }
    
    /**
     * Classifies an input instance
     * @param input The instance to classify
     * @return The classification result from polling with the given instance
     */
    @Override
    public double poll(Feature[] input) {
        double i = input[feature].getVal();
        double t = 0;
        
        //Chooses which child the node should send the instance to
        if (positive)
            if (i>boundary)
                t = left.poll(input);
            else
                t = right.poll(input);
        else
            if (i<boundary)
                t = right.poll(input);
            else
                t = left.poll(input);
            
        return t;
    }
    
    
}
