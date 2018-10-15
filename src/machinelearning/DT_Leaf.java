/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machinelearning;

/**
 * Represents a leaf of a decision tree, very similar to a node however it now outputs a value straight away when classifying, this value is determined in the same way that a node determines
 * which child to pass the instances on to
 * @author Owner
 */
public class DT_Leaf extends DT_Component{
    boolean positive;
    int feature;
    public DT_Leaf(Instance[] instances_, boolean[] added_, int feature_, double dt_Boundary, double bias) {
        super(instances_, added_, dt_Boundary);
        int true_ = 0;
        int false_ = 0;
        feature = feature_;
        
        //Iterate over all instances given and count how many of the specified feature is the same as their class label
        for (Instance i : instances)
        {
            if ((i.features[feature_].getVal()==1&&i.get_Class()==1)||(i.features[feature_].getVal()==0&&i.get_Class()==0)) 
                true_++;
            else if ((i.features[feature_].getVal()==1&&i.get_Class()==0)||(i.features[feature_].getVal()==0&&i.get_Class()==1)) 
                false_++;
        }
        //Sets which value should be returned if the features positive
        positive = true_>bias*false_;
        //Sets the boundary based on how pure the instances classes are
        if (instances.length>0)
        {
            if (true_/(double)instances.length>dt_Boundary) 
                boundary = -0.5;
            else if (false_/(double)instances.length<1-dt_Boundary) 
                boundary = 1.5;
            else 
                boundary = 0.5;
        }
        else
        {
            boundary = 0.5;
        }
    }
    
    @Override
    public double poll(Feature[] input) {
        double i = input[feature].getVal();
        double t = 0;
        if (positive)
            if (i>boundary)
                t = 1;
            else
                t = 0;
        else
            if (i<boundary)
                t = 1;
            else
                t = 0;
            
        return t;
    }
    
}
