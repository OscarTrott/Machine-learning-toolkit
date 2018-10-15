/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machinelearning;

/**
 * Represents a single component: leaf or node, which is a part of a decision tree
 * @author Owner
 */
public abstract class DT_Component {
    Instance[] instances;
    double boundary;
    boolean[] added;
    public DT_Component(Instance[] instances_, boolean[] added_,double dt_Boundary)
    {
        added = added_;
        instances = instances_;
    }
    
    /**
     * Gets the index of the next feature with the lowest impurity and which has not yet been used in a previous node/leaf of the decision tree
     * @return index of next feature to be used to split the data
     */
    public final int nextFeature()
    {
        int at = 0;
        double minImpurity = Double.POSITIVE_INFINITY;
        //Finds the next feature with the lowest impurity
        for (int i = 0; i < instances[0].features.length; i++)
        {
            if (!added[i]){
                double imp = impurity(instances, i);
                if (imp<minImpurity){
                    minImpurity = imp; //Change this, they're the same
                    at = i;
                }
            }
        }
        return at;
    }
    
    /**
     * Calculates the impurity for the given feature over the given instances
     * @param instances instances over which the impurity should be calculated
     * @param feature index of the feature for which the impurity is calculated
     * @return value of the impurity for the feature over the instances
     */
    public final double impurity(Instance[] instances, int feature)
    {
        double class1 = 0;
        double class2 = 0;
        double true1 = 0;
        double false1 = 0;
        double true2 = 0;
        double false2 = 0;
                
        double result = 0;
        
        for (Instance i : instances)
        {
            if (i.features[feature].getVal()==1) 
            {
                if (i.get_Class() == 1) true1++;
                else false1++;
                class1++;
            }
            else 
            {
                if (i.get_Class() == 0) true2++;
                else false2++;
                class2++;
            }
        }
        
        double t1 = Math.log10(true1/(double)class1);
        double t2 = Math.log10(true2/(double)class2);
        
        double t3 = (class1/instances.length);
        double t4 = -(true1/(class1>0?class1:Double.POSITIVE_INFINITY))*t1;
        
        double t5 = (class2/instances.length);
        double t6 = -(true2/(class2>0?class2:Double.POSITIVE_INFINITY))*t2;
        
        result = t3 * t4 + t5 * (t6);
        
        return result;
    }
    
    /**
     * Count how many features have so far been used to split the data
     * @return The number of features which have not been added
     */
    public final int countNotAdded()
    {
        int count = 0;
        for (int i = 0; i < added.length; i++) 
            count += added[i]?0:1;
        return count;
    }
    /**
     * Returns the class to which the input features best matches
     * @param input array of features from an instances
     * @return double representing the class which it has been determined to belong to
     */
    public abstract double poll(Feature[] input);
}
