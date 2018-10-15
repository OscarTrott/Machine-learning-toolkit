/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machinelearning;

/**
 * Determines the required methods and variables required for a classifier to be implemented
 * @author Owner
 */
public abstract class Classifier {
    
    /**
     * Train the classifier with the given training instances, it assumes that the instances have an associated class label
     * @param instances The array of instances with which the classifier will be trained with, If this is called multiple times the classifier will only be trained with the last call of the method
     */
    public abstract void train(Instance[] instances);
    
    /**
     * Classifies the single passed instance into class 1 or 2
     * @param i The passed instance which is to be classified
     * @return A double value representing the faith with which it was classified, 1 -> 100% class 1, 0 -> 100% class 0, 0.5 -> could be either
     */
   public abstract double classify(Instance i);
   
   /**
    * Classifies all instances passed in, outputs an array of doubles where the index of one element of the output relates to the same index of the input
    * @param instance The instances which are to be classified
    * @return The double array containing the classified confidence of each instance classified
    */
    public double[] classifyAll(Instance[] instance)
    {
        double[] results = new double[instance.length];
        int count = 0;
        
        //Classify all instances and add the result to the array
        for (Instance i : instance)
        {
            results[count++] = classify(i);
        }
        
        return results;
    }
}
