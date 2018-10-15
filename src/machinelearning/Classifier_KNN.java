/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machinelearning;

/**
 * A K Nearest Neighbours classifier, extends the Classifier class
 * @author Owner
 */
public class Classifier_KNN extends Classifier{
    Instance[] training; //Holds the training data points
    double k; 
    double gistWeight; //Bias weighting for ggist features
    double decisionBoundary; //Default: 0.5714, based off of the prior probability for instances to be classed as class 1
    Annotation[] annotationConfidence; //The annotation confidence of each training instance
    
    @Override
    public void train(Instance[] instances) {
        training = instances;
        //k = Math.round(Math.round(Math.pow(instances.length, 0.5))); //Sets k to be equal to the sqaure root of the number of training instances, not used
    }
    
    /**
     * Creates a KNN classifier with the given parameters
     * @param k_ The number of neighbours a test instance should use to determine its own class
     * @param gistWeight_ The weighting bias given to gist features
     * @param decisionBoundary_ The decision boundary, the number which must be exceeded by the total value of the k nearest neighbours to the test instance
     * @param annotationConfidence_ An array of Annotations representing how reliable the training instance annotations are
     */
    public Classifier_KNN(double k_, double gistWeight_, double decisionBoundary_, Annotation[] annotationConfidence_)
    {
        annotationConfidence = annotationConfidence_;
        decisionBoundary = decisionBoundary_;
        k = k_;
        gistWeight = gistWeight_;
    }

    
    @Override
    public double[] classifyAll(Instance[] instances)
    {
        double[] results = new double[instances.length];
        Instance[] neighbours = new Instance[(int)k+1];
        int instance = 0;
        //For each instance check its knn
        for (Instance i : instances)
        {
            //Iterate over all instances in the training array, if it is closer than the furthest away instance in the neighbours array then add it to the array
            for (Instance d : training)
            {
                if (neighbours[neighbours.length-1] == null || distance(i,d) < distance(i,neighbours[neighbours.length-1])) 
                    neighbours = addInstance(neighbours, i, d);
            }
            int count = 0;
            
            //Get the count of the classes for the knn and evaluate the class of i
            for (int j = 0; j < neighbours.length; j++)
            {
                if (getConfidence(neighbours[j])!=1 && neighbours[j].get_Class() == 0)count += 0.33;
                else count += getConfidence(neighbours[j])*neighbours[j].get_Class();
            }
            
            results[instance++] = (count/(k*decisionBoundary)); //Returns the likelihood of the point belonging to the class
        }
        return results;
    }
    
    /**
     * Adds an instance to the neighbours array using euclidean distance to order them, maintains ordering
     * @param array The array to which the instance should be added
     * @param i The instance with which we are currently calculating the distance to
     * @param d The instance which is being added to the array, if this method is called then either this should be closer than the furthest away instance or there should be free space within the array
     * @return The neighbours array modified with the new addition
     */
    private Instance[] addInstance(Instance[] array, Instance i, Instance d)
    {
        int last = array.length-1; //Sets the first instance with which the given instance i is compared against as the last instance in the array
        
        //Iterate over the null entries in the array, only happens the first k times this method is called
        while (last>-1&&array[last] == null) last--;
        
        double dist1 = 0; //distance of instance1
        double dist2 = 0; //Distance of instance2
        
        //If the array is not empty then set the above distances
        if (last>=0)
        { 
            dist1 = distance(i, array[last]);
            dist2 = distance(i, d);
        }
        
        //While the distance of the instance i is lower than the current instance in the array move that instance up one spot in the array
        while (last>-1&& dist1 > dist2)
        {
            if (last != array.length-1) array[last+1] = array[last]; //If the current instance is not the last one in the array, move it up by one in the array
            last--;
            if (last != -1) dist1 = distance(i, array[last]); //If there is another instance to compare to, get its distance
        }
        //Add the new entry in its position
        array[last+1] = d;
        return array;
    }
    
    /**
     * Gets the euclidean distance between i1 and i2 giving a bias to the gist features based on the gist weighting parameter, note that the result is not square rooted, this is because 
     * the relative size of the distance between any two instances will not change even if both values are rooted, e.g 16^0.5 > 9^0.5 and 16 > 9, the relative size properties remain whether the two
     * have been rooted or not
     * @param i1 The first instance which will be compared
     * @param i2 The second instance which will be compared
     * @return A double value representing the distance between the two instances in n-dimensional space, where n is the number of features
     */
    private double distance(Instance i1, Instance i2)
    {
        if (i2 == null || i1 == null) return 0; //If either instance is null return 0
        
        double result = 0;
        double p = 2;
        
        //Iterate over all features and find E(x-y)^2
        for (int i = 0; i< i1.features.length && i<i2.features.length;i++)
        {
            if (i1.features[i].getType().equals(FeatureType.GIST)) result += Math.pow(gistWeight*(i1.features[i].getVal() - i2.features[i].getVal()), p);
            else result += Math.pow(i1.features[i].getVal() - i2.features[i].getVal(), p);
        }
        return result;
    }
    
    /**
     * Get the annotation confidence for the given instance, this is necessary due to how the splitting is done when cross validation is being employed
     * @param i The instance whose annotation confidence we require
     * @return The annotation confidence of i
     */
    private double getConfidence(Instance i)
    {
        for (int j = 0; j < annotationConfidence.length; j++)
        {
            if (i.get_instanceNum() == annotationConfidence[j].id) return annotationConfidence[j].confidence;
        }
        return 1;
    }
    
    @Override
    public double classify(Instance i) {
        //Not implemented in this classifier due to classifyAll being overwritten and it would only ever be called from that method
        throw new UnsupportedOperationException("Not supported yet.");
    }
}
