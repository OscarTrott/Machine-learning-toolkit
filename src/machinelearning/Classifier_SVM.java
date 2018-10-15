/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machinelearning;

/**
 * A form of Support Vector Machine implementation, does not use sub-gradient descent, calculates the gradient for each feature and combines them all into a single gradient which is then used
 * to update the weights
 * @author Owner
 */
public class Classifier_SVM extends Classifier{
    double learningRate; //Rate at which the hyperplane is learned
    double[] w; //Weight vector
    double b; //Vector origin offset
    double lambda; //Hinge loss regularisation bias
    
    /**
     * Creates a Support vector machine with the given parameters
     * @param learningRate_ The learning rate, how fast the SVm learns from the gradient
     * @param b_ The offset from the origin for w
     */
    public Classifier_SVM(double learningRate_, double b_)
    {
        b = b_;
        learningRate = learningRate_;
        lambda = 0.0;
    }
    
    @Override
    public void train(Instance[] instances) {
        //Setting all weight values to be default of one
        w = new double[instances[0].features.length];
        double[] gradient = new double[w.length]; //Gradient vector
        boolean converged = false;
        
        //Set the initial weight vector to be one in all features
        for (int i = 0; i < w.length; i++)
        {
            w[i] = 1;
        }
        int iteration = 0;
        
        //Iterate until convergence or the limit is hit, mostly the limit is hit before convergence
        while (!converged && iteration++ < 100) //685
        {
            converged = true;
            
            //Find the gradient for each instance which the weight vector needs to be changd by
            for (int inst = 0; inst < instances.length; inst++)
            {
                //Update the gradient vectors values
                for (int i = 0; i < w.length; i++)
                {
                    double g = hingeLoss(instances[inst], i);
                    int class_ = -(instances[inst].get_Class()==1?1:-1); //Invert the value outputted by the hinge loss based on the instances class
                    gradient[i] = class_*g;
                }
                
                //Update the weight of each feature from the gradient vector
                for (int i = 0; i < gradient.length; i++)
                {
                    //If the gradient of feature i is not zero then update the weight vector
                    if (gradient[i]!=0) {
                        converged = false; //A change was made so the system has not converged
                        double temp = gradient[i]*learningRate;
                        w[i] += temp; //Update this features weight
                    }
                }
            }
        }
    }
    
    /**
     * The hinge loss function is used to provide the gradient needed to be catered to which creates a self reinforcing system, the gradient is squared to make the size of the gradient
     * more proportional to the distance how badly it is currently working, e.g the further wrong a point is the more drastic a change is needed to be made
     * @param x The training instance being used to estimate the SVM's current best effort, if it's on the wrong side then the returned value will be the gradient needed to be used to self-correct
     * @param feature The feature whose current weighting we are evaluating
     * @return The gradient needed to be used to converge towards the correct weight vector
     */
    private double hingeLoss(Instance x, int feature)
    {
        double dot = Math.pow(w[feature]-x.features[feature].getVal(),2);
        return max(0, 1-dot);
    }
    
    @Override
    public double classify(Instance i) {
        double result;
        double t = 1/(dotProduct(w,i.features)-b);
        result = t >= 0 ? 0 : 1;
        return result;
    }

    /**
     * Finds the dot product of two vectors of differing type
     * @param f1 array of doubles
     * @param f2 array of feature
     * @return the dot product of f1 and f2
     */
    private double dotProduct(double[] f1, Feature[] f2)
    {
        double result = 0;
        
        for (int i = 0; i<f1.length ; i++)
            result += f1[i]*f2[i].getVal();
        return result;
    }
    
    /**
     * Returns the value of the max of the two parameters
     * @param d1 param1
     * @param d2 param2
     * @return max of d1 and d2
     */
    private double max(double d1, double d2)
    {
        return d1>d2 ? d1 : d2;
    } 
    
    /**
     * Could not be used in this implementation due to each feature being treated as an individual with no knowledge of the other features Although I do understand how it works and 
     * calculates the distance between two points based on using dot product in a higher dimensionality space which has been reduced into an equation, this is the Radial Basis Function kernel
     * @param i1 vector/point one
     * @param i2 vector/point two
     * @return The dot product of the two vectors/points if it was done in a higher dimensionality space
     */
    private double kernelFunction(Instance i1, Instance i2)
    {
        double result = 0;
        double sigma = 1;
   
        result = distance(i1, i2);
        result /= 2*Math.pow(sigma, 2);
        
        return Math.exp(-result);
    }
    
    /**
     * Euclidean distance between two vector/points
     * @param i1 vector/point one
     * @param i2 vector/point two
     * @return the euclidean distance between i1 and i2
     */
    private double distance(Instance i1, Instance i2)
    {
        double result = 0;
        
        for (int i = 0; i < i1.features.length; i++)
        {
            result += Math.pow(i1.features[i].getVal() - i2.features[i].getVal(), 2);
        }
        
        return Math.pow(result, 0.5);
    }    
}
