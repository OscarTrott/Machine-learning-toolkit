/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machinelearning;

/**
 * A variation of a Gaussian Naive Bayes classifier, this classifier was not used in either of my kaggle submissions as I did not feel that its overall effectiveness was good enough
 * @author Owner
 */
public class Classifier_NB extends Classifier {
    double[] posCnnDev;
    double[] posCnnAverage;
    double[] posGistDev;
    double[] posGistAverage;
    double[] negCnnDev;
    double[] negCnnAverage;
    double[] negGistDev;
    double[] negGistAverage;
    double decWeight;
    double priorProb;
    
    /**
     * My implementation of a Naive Bayes classifier, unfortunately it is not a "proper" Naive Bayes as it does not make use of the prior probabilities, however I was getting ~70% accuracy
     * in cross validation tests and I didn't want to use it in the final submission anyway so I never got around to fixing it completely
     * @param decWeight_ 
     */
    public Classifier_NB(double decWeight_)
    {
        decWeight = decWeight_;
        priorProb = 5.714;
    }
    
    @Override
    public void train(Instance[] instances) {
        //These values are all deduced from the training set of instances
        posCnnDev = new double[4096]; //Holds the deviation of the CNN values attained from all instances with a positive class classification
        posCnnAverage = new double[4096]; //Holds the average of the CNN values attained from all instances with a positive class classification
        posGistDev = new double[512]; //Holds the deviation of the gist values attained from all instances with a positive class classification
        posGistAverage = new double[512]; //Holds the average of the gist values attained from all instances with a positive class classification
        negCnnDev = new double[4096]; //Holds the deviation of the CNN values attained from all instances with a negative class classification
        negCnnAverage = new double[4096]; //Holds the average of the CNN values attained from all instances with a negative class classification
        negGistDev = new double[512]; //Holds the deviation of the gist values attained from all instances with a negative class classification
        negGistAverage = new double[512]; //Holds the average of the gist values attained from all instances with a negative class classification
        
        //Iterate through each feature index and set the various variables above for each
        for (int iter = 0; iter < instances[0].features.length; iter++)
            standardDev(instances,iter);
    }

    @Override
    public double classify(Instance i) {
        int classified = 0;
        double prob1 = 0;
        double prob0 = 0;
        double count = 0;
        
        Feature[] posCnnAverageInstance = new Feature[posCnnAverage.length + posGistAverage.length];
        Feature[] negCnnAverageInstance = new Feature[posCnnAverage.length + posGistAverage.length];
        for (int iter = 0; iter < posCnnAverage.length; iter++)
        {
            posCnnAverageInstance[iter] = new Feature(FeatureType.CNN, posCnnAverage[iter]);
            negCnnAverageInstance[iter] = new Feature(FeatureType.CNN, negCnnAverage[iter]);
        }
        for (int iter = 0; iter < posGistAverage.length; iter++)
        {
            posCnnAverageInstance[iter+4096] = new Feature(FeatureType.GIST, posGistAverage[iter]);
            negCnnAverageInstance[iter+4096] = new Feature(FeatureType.GIST, negGistAverage[iter]);
        }
        for (int iter = 0; iter< i.features.length; iter++)
        {
            if (i.features[iter].getVal() != 0){
            if (i.features[iter].getType() == FeatureType.CNN)
            {
                double t1 = 1/Math.pow(2*Math.PI*Math.pow(posCnnDev[iter],2), 0.5);
                double t3 = 2*Math.pow(posCnnDev[iter],2);
                double t4 = Math.pow(i.features[iter].getVal() - posCnnAverageInstance[iter].getVal(), 2);
                double t2 = Math.exp(- 1 * (t4/t3));
                prob1 = t1 * t2;
            }
            else prob1 = 1/Math.pow(2*Math.PI*Math.pow(posGistDev[iter-4096],2), 0.5) * Math.exp( - 0.5*(Math.pow(i.features[iter].getVal() - posCnnAverageInstance[iter].getVal(),2)/Math.pow(posGistDev[iter-4096],2)));
            if (i.features[iter].getType() == FeatureType.CNN)  
            {
                double t1 = 1/Math.pow(2*Math.PI*Math.pow(negCnnDev[iter],2), 0.5);
                double t3 = 2*Math.pow(negCnnDev[iter],2);
                double t4 = Math.pow(i.features[iter].getVal() - negCnnAverageInstance[iter].getVal(), 2);
                double t2 = Math.exp(-(t4/t3));
                prob0 = t1*t2;
            }
            else prob0 = 1/Math.pow(2*Math.PI*Math.pow(negGistDev[iter-4096],2), 0.5)*Math.exp( - 0.5*(Math.pow(i.features[iter].getVal() - negCnnAverageInstance[iter].getVal(),2)/Math.pow(negGistDev[iter-4096],2)));
            prob1 = Math.pow(Math.log(prob1),2); prob0 = Math.pow(Math.log(prob0),2);
            if (prob1>decWeight*prob0) classified++;
            count++;
            }
        
        }
        
        
        
        classified = classified<count/2 ? 0 : 1;
        
        return classified;
    }
    
    /**
     * Sets the deviation and average arrays index for specified by the parameter feature
     * @param i The array of training instances from which the statistics will be drawn from
     * @param feature The index of feature whose deviation and mean is currently being found
     */
    private void standardDev(Instance[] i, int feature)
    {
        //Sets initial values
        double posDev = 0;
        double posMean = 0;
        double posCnnTot = 0;
        double posGistTot = 0;
        int posCount = 0;
        double negDev = 0;
        double negMean = 0;
        double negCnnTot = 0;
        double negGistTot = 0;
        int negCount = 0;
        
        //Iterate over the instances and update the values as required
        for (Instance instance : i)
        {
            //if (instance.features[feature].getVal() != 0) 
            {
                posDev += Math.pow(instance.features[feature].getVal() - posMean, 2);
                negDev += Math.pow(instance.features[feature].getVal() - negMean, 2);
                if (instance.get_Class() == 1)
                {
                    if (instance.features[feature].getType() == FeatureType.CNN) 
                        posCnnTot += instance.features[feature].getVal();
                    else 
                        posGistTot += instance.features[feature].getVal();
                    posMean += instance.features[feature].getVal();
                    if (instance.features[feature].getVal() != 0) 
                        posCount++;
                }
                else
                {
                    if (instance.features[feature].getType() == FeatureType.CNN) 
                        negCnnTot += instance.features[feature].getVal();
                    else 
                        negGistTot += instance.features[feature].getVal();
                    negMean += instance.features[feature].getVal();
                    if (instance.features[feature].getVal() != 0) 
                        negCount++;
                }
            }
        }
        posMean /= posCount; 
        negMean /= negCount; 
        
        //Sets the relevant arrays values
        if (i[0].features[feature].getType() == FeatureType.CNN) 
        {
            posCnnAverage[feature] = posCnnTot / 4096;
            negCnnAverage[feature] = negCnnTot / 4096;
        }
        else 
        {
            negGistAverage[feature-4096] = negGistTot / 512;
            posGistAverage[feature-4096] = posGistTot / 512;
        }
        
        
        posDev /= posCount;
        posDev = Math.pow(posDev,0.5);
        
        negDev /= negCount;
        negDev = Math.pow(negDev,0.5);
        
        if (i[0].features[feature].getType() == FeatureType.CNN) posCnnDev[feature] = posDev;
        else posGistAverage[feature-4096] = posGistDev[feature-4096] = posDev;
        if (i[0].features[feature].getType() == FeatureType.CNN) negCnnDev[feature] = negDev;
        else posGistAverage[feature-4096] = negGistDev[feature-4096] = negDev;
    }
    
}
