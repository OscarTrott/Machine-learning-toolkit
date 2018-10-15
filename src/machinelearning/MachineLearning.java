/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machinelearning;

//Imports
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * This class handles the general flow of the program
 * @author Owner
 */
public class MachineLearning {
    Classifier[] classifiers;
    File inputFeatures;
    File testingCSV;
    File annotationConfidenceFile;
    Instance[] trainingInstances;
    Instance[] testingInstances;
    Annotation[] annotationConfidence;
    double[][] classifications;
    PrintWriter pr;
    double accuracy;
    
    /**
     * Starts the program and can be set such that iteration across different input parameters can be done
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        MachineLearning c = new MachineLearning();
        double a = 0; //Current best accuracy produced by any iteration of the program
        
        //Parameters at a
        double decWeight_ = 0; 
        double k_ = 0;
        double gist_ = 0;
        double dec_ = 0;
        double learn_ = 0;
        double b_ = 0;
        double dt_Boundary_ = 0;
        
        //for (int k = 30; k < 41; k++) //Iterate over KNN k
            //for (double gistWeight = 0.2; gistWeight < 1.0; gistWeight += 0.1) //Iterate over gistWeighting for KNN
                //for (double decisionBoundary = 0.4; decisionBoundary < 0.61; decisionBoundary += 0.01) //Iterate over decision boundary for KNN
                    //for (double learningRate = 0.11; learningRate < 0.15; learningRate += 0.01) //Learning rate for SVM
                        //for (double decWeight = 0.8; decWeight < 1.3; decWeight += 0.05) //Decision weight for NB
                            //for (double dt_Boundary = 3; dt_Boundary < 5.2; dt_Boundary += 0.11) //The decision boundary used within each node to determine which child it should pass the instance on to
                                //for (double b = 0; b <= 5; b += 1) //b value for SVM
                                //for (double bias = 3.5; bias < 4.5; bias += 0.1) //Bias used in each decision tree
                                {
                                    //Print out the values being used in this simulation iteration
                                    System.out.println("----------------------------------"); 
                                    System.out.println("dt_Boundary: "+/*dt_Boundary+*/", b: "+/*b+*/", decision weight: "+/*decWeight+*/", k: "+/*k+*/", decision boundary: "+/*decisionBoundary+*/", learning rate: "+/*learningRate+*/", gist weight: "/*+gistWeight*/);
                                    c.execute(args[0], args[1], 30, 1, 0.48, 0.12, 1.07, 1, 0.8, 3.88); //Run the simulatio with these parameters
                                    
                                    //Update best parameters
                                    if (c.getAccuracy()>a){ 
                                        a = c.getAccuracy();
                                        //decWeight_ = decWeight;
                                        //k_ = k;
                                        //gist_ = gistWeight;
                                        //dec_ = decisionBoundary;
                                        //learn_ = learningRate;
                                        //b_ = b;
                                        //dt_Boundary_ = dt_Boundary;
                                    }
                                }
        //Print out the best parameters found
        System.out.println("----------------------------------");
        System.out.println("detBoundary"+dt_Boundary_+"deviation weight: "+decWeight_+", k: "+k_+", decision boundary: "+dec_+", learning rate: "+learn_+", gist weight: "+gist_ +", accuracy: "+a+", b: "+b_);
    }
    
    /**
     * Executes the simulation with all given parameters and classifiers which have been set prior to runtime
     * @param training The file path of the training file
     * @param testing The file path of the testing file
     * @param k The k for the knn classifier
     * @param gistWeight The gist weight for knn classifier
     * @param decisionBoundary the decision boundary for the knn classifier
     * @param learningRate the learning rate for the SVM classifier
     * @param decWeight The decision weighting for the naive bayes classifier
     * @param b The b offset for the SVM classifier
     * @param dt_Boundary The dtBoundary for the RF classifier
     * @param bias The bias for the RF classifier
     */
    public void execute(String training, String testing, double k, double gistWeight, double decisionBoundary, double learningRate, double decWeight, double b, double dt_Boundary, double bias) {
        //Check if wwe are training and testing with the same data
        boolean crossVal = training.equals(testing);
        
        //Create testing, training and annotation confidence files
        inputFeatures = new File(training);
        testingCSV = new File(testing);
        annotationConfidenceFile = new File("annotation_confidence.csv");
        annotationConfidence = new Annotation[950];
        
        //Create parser instances for the training and testing data
        Parser trainingParser = new Parser(inputFeatures);
        Parser testingParser = new Parser(testingCSV);
        trainingParser.parse();
        testingParser.parse();
        
        //Create and fill the training and testing instances arrays
        trainingInstances = new Instance[trainingParser.get_Instances().size()];
        trainingInstances = trainingParser.get_Instances().toArray(trainingInstances);
        
        testingInstances = new Instance[testingParser.get_Instances().size()];
        testingInstances = testingParser.get_Instances().toArray(testingInstances);
        
        //Scale the data
        //scale(trainingInstances);
        //scale(testingInstances);
        
        //Parse the annotation confidence file and create an array containing the data
        BufferedReader br = null;
        try {
            br = new BufferedReader(new FileReader(annotationConfidenceFile));
        } catch (FileNotFoundException ex) {
            Logger.getLogger(MachineLearning.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        String line; //Holds the entire line read from the file
        String[] splitLine; //Holds the comma separated components of the reaad line
        int point = 0;
        try {
            while ((line=br.readLine())!=null)
            {
                splitLine = line.split(",");
                if (!splitLine[1].equals("confidence")) //Check if it's the first line, this was added as various versions of the file had been created by me and some had no first line
                    annotationConfidence[point++] = new Annotation(Integer.parseInt(splitLine[0]),Double.parseDouble(splitLine[1])); //Parse the instances number and the confidence associated with it
            }
        } catch (IOException ex) {
            Logger.getLogger(MachineLearning.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        //Set which classifiers are being used, can be varied prior to runtime
        classifiers = new Classifier[3];
        classifiers[0] = new Classifier_KNN(k, gistWeight, decisionBoundary, annotationConfidence);
        classifiers[1] = new Classifier_SVM(learningRate, b);
        //classifiers[0] = new Classifier_NB(decWeight);
        classifiers[2] = new Classifier_RF(dt_Boundary, 5, bias); 
        
        int classifier; //Holds which classifier is currently in use
        
        //Classify each test instance with each classifier implementation
        classifications = new double[classifiers.length][];
        if (crossVal) //Checks if we are performing 3-way cross validation
        {
            //Set the first classifier to be used
            classifier = 0;
            for (Classifier c : classifiers) {
                point = 0; //Reset the index of the instance being classified
                
                //Partition the data into three equal sets, if the total number is not divisible by three then the partitions will be uneven
                Instance[] partition1 = Arrays.copyOfRange(testingInstances, 0, Math.round(Math.round(testingInstances.length/3.0)));
                Instance[] partition2 = Arrays.copyOfRange(testingInstances, Math.round(Math.round(testingInstances.length/3.0)), Math.round(Math.round(testingInstances.length/1.5)));
                Instance[] partition3 = Arrays.copyOfRange(testingInstances, Math.round(Math.round(testingInstances.length/1.5)), testingInstances.length-1);
                
                //Create the array for this classifiers output
                classifications[classifier] = new double[testingInstances.length];
                
                //Train classifier on partitions 2 and 3
                c.train(mergeArrays(partition3,partition2));
                for (double d : c.classifyAll(partition1))
                    //Classify partition 1
                    classifications[classifier][point++] = d;
                
                //Train classifier on partition 1 and 3
                c.train(mergeArrays(partition1,partition3));
                for (double d : c.classifyAll(partition2))
                    //Classify partition 2
                    classifications[classifier][point++] = d;
                
                //Train classifier on partition 1 and 2
                c.train(mergeArrays(partition2,partition1));
                for (double d : c.classifyAll(partition3))
                    //Classify partition 3
                    classifications[classifier][point++] = d;
                
                classifier++;
            }
              
                
        } else {
            //Sets the first classifier to be used
            classifier = 0;
            classifications = new double[classifiers.length][];
            
            //Iterate over the classifier, training and classifying each with the training and testing data respectively
            for (Classifier c : classifiers) {
                c.train(trainingInstances);
                classifications[classifier++] = c.classifyAll(testingInstances);
            }
        }
        
        //Check which output is the majority out of all classifier executions
        int[] results = new int[classifications[0].length];
        double count; //Counts how many class one's there have been for a given test instance
        for (int a = 0; a < classifications[0].length; a++)
        {
            count = 0;
            
            //Increment count for each output of class 1
            for (double[] classification : classifications) {
                count += classification[a]>1?1:classification[a];
                //System.out.print(Math.round(Math.round(classification[a])));
            }

            int random = (int) Math.round(Math.random());
            double border = classifications.length/2.0;
            if (count > border) results[a] = 1; 
            else if (count == border) results[a] = random; //6
            else results[a] = 0;
            //System.out.println(":"+(int)results[a] + "  |  " + testingInstances[a].get_Class()); //Shows results for each classifier for each instance
            //System.out.println();
        }
        
        count = 0;
        int truePos = 0;
        int falsePos = 0;
        int falseNeg = 0;
        
        //Create the output file
        File output = new File("output.csv");
        try {
            pr = new PrintWriter(new FileWriter(output));
        } catch (IOException ex) {
            System.out.println(ex.getMessage());
        }
        
        //Print the results into a file and prints the precision and recall of this execution
        pr.println("ID,prediction");
        for (int in = 1; in <= classifications[0].length; in++)
        {
            pr.println("" + in + "," + results[in-1]);
            if (testingInstances[in-1].get_Class() == results[in-1] && testingInstances[in-1].get_Class() == 1)
            {
                truePos++;
                count++;
            }
            else if (testingInstances[in-1].get_Class() == results[in-1] && testingInstances[in-1].get_Class() == 0) count++;
            else if (testingInstances[in-1].get_Class() != results[in-1] && testingInstances[in-1].get_Class() == 1) falseNeg++;
            else falsePos++;
        }
        System.out.println("Precision: "+(truePos/((double)truePos+falsePos)));
        System.out.println("Recall: "+(truePos/((double)truePos+falseNeg)));
        System.out.println(count+"/"+classifications[0].length);
        
        pr.close();
        System.out.println(count/(double)classifications[0].length);
        accuracy = count/(double)classifications[0].length;
    }
    
    /**
     * Normalise the input instances features by setting them all to be between 0 and 1 relative to their features max value
     * @param instances The instances whose features are to be normalised
     */
    private void scale(Instance[] instances)
    {
        double cnnMax = 0;
        double gistMax = 0;
        
        for (int i = 0; i < instances[0].features.length; i++)
        {
            for (int j = 0; j < instances.length; j++)
            {
                if (instances[j].features[i].getType() == FeatureType.CNN)
                {
                    cnnMax = instances[j].features[i].getVal()>cnnMax?instances[j].features[i].getVal():cnnMax;
                }
                else
                {
                    gistMax = instances[j].features[i].getVal()>gistMax?instances[j].features[i].getVal():gistMax;
                }
            }
        }
        for (int j = 0; j < instances[0].features.length; j++)
        {
            for (int i = 0; i < instances.length; i++)
            {
                double t;
                if (instances[i].features[j].getType() == FeatureType.CNN)
                {
                    t = instances[i].features[j].value/cnnMax;
                    instances[i].features[j].value = t;//minimum cnn value is zero, (x-xMin)/(xMax-xMin) -> x/xMax
                }
                else
                {
                    t = instances[i].features[j].value/gistMax;
                    instances[i].features[j].value = t;
                }
            }
        }
    }
    
    /**
     * Returns the accuracy of the last run of the simulation
     * @return Last accuracy achieved
     */
    public double getAccuracy()
    {
        return accuracy;
    }
    
    /**
     * Merge the two input arrays into one and output that array, the two are merged by extending one with the other
     * @param i1 First array to be merged
     * @param i2 Second array to be merged
     * @return Output array containing the two input arrays
     */
    private Instance[] mergeArrays(Instance[] i1, Instance[] i2)
    {
        Instance[] result = new Instance[i1.length+i2.length];
        
        for (int i = 0; i < i1.length; i++)
            result[i] = i1[i];
        for (int i = i1.length; i < i1.length+i2.length; i++)
            result[i] = i2[i-i1.length];
        
        return result;
    }
}
