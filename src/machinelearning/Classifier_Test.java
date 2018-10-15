/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machinelearning;

/**
 * This is a test classifier used to evaluate the overall system being used to simulate the entire program, it outputs a value either set, generally one, or something which counts the 
 * number of features which equal zero
 * @author Owner
 */
public class Classifier_Test extends Classifier {

    @Override
    public void train(Instance[] instances) {
        //Do nothing
    }

    @Override
    public double classify(Instance i) {
        int count = 0;
        for (Feature f : i.features)
        {
            if (f.getVal()==0)count++;
        }
        return count>i.features.length/2.0?1:0;
    }
    
}
