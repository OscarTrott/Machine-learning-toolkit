/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machinelearning;

import java.util.Arrays;

/**
 * Random forest classifier implementation
 * @author Owner
 */
public class Classifier_RF extends Classifier {
    Classifier_DT[] decisionTrees;
    int dtNum;
    boolean[] added;
    
    /**
     * Creates a Random Forest classifier with the parameters specified
     * @param dt_Boundary The boundary value that each decision tree will be set at
     * @param dtNum_ The number of decision trees which shall be created
     * @param bias The bias which shall be passed into each decision tree
     */
    public Classifier_RF(double dt_Boundary, int dtNum_, double bias)
    {
        dtNum = dtNum_;
        decisionTrees = new Classifier_DT[dtNum_];
        
        //Create the decision trees
        for (int i = 0; i < dtNum_; i++)
        {
            decisionTrees[i] = new Classifier_DT(dt_Boundary, bias);
        }
    }
    
    @Override
    public void train(Instance[] instances) {
        added = new boolean[instances.length];
        Arrays.fill(added, false);
        
        //Partition the data into random subsets
        Instance[][] partitions = new Instance[dtNum][];
        int partSize = instances.length;
        int count = partitions.length;
        
        //Initialise each partition
        for (int i = 0; i < partitions.length; i++)
        {
            partitions[i] = new Instance[partSize/count];
            partSize -= partSize/count--;
        }
        
        //Reverse the order of the partitions, due to the way they are filled with the training instances
        for (int i = 0; i < partitions.length/2; i++)
        {
            Instance[] t = partitions[i];
            partitions[i] = partitions[(partitions.length-1)-i];
            partitions[(partitions.length-1)-i] = t;
        }
        
        int point = 0;
        
        //While there are still instances to add to the partitions add one to the next partition
        while (stillToAdd())
        {
            addTo(instances,partitions[point]);
            point = (point+1)%partitions.length;
        }
        
        
        //Train each decision tree on its data partition
        for (int i = 0; i < dtNum; i++)
        {
            decisionTrees[i].train(partitions[i]); 
        }
    }
    
    /**
     * Adds a random instance from the addFrom array, which has not yet been added to a class, to the addTo array
     * @param addFrom The array from which we source a random instance
     * @param addTo The array which we add the random instance to
     * @return The addTo array with the new addition included
     */
    private Instance[] addTo(Instance[] addFrom, Instance[] addTo)
    {
        int p = 0;
        while (addTo[p]!=null) p++; //Find the first free cell of the addTo array
        boolean finished = false;
        int random;
        
        //Choose a random instance from the addFrom array and add it to the first free space in the addTo array
        while (!finished)
        {
            random = Math.round(Math.round(Math.random()*(added.length-1)));
            if (!added[random])
            {
                added[random] = true;
                finished = true;
                addTo[p] = addFrom[random];
            }
        }
        return addTo;
    }
    
    /**
     * Checks whether there are still instances which can be added to the partitions
     * @return true if there is, false otherwise
     */
    private boolean stillToAdd()
    {
        for (boolean b : added) if (!b) return true;
        return false;
    }

    @Override
    public double classify(Instance i) {
        int count = 0;
        
        for (Classifier_DT dt : decisionTrees)
        {
            count += dt.classify(i);
        }
        
        return count/decisionTrees.length>0.5?0:1;
    }
}
