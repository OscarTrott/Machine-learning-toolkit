/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machinelearning;

/**
 * Encodes a single instance as parsed from the input files
 * @author Owner
 */
public class Instance {
    int classifiedAs;
    final Feature[] features;
    final int instanceNum;
    
    /**
     * Create an instances with a given class
     * @param features_ features of this instance
     * @param class_ class, 1 or 0, which this instance belongs to
     * @param instanceNum_ the instances instance number, unique for each instance
     */
    public Instance(Feature[] features_, int class_, int instanceNum_)
    {
        instanceNum = instanceNum_;
        features = features_;
        classifiedAs = class_;
    }
    
    /**
     * Creates an instance with no set class
     * @param features_ features of this instance
     * @param instanceNum_ the instances instance number, unique for each instance
     */
    public Instance(Feature[] features_, int instanceNum_)
    {
        instanceNum = instanceNum_;
        features = features_;
        classifiedAs = 0;
    }
    
    public int get_instanceNum()
    {
        return instanceNum;
    }
    
    public void  set_Class(int class_)
    {
        classifiedAs = class_;
    }
    
    public int get_Class()
    {
        return classifiedAs;
    }
}
