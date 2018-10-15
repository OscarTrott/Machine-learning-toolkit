/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machinelearning;

/**
 * Represents a single feature of an instance
 * @author Owner
 */
public class Feature {
    double value; //Value of the feature
    FeatureType type; //Features type, gist or CNN
    
    public Feature(FeatureType type_, double value_)
    {
        value = value_;
        type = type_;
    }
    
    public FeatureType getType()
    {
        return type;
    }
    
    public double getVal()
    {
        return value;
    }
}
