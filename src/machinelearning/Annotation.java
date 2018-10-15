/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machinelearning;

/**
 * Holds the instance id and its associated annotation confidence
 * @author Owner
 */
public class Annotation {
    public int id;
    public double confidence;
    
    /**
     * Create an annotation association between the instance with ID id_ and with confidence confidence_
     * @param id_ The ID of the instance with the associated annotation confidence
     * @param confidence_  The confidence of the annotation which the instance is labelled with
     */
    public Annotation(int id_, double confidence_)
    {
        id = id_;
        confidence = confidence_;
    }
}
