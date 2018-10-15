/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machinelearning;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Parses the given file and returns an array of instances as taken from the input file
 * @author Owner
 */
public class Parser {
    File inputCSV;
    ArrayList<Instance> instances; //Uses an arraylist due to not knowing how many entries there are within a given input file
    
    /**
     * Creates a parser with the given file as input
     * @param inputCSV_ file which should be parsed
     */
    public Parser(File inputCSV_)
    {
        inputCSV = inputCSV_;
    }
    
    /**
     * Tells the parser to parse the input file and hold the instances parsed
     */
    public void parse()
    {
        int cnnNum = 0; //Counts the number of cnn features, sheerly for programmatic purposes
        int gistNum = 0; //Counts the number of gist feature
        int csvPos = 1; //holds the position in the csv line
        String line;
        String[] splitLine = {""};
        BufferedReader br = null;
        
        //Create the buffered reader
        try {
            br = new BufferedReader(new FileReader(inputCSV));
        } catch (FileNotFoundException ex) {
            System.out.println(ex.getMessage());
        }
        
        //Read the first line
        try {
            line = br.readLine();
            splitLine = line.split(",");
        } catch (IOException ex) {
            System.out.println(ex.getMessage());        
        }
        
        //Count the CNN's
        while(splitLine[csvPos].equals("CNNs"))
        {
            cnnNum++;
            csvPos++;
        }
        //Count the gist's
        while(csvPos <= splitLine.length - 1&& splitLine[csvPos].equals("GIST"))
        {
            gistNum++;
            csvPos++;
        }
        
        instances = new ArrayList();
        Feature[] features;
        int instance = 1;
        
        //Parse each instance into an entry in the instances arraylist while there is still a line to be parsed
        try {
            while((line = br.readLine()) != null)
            {
                features = new Feature[cnnNum+gistNum];
                splitLine = line.split(",");
                for (int i = 0; i < cnnNum; i++)
                {
                    features[i] = new Feature(FeatureType.CNN, Double.parseDouble(splitLine[i+1]));
                }
                for (int i = 0; i < gistNum; i++)
                {
                    features[i+cnnNum] = new Feature(FeatureType.GIST, Double.parseDouble(splitLine[i+1+cnnNum]));
                }
                if (splitLine.length != 4609) instances.add(new Instance(features, Integer.parseInt(splitLine[cnnNum+gistNum+1]), instance++));
                else instances.add(new Instance(features, instance++));
            }
        } catch (IOException ex) {
            System.out.println(ex.getMessage());
        }
        
    }
    
    /**
     * Returns the instances parsed by this parser
     * @return the instances parsed from the input file
     */
    public ArrayList<Instance> get_Instances()
    {
        return instances;
    }
}
