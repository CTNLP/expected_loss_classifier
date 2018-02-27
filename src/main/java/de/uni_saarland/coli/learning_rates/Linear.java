/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package de.uni_saarland.coli.learning_rates;

/**
 *
 * @author christoph_teichmann
 */
public class Linear {
    
    /**
     * 
     */
    private final double nominator;
    
    /**
     * 
     */
    private final double denominatorBase;
    
    /**
     * 
     */
    private double step = 0;

    /**
     * 
     * @param nominator
     * @param denominatorBase 
     */
    public Linear(double nominator, double denominatorBase) {
        this.nominator = nominator;
        this.denominatorBase = denominatorBase;
    }
    
    /**
     * 
     * @param index
     * @return 
     */
    public double getRate(int... index) {
        return (nominator / ((step)+nominator) );
    }
    
    /**
     * 
     */
    public void nextStep() {
        ++this.step;
    }
    
}
