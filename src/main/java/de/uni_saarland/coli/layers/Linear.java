/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package de.uni_saarland.coli.layers;

import de.uni_saarland.coli.learning_rates.LearningRate;
import de.uni_saarland.coli.util.ArrayMath;
import java.util.Arrays;
import java.util.function.DoubleSupplier;

/**
 *
 * @author christoph_teichmann
 */
public class Linear extends Layer {
    
    /**
     * 
     */
    private final double[][] weights;
    
    /**
     * 
     */
    private final double[] biases;
    
    /**
     * 
     */
    private final double[][] weightGradient;
    
    /**
     * 
     */
    private final double[] biasGradient;

    /**
     * 
     * @param id
     * @param indim
     * @param outdim
     * @param bias
     * @param previous 
     */
    public Linear(int id, int indim, int outdim, boolean bias, Layer previous) {
        super(id, previous);
        
        if(outdim < 1) {
            throw new IllegalArgumentException("Need at least one output dimension.");
        }
        
        if(bias) {
            this.biases = new double[outdim];
            biasGradient = new double[outdim];
        } else {
            this.biases = null;
            biasGradient = null;
        }
        
        this.weights = new double[outdim][indim];
        this.weightGradient = new double[outdim][indim];
    }
    
    /**
     * 
     * @param id
     * @param indim
     * @param bias
     * @param outdim 
     */
    public Linear(int id, int indim, boolean bias, int outdim) {
        this(id, indim, outdim, bias, null);
    }

    @Override
    protected double[] applyLayer(double[] input) {
        if(input.length != this.weights[0].length) {
            throw new IllegalArgumentException("Input size and input dimension do not match.");
        }
        
        double[] outs = new double[this.weights.length];
        
        for(int i=0;i<this.weights.length;++i) {
            double[] w = this.weights[i];
            
            outs[i] = ArrayMath.innerProduct(input,w);
            
            if(this.biases != null) {
                outs[i] += this.biases[i];
            }
            
        }
        
        return outs;
    }

    @Override
    protected double[] computeBackpropagation(double[] losses, double[] localInputs) {
        double[] back = new double[localInputs.length];
        
        for(int i=0;i<losses.length;++i) {
            double[] w = this.weights[i];
            
            for(int in=0;in<w.length;++in) {
                double g = losses[i]*w[in];
                
                back[in] += g;
            }
        }
        
        return back;
    }

    @Override
    protected void accumulateLocal(double[] losses, double[] localInputs) {
        if(localInputs.length != this.weights[0].length) {
            throw new IllegalArgumentException("input dimensions and input passed do not match");
        }
        
        if(losses.length != this.weights.length) {
            throw new IllegalArgumentException("output dimensions and losses passed do not match");
        }
        
        for(int i=0;i<this.weightGradient.length;++i) {
            double[] grad = this.weightGradient[i];
            
            for(int j=0;j<grad.length;++j) {
                double g = losses[i]*localInputs[j];
                
                grad[j] += g;
            }
        }
        
        if(this.biases != null) {
            for(int i=0;i<this.biasGradient.length;++i) {
                this.biasGradient[i] += losses[i];
            }
        }
    }

    @Override
    protected void clearGradientLocal() {
        for(double[] w : this.weightGradient) {
            Arrays.fill(w, 0.0);
        }
        
        if(this.biasGradient != null) {
            Arrays.fill(this.biasGradient, 0.0);
        }
    }

    @Override
    protected void stepLocal(int id, LearningRate rate) {
        int position = 0;
        
        for(int j=0;j<this.weightGradient.length;++j) {
            double[] w = this.weightGradient[j];
            
            for(int i=0;i<w.length;++i) {
                int pos = position++;
                
                this.weights[j][i] -= rate.getStep(id, pos, w[i]);
            }
        }
        
        if(this.biases != null) {           
            for(int i=0;i<this.biasGradient.length;++i) {
                int pos = position++;
                
                this.biases[i] -= rate.getStep(id, pos, this.biasGradient[i]);
            }
        }
    }

    @Override
    protected void initializeLocal(DoubleSupplier apply) {
        for(double[] w : this.weights) {
            for(int i=0;i<w.length;++i) {
                w[i] = apply.getAsDouble();
            }
        }
        
        if(this.biases != null) {
            for(int i=0;i<this.biases.length;++i) {
                this.biases[i] = apply.getAsDouble();
            }
        }
    }
    
}
