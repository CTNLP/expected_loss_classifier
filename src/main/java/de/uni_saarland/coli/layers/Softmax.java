package de.uni_saarland.coli.layers;

import de.uni_saarland.coli.learning_rates.LearningRate;
import de.uni_saarland.coli.util.ArrayMath;
import java.util.function.DoubleSupplier;

/**
 * 
 * @author christoph_teichmann
 */
public class Softmax extends Layer {

    /**
     * 
     * @param id
     * @param previous 
     */
    public Softmax(int id, Layer previous) {
        super(id, previous);
    }
    
    /**
     * 
     * @param id 
     */
    public Softmax(int id) {
        super(id);
    }

    @Override
    protected double[] applyLayer(double[] input) {
        double[] result = new double[input.length];
        
        double max = ArrayMath.max(input);
        double sum = ArrayMath.expSum(input, max);
        
        for(int i=0;i<input.length;++i) {
            result[i] = Math.exp(input[i]-max) / sum;
        }
        
        return result;
    }

    @Override
    protected double[] computeBackpropagation(double[] losses, double[] localInputs) {       
        if(losses.length != localInputs.length) {
            throw new IllegalArgumentException("Input output array length mismatch.");
        }
        double[] probs = this.applyLayer(localInputs);
        
        double[] backgradient = new double[localInputs.length];
        
        for(int i=0;i<losses.length;++i) {
            for(int j=0;j<probs.length;++j) {
                double grad = losses[i]*(probs[i]*((i==j ? 1 : 0)- probs[j]));
                
                backgradient[j] += grad;
            }
        }
        
        return backgradient;
    }

    @Override
    protected void accumulateLocal(double[] losses, double[] localInputs) {}

    @Override
    protected void clearGradientLocal() {}

    @Override
    protected void stepLocal(int id, LearningRate rate) {}

    @Override
    protected void initializeLocal(DoubleSupplier apply) {}

}
