package de.uni_saarland.coli.learning_rates;

/**
 * 
 * @author christoph_teichmann
 */
public interface LearningRate {
    
    /**
     * 
     * @param layer
     * @param pos
     * @param gradient
     * @return 
     */
    public double getStep(int layer, int pos, double gradient);
    
}
