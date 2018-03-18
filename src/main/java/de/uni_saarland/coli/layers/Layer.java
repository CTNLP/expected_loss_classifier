package de.uni_saarland.coli.layers;

import de.uni_saarland.coli.learning_rates.LearningRate;

public abstract class Layer {

    /**
     *
     */
    private final int id;

    /**
     *
     */
    private final Layer previous;

    /**
     *
     */
    private double[] forward;

    /**
     *
     * @param id
     * @param previous
     */
    public Layer(int id, Layer previous) {
        this.id = id;
        this.previous = previous;
    }

    /**
     *
     * @param id
     */
    public Layer(int id) {
        this.id = id;
        this.previous = null;
    }

    /**
     *
     * @param input
     * @return
     */
    public double[] forward(double[] input) {
        this.forward = this.previous == null ? input : this.previous.forward(input);

        return this.applyLayer(this.forward);
    }


    protected abstract double[] applyLayer(double[] input);

    /**
     *
     * @param losses
     */
    public void accumulateGradient(double[] losses) {
        if(forward == null) {
           throw new IllegalStateException("No forward pass data stored");
        }

        accumulateLocal(losses,this.forward);

        if(previous != null) {
            double[] inputLosses = computeBackpropagation(losses,forward);

            this.previous.accumulateGradient(inputLosses);
        }
    }

    /**
     *
     * @param losses
     * @param localInputs
     * @return
     */
    protected abstract double[] computeBackpropagation(double[] losses, double[] localInputs);

    /**
     *
     * @param losses
     * @param localInputs
     */
    protected abstract void accumulateLocal(double[] losses, double[] localInputs);

    /**
     *
     * @param rate
     */
    public abstract void step(LearningRate rate);

}
