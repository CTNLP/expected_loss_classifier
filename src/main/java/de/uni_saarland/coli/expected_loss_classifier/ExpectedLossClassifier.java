package de.uni_saarland.coli.expected_loss_classifier;

import de.uni_saarland.coli.util.ArrayMath;

public class ExpectedLossClassifier {

    /**
     *
     */
    private final double[][][] weights;

    /**
     *
     * @param inputDimensions
     * @param resultDimensions
     */
    public ExpectedLossClassifier(int inputDimensions, int[] resultDimensions) {
        this.weights = new double[resultDimensions.length][][];

        for(int i=0;i<weights.length;++i) {
            int inDim = i==0 ? inputDimensions : resultDimensions[i-1];
            int outDim = resultDimensions[i];

            double[][] layerWeights = new double[outDim][inDim];
        }
    }

    // TODO: tools for optimization and initialization

    /**
     *
     * @param input
     * @return
     */
    public double[][] compute(double[] input) {
        double[][] result = new double[weights.length][];

        for(int i=0;i<weights.length;++i) {
            double[] ins = i==0 ? input : result[i-1];
            double[][] ws = weights[i];

            double[] done = new double[ws.length];
            result[i] = done;

            double sum = 0.0;
            for(int j=0;j<ws.length;++j) {
                double inner = inner_product(ws[j],ins);

                sum += done[j] = i == result.length-1 ? Math.exp(inner) : Math.max(0,inner);
            }

            if(i == result.length-1) {
                for(int j=0;j<done.length;++j) {
                    done[j] /= sum;
                }
            }
        }


        return result;
    }

    /**
     *
     * @param input
     * @return
     */
    public double[] computeProbabilities(double[] input) {
        double[][] all = this.compute(input);

        return all[all.length-1];
    }


    /**
     *
     * @param input
     * @return
     */
    public int  chooseOne(double[] input) {
        double[] choices = this.computeProbabilities(input);

        int best = ArrayMath.argmax(choices);

        return best;
    }

    /**
     *
     * @param w
     * @param ins
     * @return
     */
    private double inner_product(double[] w, double[] ins) {
        double amount = 0.0;

        for(int i=0;i<w.length;++i) {
            amount += w[i]*ins[i];
        }

        return amount;
    }

}
