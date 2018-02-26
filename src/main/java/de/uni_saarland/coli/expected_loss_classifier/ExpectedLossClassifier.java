package de.uni_saarland.coli.expected_loss_classifier;

import de.uni_saarland.coli.util.ArrayMath;

import java.util.Arrays;

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

        for (int i = 0; i < weights.length; ++i) {
            int inDim = i == 0 ? inputDimensions : resultDimensions[i - 1];
            int outDim = resultDimensions[i];

            double[][] layerWeights = new double[outDim][inDim];
        }
    }

    /**
     *
     * @param layer
     * @param output
     * @param weight
     * @param value
     */
    public void setParameter(int layer, int output, int weight, double value) {
        this.weights[layer][output][weight] = value;
    }

    /**
     *
     * @param layer
     * @param output
     * @param weight
     * @param value
     */
    public void addValue(int layer, int output, int weight, double value) {
        this.weights[layer][output][weight] += value;
    }

    /**
     *
     * @return
     */
    public int getNumberOfLayers() {
        return this.weights.length;
    }

    /**
     *
     * @param layer
     * @return
     */
    public int getNumberOfOutputs(int layer) {
        return this.weights[layer].length;
    }

    /**
     *
     * @param layer
     * @param output
     * @return
     */
    public int getNumberOfWeights(int layer, int output) {
        return this.weights[layer][output].length;
    }

    /**
     *
     * @param inputs
     * @param losses
     * @return
     */
    public double[][][] gradient(double[] inputs, double[] losses) {
        double[][] outputs = this.compute(inputs);

        return this.gradient(inputs, losses, outputs);
    }

    /**
     *
     * @param inputs
     * @param losses
     * @param outputs
     * @return
     */
    private double[][][] gradient(double[] inputs, double[] losses, double[][] outputs) {
        double[][][] parameterGrads = new double[outputs.length][][];
        double[][] nonLinGrads = new double[outputs.length][];

        for (int layer = outputs.length - 1; layer >= 0; --layer) {
            double[] multipliers = (layer + 1 == outputs.length) ? losses : nonLinGrads[layer + 1];
            double[] ins = (layer == 0) ? inputs : outputs[layer - 1];

            double[][] paraGrads = new double[outputs[layer].length][];
            parameterGrads[layer] = paraGrads;

            double[] nonLGrads = new double[ins.length];
            nonLinGrads[layer] = nonLGrads;

            for (int cell = 0; cell < paraGrads.length; ++cell) {
                double[] local = new double[weights[layer][cell].length];
                if (layer + 1 != outputs.length && multipliers[cell] <= 0) {
                    Arrays.fill(local, 0.0);
                    continue;
                }

                for (int i = 0; i < local.length; ++i) {
                    if (layer + 1 != outputs.length) {
                        double val = multipliers[cell];

                        local[i] = val * inputs[i];
                        nonLGrads[i] += val * weights[layer][cell][i];
                    } else {
                        double[] probs = outputs[i];
                        for (int iPos = 0; iPos < local.length; ++iPos) {
                            double prob1 = probs[cell];
                            double prob2 = probs[iPos];

                            double val = (multipliers[cell] * prob1) * ((iPos == cell ? 1 : 0) - prob2);

                            local[i] += val * inputs[i];
                            nonLGrads[i] += val * weights[layer][cell][iPos];
                        }
                    }
                }
            }

        }

        return parameterGrads;
    }

    /**
     *
     * @param input
     * @return
     */
    public double[][] compute(double[] input) {
        double[][] result = new double[weights.length][];

        for (int i = 0; i < weights.length; ++i) {
            double[] ins = i == 0 ? input : result[i - 1];
            double[][] ws = weights[i];

            double[] done = new double[ws.length];
            result[i] = done;

            double sum = 0.0;
            for (int j = 0; j < ws.length; ++j) {
                double inner = inner_product(ws[j], ins);

                sum += done[j] = i == result.length - 1 ? Math.exp(inner) : Math.max(0, inner);
            }

            if (i == result.length - 1) {
                for (int j = 0; j < done.length; ++j) {
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

        return all[all.length - 1];
    }

    /**
     *
     * @param input
     * @return
     */
    public int chooseOne(double[] input) {
        return ArrayMath.argmax(this.computeProbabilities(input));
    }

    /**
     *
     * @param w
     * @param ins
     * @return
     */
    private double inner_product(double[] w, double[] ins) {
        double amount = 0.0;

        for (int i = 0; i < w.length; ++i) {
            amount += w[i] * ins[i];
        }

        return amount;
    }

}
