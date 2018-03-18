package de.uni_saarland.coli.util;

/**
 *
 */
public class ArrayMath {

    /**
     *
     * @param inputs
     * @return
     */
    public static int argmax(double[] inputs) {
        double best = Double.NEGATIVE_INFINITY;
        int bestPos = -1;

        for(int i=0;i<inputs.length;++i) {
            double val = inputs[i];

            if(val > best) {
                best = val;
                bestPos = i;
            }
        }

        return bestPos;
    }

    /**
     *
     * @param inputs
     * @return
     */
    public static double max(double[] inputs) {
        return inputs[argmax(inputs)];
    }

    /**
     *
     * @param inputs
     * @param subtract
     * @return
     */
    public static double expSum(double[] inputs, double subtract) {
        double sum = 0.0;

        for(double d : inputs) {
            sum += Math.exp(d-subtract);
        }

        return sum;
    }

    /**
     *
     * @param inputs
     * @return
     */
    public static double expSum(double[] inputs) {
        return expSum(inputs,0);
    }

}
