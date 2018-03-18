package de.uni_saarland.coli.layers;

import de.uni_saarland.coli.util.ArrayMath;

public class Softmax {

    /**
     *
     * @param inputs
     * @return
     */
    public double[] apply(double[] inputs) {
        double max = ArrayMath.max(inputs);
        double sum = ArrayMath.expSum(inputs,max);

        double[] result = new double[inputs.length];
        for(int i=0;i<inputs.length;++i) {
            result[i] = Math.exp(inputs[i]-max)/sum;
        }

        return result;
    }

}
