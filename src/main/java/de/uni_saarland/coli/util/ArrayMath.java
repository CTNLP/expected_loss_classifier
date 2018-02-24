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


}
