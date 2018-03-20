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
import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author christoph_teichmann
 */
public class SoftmaxTest {
    
    /**
     * 
     */
    private Softmax sm;
    
    @Before
    public void setUp() {
        sm = new Softmax(5, null);
    }

    /**
     * Test of applyLayer method, of class Softmax.
     */
    @Test
    public void testApplyLayer() {
        double[] out = sm.applyLayer(new double[] {Math.log(3),Math.log(0.5)});
        
        assertEquals(out.length,2);
        assertEquals(out[0],3/3.5,0.0000001);
        assertEquals(out[1],0.5/3.5,0.0000001);
    }

    /**
     * Test of computeBackpropagation method, of class Softmax.
     */
    @Test
    public void testComputeBackpropagation() {
        double[] losses = new double[] {2,1,0};
        double[] localInputs = new double[] {Math.log(5),Math.log(1),Math.log(3)};
        
        double distortion = 0.00001;
        
        int pos = 0;
        for(double point : new double[] {0.0,-2000.0,5,0.034,20,7,34,-21,5.423542,7.231,
        9.0,-7.5,12.345235,0.232}) {
            int entry = (pos++) % 3;
            
            localInputs[entry] = point;
            
            double grad = this.sm.computeBackpropagation(losses, localInputs)[entry];
            
            localInputs[entry] -= distortion;
            double v1 = ArrayMath.innerProduct(this.sm.applyLayer(localInputs), losses);
            localInputs[entry] += 2*distortion;
            double v2 = ArrayMath.innerProduct(this.sm.applyLayer(localInputs), losses);
            
            
            double estimate = (v2-v1)/(2*distortion);
            
            assertEquals(estimate,grad,0.000001);
        }
    }
    
}
