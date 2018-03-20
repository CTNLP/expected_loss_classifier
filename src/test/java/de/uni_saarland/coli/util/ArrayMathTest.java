/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package de.uni_saarland.coli.util;

import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author christoph_teichmann
 */
public class ArrayMathTest {
    
    /**
     * 
     */
    private final static double[] TEST_VEC = new double[] {Math.log(1),Math.log(0.5),Math.log(5)};
    
    @Before
    public void setUp() {
    }

    /**
     * Test of argmax method, of class ArrayMath.
     */
    @org.junit.Test
    public void testArgmax() {
        assertEquals(ArrayMath.argmax(TEST_VEC),2);
    }

    /**
     * Test of max method, of class ArrayMath.
     */
    @org.junit.Test
    public void testMax() {
        assertEquals(ArrayMath.max(TEST_VEC),Math.log(5),0.00000002);
    }

    /**
     * Test of expSum method, of class ArrayMath.
     */
    @org.junit.Test
    public void testExpSum_doubleArr_double() {
        double d = ArrayMath.expSum(TEST_VEC);
        
        assertEquals(6.5,d,0.0000000001);
        
        d = ArrayMath.expSum(TEST_VEC, Math.log(0.5));
        
        assertEquals(13.0,d,0.0000000001);
    }

    /**
     * Test of innerProduct method, of class ArrayMath.
     */
    @org.junit.Test
    public void testInnerProduct() {
        double[] v = new double[] {3.0,-2.0};
        double[] w = new double[] {1.0,-3.0};
        
        double q = ArrayMath.innerProduct(v, w);
        
        assertEquals(q,9.0,0.000000001);
    }
    
}
