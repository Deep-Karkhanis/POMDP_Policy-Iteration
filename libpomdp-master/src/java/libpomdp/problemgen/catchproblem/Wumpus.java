/** ------------------------------------------------------------------------- *
 * libpomdp
 * ========
 * File: Wumpus.java
 * Description: Interface to implement different wumpi types for the catch
 *              environment.
 * Copyright (c) 2009, 2010 Diego Maniloff 
 --------------------------------------------------------------------------- */

package libpomdp.problems.catchproblem.java;

interface Wumpus {

    // returns a vector with the action distribution for the wumpus
    // this vector could potentially return probs > 0 for nonadjacent cells
    // if we were to implement a jumping wumpus
    public double[] getActDist(int apos, int wpos);

}