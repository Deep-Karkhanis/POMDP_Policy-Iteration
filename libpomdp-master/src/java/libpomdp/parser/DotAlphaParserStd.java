/** ------------------------------------------------------------------------- *
 * libpomdp
 * ========
 * File: DotAlphaParserStd.java
 * Description: Simple class to parse a serialized value function.
 * Copyright (c) 2009, 2010 Diego Maniloff
 * Copyright (c) 2010 Mauricio Araya
 --------------------------------------------------------------------------- */

package libpomdp.parser;

// imports
import libpomdp.common.CustomVector;
import libpomdp.common.ValueFunction;
import libpomdp.common.std.ValueFunctionStd;

import org.antlr.runtime.ANTLRFileStream;
import org.antlr.runtime.CommonTokenStream;
import org.antlr.runtime.RecognitionException;

import java.io.IOException;

public class DotAlphaParserStd {

    static Integer actions[];
    static Double  alphas[][];

    public static void parse (String filename) {
        DotAlphaLexer lex = null;

        try {
            lex = new DotAlphaLexer(new ANTLRFileStream(filename));
        } catch ( IOException ex ) {
            ex.printStackTrace();
            System.exit(1);
        }

       	CommonTokenStream tokens = new CommonTokenStream(lex);
        DotAlphaParser parser = new DotAlphaParser(tokens);

        try {
            parser.dotAlpha();
        } catch (RecognitionException e)  {
            e.printStackTrace();
            System.exit(1);
        }

	actions = parser.getActions();
	alphas  = parser.getAlphas();
    }

    public ValueFunction getValueFunction() {
	int s = actions.length;
	int d = alphas[0].length;
	ValueFunctionStd v=new ValueFunctionStd(d);
	// convert from Integer to int and Double to double
	for (int i=0; i<s; i++) {
		CustomVector vec=new CustomVector(d);
	    for (int j=0; j<d; j++)
	    	vec.set(j,alphas[i][j].doubleValue());
	    v.push(vec,actions[i].intValue());
	}
	// generate flat value function
	return v;
    }

}