package libpomdp.solve.offline.vi;

import libpomdp.common.AlphaVector;
import libpomdp.common.CustomVector;
import libpomdp.common.ValueFunction;
import libpomdp.common.std.ValueFunctionStd;
import libpomdp.solve.offline.Criteria;
import libpomdp.solve.offline.Iteration;

public class ValueConvergenceCriteria extends Criteria {

	double epsilon;
	int convCriteria;
	
	static final int MIN_ITERATIONS = 5;
	
	public boolean check(Iteration i) {
		ValueIteration vi=(ValueIteration)i;
		ValueFunction newv=vi.getValueFunction();
		ValueFunction oldv=vi.getOldValueFunction();
		if (oldv==null  || newv.size()!=oldv.size()){
			System.out.println("Eval(" + i.getStats().iterations + ") = Inf");
			return false;
		}
		((ValueFunctionStd)newv).sort();
		((ValueFunctionStd)oldv).sort();
		double conv=0;
		for(int j=0; j<newv.size(); j++){
			AlphaVector newAlpha=newv.getAlphaVector(j);
			AlphaVector oldAlpha=oldv.getAlphaVector(j);
			if (newAlpha.getAction()!=oldAlpha.getAction()){
				System.out.println("Eval(" + i.getStats().iterations + ") = Inf");
				return false;
			}
			CustomVector perf=newAlpha.getVectorCopy();
			perf.add(-1.0,oldAlpha.getVectorRef());
			double a_value=0;
			switch(convCriteria){
			case CC_MAXEUCLID:
				a_value = perf.norm(2.0);
				break;
			case CC_MAXDIST:
				a_value = perf.norm(1.0);
				break;
			}
			if (a_value > conv)
				conv=a_value;
		}
		System.out.println("Eval(" + i.getStats().iterations + ") = " + conv);
		if (conv <= epsilon && i.getStats().iterations > MIN_ITERATIONS)
			return(true);
		return false;
 	}

	@Override
	public boolean valid(Iteration vi) {
		if (vi instanceof ValueIteration){
			return true;
		}
		return false;
	}

	public ValueConvergenceCriteria(double epsilon,int convCriteria) {
		this.epsilon=epsilon;
		this.convCriteria=convCriteria;
	}
	


}
