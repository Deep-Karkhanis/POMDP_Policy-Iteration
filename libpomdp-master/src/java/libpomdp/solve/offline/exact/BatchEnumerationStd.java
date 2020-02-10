package libpomdp.solve.offline.exact;

import libpomdp.common.AlphaVector;
import libpomdp.common.std.BeliefMdpStd;
import libpomdp.common.std.PomdpStd;
import libpomdp.common.std.ValueFunctionStd;
import libpomdp.solve.offline.IterationStats;
import libpomdp.solve.offline.vi.ValueIterationStats;
import libpomdp.solve.offline.vi.ValueIterationStd;


public class BatchEnumerationStd extends ValueIterationStd {
	
	BeliefMdpStd bmdp;
	double delta;
	
	public BatchEnumerationStd(PomdpStd pomdp,double delta){
		startTimer();
		initValueIteration(pomdp);
		this.delta=delta;
		bmdp=new BeliefMdpStd(pomdp);
		current = new ValueFunctionStd(pomdp.nrStates());
		current.push(new AlphaVector(bmdp.nrStates()));
		registerInitTime();
	}
	
	@Override
	public IterationStats iterate() {
		startTimer();
		old=current;
		current = new ValueFunctionStd(bmdp.nrStates());
		for(int a=0; a<bmdp.nrActions(); a++){
			for (int idx=0;idx<old.size();idx++){
				AlphaVector prev=old.getAlphaVector(idx);
				AlphaVector alpha=new AlphaVector(bmdp.nrStates());
				for (int o=0;o<bmdp.nrObservations();o++){
					alpha.add(bmdp.projection(prev,a,o));
				}
				current.push(alpha);
			}
			current.crossSum(bmdp.getRewardValueFunction(a));
		}
		((ValueIterationStats) iterationStats).registerLp(current.prune(delta));
		registerValueIterationStats();
    	return iterationStats;
	}

}
