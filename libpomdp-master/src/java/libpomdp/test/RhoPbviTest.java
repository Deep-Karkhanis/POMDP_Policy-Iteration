package libpomdp.test;

import libpomdp.common.rho.RhoPomdp;
import libpomdp.common.std.PomdpStd;
import libpomdp.parser.FileParser;
import libpomdp.solve.offline.Criteria;
import libpomdp.solve.offline.MaxIterationsCriteria;
import libpomdp.solve.offline.vi.ValueConvergenceCriteria;
import libpomdp.solve.offline.vi.ValueIterationStats;
import libpomdp.solve.offline.pointbased.PbParams;
import libpomdp.solve.offline.pointbased.PointBasedStd;

public class RhoPbviTest {

	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		//tiger/tiger.95.POMDP
		PomdpStd pomdp=(PomdpStd)FileParser.loadPomdp("data/problems/tiger/tiger.95.POMDP", FileParser.PARSE_CASSANDRA_POMDP);
		double epsi=1e-6*(1-pomdp.getGamma())/(2*pomdp.getGamma());
		RhoPomdp rpomdp=new RhoPomdp(pomdp,new TigerRho());
		PbParams params=new PbParams(PbParams.BACKUP_SYNC_FULL,PbParams.EXPAND_GREEDY_ERROR_REDUCTION,100);
		PointBasedStd algo= new PointBasedStd(rpomdp,params);
		algo.addStopCriteria(new MaxIterationsCriteria(100));
		algo.addStopCriteria(new ValueConvergenceCriteria(epsi,Criteria.CC_MAXDIST));
		algo.run();
		System.out.println(algo.getValueFunction());
		ValueIterationStats stat=(ValueIterationStats) algo.getStats();
		System.out.println(stat);
		//System.out.println(val);
	}

}
