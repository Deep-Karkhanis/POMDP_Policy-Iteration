#include <AIToolbox/POMDP/Algorithms/PolicyIteration.hpp>
#include <queue>
#include <algorithm>
#include <map>

namespace AIToolbox::POMDP {
    PolicyIteration::PolicyIteration(const unsigned h, const double t, AIToolbox::POMDP::Belief b) :
            horizon_(h)
    {
        setTolerance(t);
        bel_ = b;    
    }

    void PolicyIteration::setHorizon(const unsigned h) {
        horizon_ = h;
    }
    void PolicyIteration::setTolerance(const double t) {
        if ( t < 0.0 ) throw std::invalid_argument("Tolerance must be >= 0");
        tolerance_ = t;
    }

    unsigned PolicyIteration::getHorizon() const {
        return horizon_;
    }

    double PolicyIteration::getTolerance() const {
        return tolerance_;
    }

    VList PolicyIteration::crossSum(const VList & l1, const VList & l2, const size_t a, const bool order) {
        VList c;

        if ( !(l1.size() && l2.size()) ) return c;

        // We can get the sizes of the observation vectors
        // outside since all VEntries for our input VLists
        // are guaranteed to be sized equally.
        const auto O1size  = l1[0].observations.size();
        const auto O2size  = l2[0].observations.size();
        for ( const auto & v1 : l1 ) {
            const auto O1begin = std::begin(v1.observations);
            const auto O1end   = std::end  (v1.observations);
            for ( const auto & v2 : l2 ) {
                const auto O2begin = std::begin(v2.observations);
                const auto O2end   = std::end  (v2.observations);
                // Cross sum
                auto v = v1.values + v2.values;

                // This step now depends on which order the two lists
                // are. This function is only used in this class, so we
                // know that the two lists are "adjacent"; however one
                // is after the other. `order` tells us which one comes
                // first, and we join the observation vectors accordingly.
                VObs obs; obs.reserve(O1size + O2size);
                if ( order ) {
                    obs.insert(std::end(obs), O1begin, O1end);
                    obs.insert(std::end(obs), O2begin, O2end);
                } else {
                    obs.insert(std::end(obs), O2begin, O2end);
                    obs.insert(std::end(obs), O1begin, O1end);
                }
                c.emplace_back(std::move(v), a, std::move(obs));
            }
        }

        return c;
    }


    void PolicyIteration::makeInitialFSC()
    {
        // MDP::Values vals;
        // vals.resize(S,1);
        // for(int i=0;i<(int)S;i++)
        //     vals(i,0) = 0;
        // VObs succO;
        // // succO.resize(O,1);
        // std::vector<size_t> obsVal(O,0);
        // // for(int i=0;i<(int)O;i++)
        // //     vals(i,0) = 0;
        // succO=obsVal;
        // MchnState ms(vals,0,succO,0);
        fsc_F.nodes = std::vector<MchnState>(0);
        bool custom_fsc;
        std::cin>>custom_fsc; //$$$$$$$$$$$$$$$$$
        std::cin>>howard_; //$$$$$$$$$$$$$$$$$
        std::cin>>B_; //$$$$$$$$$$$$$$$$$
        std::cin>>nodeLimit_; //$$$$$$$$$$$$$$$$$

        
        if(custom_fsc == 1)
        {
            int inpFscSz;
            std::cin>>inpFscSz;      //$$$$$$$$$$$$$$$$$      
            for(int mchSt=0;mchSt<(int)inpFscSz;mchSt++)
            {
                MDP::Values vals;
                vals.resize(S,1);
                VObs obsVal(O,0);
                int node_act;
                std::cin>>node_act;  //$$$$$$$$$$$$$$$$$
                for(int i=0;i<(int)S;i++)
                    std::cin>>vals(i,0);
                for(int i=0;i<(int)O;i++)
                    std::cin>>obsVal[i];
                MchnState ms(vals,node_act,obsVal,mchSt);
                fsc_F.nodes.push_back(ms);
            }
            fsc_F.maxId = inpFscSz;

            double maxExp=fsc_F.nodes[0].values.dot(bel_);
            int maxNode = 0;
            for(int i=0;i<inpFscSz;i++)
            {    
                double val = (fsc_F.nodes[i].values.dot(bel_)); 
                if(maxExp < val)
                {
                    maxExp = val;
                    maxNode = i;
                }
            }
            fsc_F.nodes[maxNode].mark = 1;
            fsc_F.maxE = maxExp;
        }
        else
        {
            MDP::Values vals;
            vals.resize(S,1);
            VObs obsVal(O,0);
            for(int i=0;i<(int)S;i++)
                vals(i,0)=0;
            MchnState ms(vals,0,obsVal,0);
            fsc_F.nodes.push_back(ms);
            fsc_F.maxId=0;
            fsc_F.maxE = 0;
        }    
        fsc_ = fsc_F;
        return;
    }

    void PolicyIteration::updateFSC(const VList &w)
    {
        if(w.size() == 0)
            return;
        for(int i=0;i<(int)(fsc_.nodes.size());i++)
            fsc_.nodes[i].mark=0;
        fsc_.changed=0;
        // std::cout<<" chkpB1"<<std::endl<<std::flush;
        std::vector<size_t> nonDups(0);        
        if(howard_ == 1)
        {
            for(size_t alpDInd=0;alpDInd<(w.size()); alpDInd++)
            {
                int dup = 0;
                auto alpD = w[alpDInd];
                for(size_t alpInd=0;alpInd<(fsc_.nodes.size()); alpInd++)
                {
                    auto alp=fsc_.nodes[alpInd];
                    if(alp.action == alpD.action && alp.obsSucc == alpD.observations 
                            // && alp.mark != 1
                            )
                    {
                        dup=1;
                        for(int ss=0;ss<(alp.values.rows());ss++)
                        {
                            if(alpD.values(ss,0) != alp.values(ss,0))
                            {
                                dup=0;
                                break;
                            }
                        }
                        if(dup == 1)
                        {
                            fsc_.nodes[alpInd].mark=1;
                            break;
                        }
                    }
                }
                if(dup==0)
                    nonDups.push_back(alpDInd);
            }
        }
        else
        {
            for(size_t alpDInd=0;alpDInd<(w.size()); alpDInd++)
                nonDups.push_back(alpDInd);
            
        }
        // std::cout<<" chkpB2"<<std::endl<<std::flush;
        
        std::vector<size_t> toErase(0);
        std::map<int,int> eraseReplaceID;
        for(size_t nonDInd=0;nonDInd<(nonDups.size()); nonDInd++)
        {
            size_t alpDInd = nonDups[nonDInd];
            auto alpD = w[alpDInd];
            size_t nextId = fsc_.maxId + 1; 
            if(howard_ == 1)
            {
                for(size_t alpInd=0;alpInd<(fsc_.nodes.size()); alpInd++)
                {
                    auto alp=fsc_.nodes[alpInd];
                    if(fsc_.nodes[alpInd].mark==1)
                        continue;
                    bool doms = 1;
                    for(int ss=0;ss<(alp.values.rows());ss++)
                    {
                        if(alpD.values(ss,0) < alp.values(ss,0))
                        {
                            doms=0;
                            break;
                        }
                    }
                    if(doms==1)
                    {
                        toErase.push_back(alp.iden);
                        eraseReplaceID[alp.iden] = nextId;
                    }
                }
            }
            MchnState newMs(alpD.values,alpD.action,alpD.observations,nextId);
            newMs.mark=1;
            fsc_.nodes.push_back(newMs);
            fsc_.maxId++;
        }
        // std::cout<<" chkpB3"<<std::endl<<std::flush;
        
        if(howard_ == 1)
        {   
            // std::cout<<" chkpB3a"<<std::endl<<std::flush;
            
            for(size_t delCnt = 0;delCnt < toErase.size();delCnt++)
            {
                std::vector<MchnState>::iterator delAt;
                for(std::vector<MchnState>::iterator alpI=fsc_.nodes.begin();
                        alpI != fsc_.nodes.end(); alpI++)
                {
                    MchnState alp = *alpI;
                    if(alp.iden != toErase[delCnt])
                        continue;
                    else
                        delAt = alpI;
                }
                fsc_.nodes.erase(delAt);
            }
            // std::cout<<" chkpB3b"<<std::endl<<std::flush;
            
            for(size_t i=0;i<fsc_.nodes.size();i++)
            {
                for(int j=0;j<(int)O;j++)
                {
                    if(find(toErase.begin(),toErase.end(),fsc_.nodes[i].obsSucc[j])
                         != toErase.end())
                        fsc_.nodes[i].obsSucc[j] = eraseReplaceID[fsc_.nodes[i].obsSucc[j]];
                }
            }        
            // std::cout<<" chkpB3c"<<std::endl<<std::flush;
            
            int noofDel=0;
            for(int newFN = 0; newFN < (int)(fsc_.nodes.size()); newFN++)
            {
                if(fsc_.nodes[newFN].mark != 1)
                    continue;
                
                fsc_.nodes[newFN].mark = 2;
                noofDel++;
                std::queue<int> stateQueue;
                stateQueue.push(newFN);

                int loop=0;
                while(!stateQueue.empty())
                {   
                    loop++;
                    int ind = stateQueue.front();
                    stateQueue.pop();
                    for(int succN=0;succN<(int)O;succN++)
                    {
                        int succID = fsc_.nodes[ind].obsSucc[succN];
                        int succInd;
                        for(int j=0;j<(int)(fsc_.nodes.size());j++)
                        {
                            if((int)fsc_.nodes[j].iden==succID)
                            {
                                succInd = j;
                                break;
                            }
                        }
                        if(fsc_.nodes[succInd].mark != 2)
                        {
                            fsc_.nodes[succInd].mark = 2;
                            noofDel++;
                            stateQueue.push(succInd);
                        }
                    }
                }
            }
            noofDel = fsc_.nodes.size() - noofDel;
            // std::cout<<" chkpB3d"<<std::endl<<std::flush;
            
            for(int i = 0; i < noofDel; i++)
            {
                for(std::vector<MchnState>::iterator j=fsc_.nodes.begin();
                            j != fsc_.nodes.end();j++)
                    {
                        if((j->mark) != 2)
                        {
                            fsc_.nodes.erase(j);
                            break;
                        }
                    }
            }
            int noofIds = (int)(fsc_.nodes.size());
            std::vector<int> oldIds(0);
            for(int i=0;i<noofIds;i++)
                oldIds.push_back(fsc_.nodes[i].iden);
            
            std::map<int,int> oldToNewID;
            for(int i=0;i<noofIds;i++)
                oldToNewID[oldIds[i]]=i;

            // std::cout<<" chkpB3e"<<std::endl<<std::flush;
            
            for(int i=0;i<noofIds;i++)
            {
                fsc_.nodes[i].iden = oldToNewID[fsc_.nodes[i].iden];
                fsc_.nodes[i].mark=0;
                for(int j=0;j<(int)O;j++)
                {
                    fsc_.nodes[i].obsSucc[j] = 
                            oldToNewID[fsc_.nodes[i].obsSucc[j]];
                }
            }        
            fsc_.maxId = fsc_.nodes.size();
        }
        // std::cout<<" chkpB4"<<std::endl<<std::flush;
        
        for(int i=0;i<(int)(fsc_.nodes.size());i++)
            fsc_.nodes[i].mark=0;
        if(nonDups.size()>0)
            fsc_.changed=1;
        
    }
}

            