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
        // std::cout<<"is custom?"<<std::endl<<std::flush;;
        std::cin>>custom_fsc; //$$$$$$$$$$$$$$$$$
        std::cin>>howard_; //$$$$$$$$$$$$$$$$$
        std::cin>>B_; //$$$$$$$$$$$$$$$$$
        std::cin>>nodeLimit_; //$$$$$$$$$$$$$$$$$

        // std::cout<<"custom_fsc is "<<custom_fsc<<std::endl<<std::flush;;

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
            // if(howard_ == 0)
            // {
            //     fsc_.nodes[0].mark = 1;
            // }
        }    
        // std::vector<* MchnState> adjL(1,&(fsc_.nodes[0]));
        // fsc_.edges = std::vector<std::vector<* MchnState> >(1,adjL);
        // fsc_.nodes[0].adjList = &(fsc_.edges[0]);
        // return std::vector<VList>(1,std::vector<VEntry>(1,VEntry(vals,0,std::vector<size_t>(O,0))));
        fsc_ = fsc_F;
        return;
    }

    void PolicyIteration::updateFSC(const VList &w)
    {
        if(w.size() == 0)
            return;
        if(howard_ == 1)
            for(int i=0;i<(int)(fsc_.nodes.size());i++)
                fsc_.nodes[i].mark=0;
        //possiblility of improvement
        FSC cpyFsc = fsc_;
        std::vector<size_t> nodeIds(0);
        
        for(size_t alpDInd=0;alpDInd<(w.size()); alpDInd++)
        {
            const VEntry &alpD = w[alpDInd];
            // std::cout<<alpDInd<<w[alpDInd].observations[1]<<" "<<int(alpD.observations[1])<<std::endl;
            int dontAdd = 0;
            for(size_t alpInd=0;alpInd<(cpyFsc.nodes.size()); alpInd++)
            {
                // cpyFsc.nodes[alpInd].mark = 1;
                auto alp=cpyFsc.nodes[alpInd];
                
                if(alp.action == alpD.action && alp.obsSucc == alpD.observations 
                           // && alp.mark != 1
                           )
                {
                    dontAdd=1;
                    for(int ss=0;ss<(alp.values.rows());ss++)
                    {
                        if(alpD.values(ss,0) != alp.values(ss,0))
                        {
                            dontAdd=0;
                            break;
                        }
                    }
                    if(dontAdd == 1)
                    {
                        fsc_.nodes[alpInd].mark=1;
                        break;
                    }
                }
            }
            // std::cout<<"chkp3"<<std::endl<<std::flush;
        
            if(dontAdd)
            {
                // std::cout<<"did not add1"<<std::endl<<std::flush;
                continue;
            }
            
            for(size_t alpInd=0;alpInd<(cpyFsc.nodes.size()); alpInd++)
            {
                if(fsc_.nodes[alpInd].mark==1)
                    continue;
                auto alp=cpyFsc.nodes[alpInd];
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
                    fsc_.nodes[alpInd].action = alpD.action;
                    fsc_.nodes[alpInd].obsSucc = alpD.observations;
                    fsc_.nodes[alpInd].mark=1;
                    dontAdd = 1;
                    break;
                }
            }
            
            if(dontAdd)
            {
                continue;
            }
            // std::cout<<"chkp3a"<<std::endl<<std::flush;
            
            size_t nextId = fsc_.maxId + 1; 
            MchnState newMs(alpD.values,alpD.action,alpD.observations,nextId);
            newMs.mark=1;
            fsc_.nodes.push_back(newMs);
            fsc_.maxId++;

            // std::cout<<"chkp3c"<<std::endl<<std::flush;
           
            // std::cout<<"------FSC_Inside_Add------"<<std::endl;
            //         for(int mchSt=0;mchSt<(int)fsc_.nodes.size();mchSt++)
            //         {
            //             std::cout<<"--"<<mchSt<<"--"<<std::endl;
            //             std::cout<<fsc_.nodes[mchSt].action<<std::endl;
            //             for(int valI=0;valI<(int)S;valI++)
            //             {
            //                 std::cout<<fsc_.nodes[mchSt].values[valI]<<" ";
            //             }
            //             std::cout<<std::endl;
            //             for(int valI=0;valI<(int)O;valI++)
            //             {
            //                 std::cout<<fsc_.nodes[mchSt].obsSucc[valI]<<" ";
            //             }
            //             std::cout<<std::endl;
            //         }
            //         std::cout<<std::endl;
        }

        if(howard_ == 1)
        {            
            int noofDel=0;
            //std::cout<<fsc_.nodes.size()<<std::endl<<std::flush;

                

            for(int newFN = 0; newFN < (int)(fsc_.nodes.size()); newFN++)
            {
                // std::cout<<"chkp3a"<<std::endl<<std::flush;
            
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
                    // std::cout<<"chkp3aa"<<std::endl<<std::flush; 
                    
                    int ind = stateQueue.front();
                    stateQueue.pop();
                    for(int succN=0;succN<(int)O;succN++)
                    {
                        // std::cout<<"chkp3aaa"<<std::endl<<std::flush; 
                        int succID = fsc_.nodes[ind].obsSucc[succN];
                        int succInd;
                        // for(int j=0;j<(int)(fsc_.nodes.size());j++)
                        // {
                        //     if((int)fsc_.nodes[j].iden==succID)
                        //     {
                        //         succInd = j;
                        //         break;
                        //     }
                        // }
                        succInd = succID;
                        // std::cout<<"chkp3aab"<<std::endl<<std::flush; 
                                    
                        // std::cout<<"FSC: ";
                        // for(int mchSt=0;mchSt<(int)fsc_.nodes.size();mchSt++)
                        // {
                        //     std::cout<<" "<<mchSt<<" "<<fsc_.nodes[mchSt].action<<" ";
                        //     for(int valI=0;valI<(int)S;valI++)
                        //         std::cout<<fsc_.nodes[mchSt].values[valI]<<" ";
                        //     for(int valI=0;valI<(int)O;valI++)
                        //         std::cout<<fsc_.nodes[mchSt].obsSucc[valI]<<" ";
                        //     std::cout<<std::endl;
                        // }
                        // std::cout<<succInd<<std::endl<<std::flush;
                        
                        if(fsc_.nodes[succInd].mark != 2)
                        {
                            // std::cout<<"chkp3aaba"<<std::endl<<std::flush; 
                            fsc_.nodes[succInd].mark = 2;
                            noofDel++;
                            stateQueue.push(succInd);
                        }
                        // std::cout<<"chkp3aac"<<std::endl<<std::flush;
                    }
                    // std::cout<<"chkp3ab"<<std::endl<<std::flush;
                }
                // std::cout<<"chkp3b"<<std::endl<<std::flush;
            
            }
            // std::cout<<"chkp3d"<<std::endl<<std::flush;
            
            noofDel = fsc_.nodes.size() - noofDel;

            /////////////////////////////////
            // noofDel=0;              //////
            /////////////////////////////////

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
            
            // std::cout<<"New Vects= "<<w.size()<<" Noof Dels= "<<noofDel<<std::endl<<std::flush; 
            
            int noofIds = (int)(fsc_.nodes.size());
            std::vector<int> oldIds(0);
            for(int i=0;i<noofIds;i++)
                oldIds.push_back(fsc_.nodes[i].iden);
            
            // std::sort(oldIds.begin(),oldIds.end());

            std::map<int,int> oldToNewID;
            for(int i=0;i<noofIds;i++)
                oldToNewID[oldIds[i]]=i;

            for(int i=0;i<noofIds;i++)
            {
                fsc_.nodes[i].iden = oldToNewID[fsc_.nodes[i].iden];
                if(howard_ == 1)
                {
                    fsc_.nodes[i].mark=0;
                }
                for(int j=0;j<(int)O;j++)
                {
                    fsc_.nodes[i].obsSucc[j] = 
                            oldToNewID[fsc_.nodes[i].obsSucc[j]];
                }
            }        
            fsc_.maxId = fsc_.nodes.size();
        }
    }


    // ValueFunction policyEval()
    // {
    // }
}

            