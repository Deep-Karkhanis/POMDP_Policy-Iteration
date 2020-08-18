#ifndef AI_TOOLBOX_POMDP_PBPI_HEADER_FILE
#define AI_TOOLBOX_POMDP_PBPI_HEADER_FILE

#include <limits>

#include <AIToolbox/Utils/Probability.hpp>
#include <AIToolbox/Utils/Prune.hpp>
#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/POMDP/TypeTraits.hpp>
#include <AIToolbox/POMDP/Utils.hpp>
#include <AIToolbox/POMDP/Algorithms/Utils/Projecter.hpp>
#include <AIToolbox/POMDP/Algorithms/PBVI.hpp>

// #include <AIToolbox/POMDP/Algorithms/Utils/Pruner.hpp>
// #include <AIToolbox/POMDP/Algorithms/Utils/WitnessLP.hpp>

#include <Eigen/Dense>
#include <iostream>

#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

#include <queue>
#include <algorithm>
#include <map>

// #include <map>

namespace AIToolbox::POMDP {
    /**
     * @brief This class implements the Incremental Pruning algorithm.
     *
     * This algorithm solves a POMDP Model perfectly. It computes solutions
     * for each horizon incrementally, every new solution building upon the
     * previous one.
     *
     * From each solution, it computes the full set of possible
     * projections. It then computes all possible cross-sums of such
     * projections, in order to compute all possible vectors that can be
     * included in the final solution.
     *
     * What makes this method unique is its pruning strategy. Instead of
     * generating every possible vector, combining them and pruning, it
     * tries to prune at every possible occasion in order to minimize the
     * number of possible vectors at any given time. Thus it will prune
     * after creating the projections, after every single cross-sum, and
     * in the end when combining all projections for each action.
     *
     * The performances of this method are *heavily* dependent on the linear
     * programming methods used. In particular, this code currently
     * utilizes the lp_solve55 library. However, this library is not the
     * most efficient implementation, as it defaults to a somewhat slow
     * solver, and its problem-building API also tends to be slow due to
     * lots of bounds checking (which are cool, but sometimes people know
     * what they are doing). Still, to avoid replicating infinite amounts
     * of code and managing memory by ourselves, we use its API. It would
     * be nice if one day we could port directly into the code a fast lp
     * implementation; for now we do what we can.
     */
    class PBPI {
        public:
            /**
             * @brief Basic constructor.
             *
             * This constructor sets the default horizon used to solve a POMDP::Model.
             *
             * The epsilon parameter must be >= 0.0, otherwise the
             * constructor will throw an std::runtime_error. The epsilon
             * parameter sets the convergence criterion. An epsilon of 0.0
             * forces IncrementalPruning to perform a number of iterations
             * equal to the horizon specified. Otherwise, IncrementalPruning
             * will stop as soon as the difference between two iterations
             * is less than the epsilon specified.
             *
             * @param h The horizon chosen.
             * @param epsilon The epsilon factor to stop the IncrementalPruning loop.
             */
            PBPI(size_t nBeliefs, unsigned h, double epsilon, double l1disteps, AIToolbox::POMDP::Belief b);

            /**
             * @brief This function sets the epsilon parameter.
             *
             * The epsilon parameter must be >= 0.0, otherwise the
             * constructor will throw an std::runtime_error. The epsilon
             * parameter sets the convergence criterion. An epsilon of 0.0
             * forces IncrementalPruning to perform a number of iterations
             * equal to the horizon specified. Otherwise, IncrementalPruning
             * will stop as soon as the difference between two iterations
             * is less than the epsilon specified.
             *
             * @param e The new epsilon parameter.
             */
            void setTolerance(double t);

            /**
             * @brief This function allows setting the horizon parameter.
             *
             * @param h The new horizon parameter.
             */
            void setHorizon(unsigned h);

            /**
             * @brief This function sets a new number of support beliefs.
             *
             * @param nBeliefs The new number of support beliefs.
             */
            void setBeliefSize(size_t nBeliefs);

            /**
             * @brief This function sets a new epsilon threshold for L1 dist.
             *
             * @param eps The epsilon threshold to eliminate nearby point.
             */
            void setL1DistEps(size_t eps);

            /**
             * @brief This function will return the currently set epsilon parameter.
             *
             * @return The currently set epsilon parameter.
             */
            double getTolerance() const;

            /**
             * @brief This function returns the currently set horizon parameter.
             *
             * @return The current horizon.
             */
            unsigned getHorizon() const;

            /**
             * @brief This function returns the currently set number of support beliefs to use during a solve pass.
             *
             * @return The number of support beliefs.
             */
            size_t getBeliefSize() const;

            /**
             * @brief This function returns the currently set threshold for minimum L1 distance between points in the belief set.
             *
             * @return The threshold for minimum L1 distance.
             */
            size_t getL1DistEps() const;

            /**
             * @brief This function solves a POMDP::Model completely.
             *
             * This function is pretty expensive (as are possibly all POMDP
             * solvers).  It generates for each new solved timestep the
             * whole set of possible ValueFunctions, and prunes it
             * incrementally, trying to reduce as much as possible the
             * linear programming solves required.
             *
             * This function returns a tuple to be consistent with MDP
             * solving methods, but it should always succeed.
             *
             * @tparam M The type of POMDP model that needs to be solved.
             *
             * @param model The POMDP model that needs to be solved.
             *
             * @return A tuple containing the maximum variation for the
             *         ValueFunction and the computed ValueFunction.
             */
            template <typename M, typename = std::enable_if_t<is_model<M>::value>, typename G>
            std::tuple<double, ValueFunction> operator()(const M & model, G &rand);

        private:
            /**
             * @brief This function computes a VList composed of all possible combinations of sums of the VLists provided.
             *
             * This function performs the job of accumulating the
             * information required to obtain the final policy. Cross-sums
             * are done between each element of each list. The resulting
             * observation output depend on the order parameter, which
             * specifies whether the first list should be put first, or the
             * second one. This parameter is needed due to the order in
             * which we cross-sum all vectors.
             *
             * @param l1 The "main" parent list.
             * @param l2 The list being cross-summed to l1.
             * @param a The action that this cross-sum is about.
             * @param order Which list comes before the other to merge
             *              observations. True for the first, false otherwise.
             *
             * @return The cross-sum between l1 and l2.
             */
            VList crossSum(const VList & l1, const VList & l2, size_t a, bool order);

            void updateFSC(const VList &w);
            void makeInitialFSC();

            size_t S, A, O, beliefSize_;;
            unsigned horizon_;
            double tolerance_;
            double l1disteps_;
            int howard_, IMthd_, nodeLimit_;
            AIToolbox::POMDP::Belief bel_;

            struct MchnState{
                MDP::Values values; //alpha vector is stored here
                size_t action;
                // std::vector<*MchnState> * adjList;
                VObs obsSucc;
                size_t iden; //identifier
                int mark;

                MchnState()
                {}

                MchnState(MDP::Values v, size_t a, VObs vo, size_t id)
                {
                    values = v;
                    action = a;
                    for(int oi = 0;oi<vo.size();oi++)
                        obsSucc.push_back(vo[oi]);
                    iden = id;
                    mark = 0;
                }
            };

            struct FSC{
                std::vector<MchnState> nodes;
                size_t maxId;
                double maxE;
                // std::vector<std::vector<*MchnState> > edges;

                FSC(){
                  nodes = std::vector<MchnState>(0);
                  maxId = 0;
                //   edges = std::vector<std::vector<*MchnState> >(0);
                }

            };

            FSC fsc_, fsc_F;


    };

    template <typename M, typename, typename G>
    std::tuple<double, ValueFunction> PBPI::operator()(const M & model, G &rand) {
        // Initialize "global" variables
        S = model.getS();
        A = model.getA();
        O = model.getO();

        // <tf
        std::vector<AIToolbox::POMDP::Belief> bList;
        bList.emplace_back(bel_);

        PBVI pbvisolver(beliefSize_, 1, tolerance_);

        // tf>
        makeInitialFSC(); // TODO: May take user input
                                            // S,A,O are object variables

            // std::cout<<"chkpA"<<std::flush;

        unsigned timestep = 0;

        Pruner prune(S);
        Projecter projecter(model);

        const bool useTolerance = checkDifferentSmall(tolerance_, 0.0);
        double variation = tolerance_ * 2; // Make it bigger

        // std::cout<<"chkp1"<<std::flush;

        Eigen::Matrix<double, Eigen::Dynamic,Eigen::Dynamic> coeffMat;
        Eigen::Matrix<double, Eigen::Dynamic, 1> valMat;
        Eigen::Matrix<double,Eigen::Dynamic, 1> solnVec;

        // std::cout<<"-----1-FSC------"<<std::endl;
        // for(int mchSt=0;mchSt<(int)fsc_.nodes.size();mchSt++)
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
        //         std::cout<<std::endl<<std::flush;



        // while ( timestep < horizon_ && ( !useEpsilon || variation > epsilon_ ) )
        std::cout<<"started with horizon = "<<horizon_<<"and node lim = "<<nodeLimit_<<"\n";
        std::cout<<"howard = "<<howard_<<", IMthd = "<<IMthd_<<"\n";
        while ( timestep<horizon_ && (fsc_.nodes.size() <= nodeLimit_) && ( !useTolerance || variation > tolerance_ ) )
        {
            ++timestep;
            std::cout<<"timestep = "<<timestep<<"\n";
            // std::cout<<"$$$$$$$$$$$$$$$$$$$$$                   $$$$$$$$$$$$$$$$$$$$$$$"<<std::endl<<std::flush;
            // std::cout<<"$$$$$$$$$$$$$$$$$$$$$     ITERATION "<<timestep<<"   $$$$$$$$$$$$$$$$$$$$$$$"<<std::endl<<std::flush;
            // std::cout<<"$$$$$$$$$$$$$$$$$$$$$                   $$$$$$$$$$$$$$$$$$$$$$$"<<std::endl<<std::flush;

            // std::cout<<"chkpB"<<std::flush;

            //Policy Evaluation
            int FSCsize = fsc_.nodes.size();
            int MatDim = FSCsize*S;
            coeffMat.resize(MatDim,MatDim);
            valMat.resize(MatDim,1);

            // std::cout<<"chkpC"<<std::flush;


            for(int i=0;i<MatDim;i++)
            {
                for(int j=0;j<MatDim;j++)
                    coeffMat(i,j)=0;
                valMat(i,0) = 0;
            }

            // std::cout<<"chkp1"<<std::flush;

            double discF = model.getDiscount();
            for(int alphaV=0;alphaV<FSCsize;alphaV++)
            {
                int vecAction = fsc_.nodes[alphaV].action;
                for(int iState=0;iState<(int)S;iState++)
                {
                    int coorRow = (alphaV*S)+iState;
                    for(int fstate=0;fstate<(int)S;fstate++)
                    {
                        double pij = model.getTransitionProbability(iState,vecAction,fstate);

                        for(int ob = 0;ob<(int)O;ob++)
                        {
                            int succVect = fsc_.nodes[alphaV].obsSucc[ob];
                            int coorCol = (succVect*S) + fstate;
                            double qjth = model.getObservationProbability(fstate,vecAction,ob);
                            coeffMat(coorRow,coorCol) -= (discF*pij*qjth);
                        }
                        double rij = model.getExpectedReward(iState,vecAction,fstate);
                        valMat(coorRow,0) += (pij*rij);
                    }
                    coeffMat(coorRow,coorRow) += 1;
                }
            }

            // std::cout<<"------CoeffMat------"<<std::endl;
            // for(int i=0;i<MatDim;i++)
            // {
            //     for(int j=0;j<MatDim;j++)
            //         std::cout<<coeffMat(i,j)<<" ";
            //     std::cout<<std::endl;
            // }

            //scope of improvement here
            // Eigen::ColPivHouseholderQR<Eigen::Matrix<double, Eigen::Dynamic,Eigen::Dynamic> > dec(coeffMat);
            // Eigen::Matrix<double,Eigen::Dynamic, 1> solnVec = dec.solve(valMat);

            // std::cout<<"chkp1"<<std::flush;


            solnVec.resize(MatDim,1);
            // std::cout<<"chkp1a"<<std::flush;


            solnVec = coeffMat.colPivHouseholderQr().solve(valMat);
            // std::cout<<"chkp1b"<<std::flush;
            // std::cout<<"chkp1c"<<solnVec.rows()<<std::flush;

            // auto solnVec = coeffMat.colPivHouseholderQr().solve(valMat);


            // double maxExp=fsc_.nodes[0].values.dot(bel_);
            double maxExp = -std::numeric_limits<double>::max();
            int maxNode = 0;
            VList alpVecs = std::vector<VEntry>(FSCsize);
            for(int i=0;i<FSCsize;i++)
            {
                alpVecs[i].observations = fsc_.nodes[i].obsSucc;
                alpVecs[i].action = fsc_.nodes[i].action;
                alpVecs[i].values.resize(S,1);
                for(int j=0;j<(int)S;j++)
                {
                    // std::cout<<"chkp1d"<<std::flush;

                    int index = i*S + j;
                    alpVecs[i].values(j,0) = solnVec(index,0);
                    if(timestep==1) fsc_.nodes[i].values[j] = alpVecs[i].values(j,0);
                    // alpVecs[i].values(j,0) = fsc_.nodes[i].values[j];

                    // std::cout<<"chkp1e"<<std::flush;

                }

                double val = (fsc_.nodes[i].values.dot(bel_));
                if(maxExp < val)
                {
                    maxExp = val;
                    maxNode = i;
                }
                // std::cout<<"chkp1c"<<std::flush;
            }
            std::cout<<"maxExp at b0: "<<maxExp<<" by Node "<<maxNode<<"\n";
            fsc_.nodes[maxNode].mark = 1;

            // std::cout<<"------EvalRes------"<<std::endl;
            // for(int mchSt=0;mchSt<FSCsize;mchSt++)
            // {
            //     std::cout<<"--"<<mchSt<<"--"<<std::endl;
            //     std::cout<<alpVecs[mchSt].action<<std::endl;
            //     for(int valI=0;valI<(int)S;valI++)
            //     {
            //         std::cout<<alpVecs[mchSt].values(valI,0)<<" ";
            //     }
            //     std::cout<<std::endl;
            //     for(int valI=0;valI<(int)O;valI++)
            //     {
            //         std::cout<<alpVecs[mchSt].observations[valI]<<" ";
            //     }
            //     std::cout<<std::endl;
            // }

            // ValueFunction v = policyEval();

            // std::cout<<"chkp1"<<std::flush;

            if(howard_ == 0)
            {
                int noofDel=0;
                //std::cout<<fsc_.nodes.size()<<std::endl<<std::flush;
                // std::cout<<"chkp3a"<<std::endl<<std::flush;
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
                            if(fsc_.nodes[succInd].mark != 2)
                            {
                                fsc_.nodes[succInd].mark = 2;
                                noofDel++;
                                stateQueue.push(succInd);
                            }
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
                    fsc_.nodes[i].mark=0;
                    for(int j=0;j<(int)O;j++)
                    {
                        fsc_.nodes[i].obsSucc[j] =
                                oldToNewID[fsc_.nodes[i].obsSucc[j]];
                    }
                }
                fsc_.maxId = fsc_.nodes.size();
            }



            // Compute all possible outcomes, from our previous results.
            // This means that for each action-observation pair, we are going
            // to obtain the same number of possible outcomes as the number
            // of entries in our initial vector w.

            // auto projs = projecter(v[timestep-1]);

            int rnd_numb;
            std::vector<int> upd_nodes(0);
            srand(time(NULL));
            int nodesChosen=0;
            while(nodesChosen == 0)
            {
                nodesChosen=0;
                for(int indNode=0;indNode<FSCsize;indNode++)
                {
                    rnd_numb = rand() % 2;
                    if(rnd_numb == 1)
                    {
                        upd_nodes.push_back(indNode);
                        nodesChosen++;
                    }
                }
            }
            if(howard_==0 &&  IMthd_ == 2)
            {
                VList alp_subS(0);
                for(size_t alpInd=0;alpInd<(alpVecs.size()); alpInd++)
                {
                    if(std::find(upd_nodes.begin(),upd_nodes.end(),alpInd)
                                != upd_nodes.end())
                    {
                        alp_subS.push_back(alpVecs[alpInd]);
                    }
                }
                alpVecs = alp_subS;
            }



            //ADD PBVI ALGO HERE
            ValueFunction v;
            v.emplace_back(alpVecs);
            auto var_and_v = pbvisolver(model, bList, v);
            variation = std::get<0>(var_and_v);
            auto w = (std::get<1>(var_and_v)).back();
            maxExp = -std::numeric_limits<double>::max();
            double maxind = 0;
            for(int i=0; i<w.size(); i++){
                double val = w[i].values.dot(bel_);
                // std::cout<<"new Node "<<i<<": "<<w[i].action<<"; ";
                // for(int j=0; j<O; j++) std::cout<<w[i].observations[j]<<" ";
                // std::cout<<"\n";
                if(maxExp < val){
                    maxExp = val;
                    maxind = i;
                }
            }
            std::cout<<"maxExp at b0: "<<maxExp<<" by Node "<<maxind<<"(V [backup])\n";
            size_t belief_points = bList.size();
            std::cout<<fsc_.nodes.size()<<", "<<w.size()<<", "<<belief_points<<"\n";
            for(int i=0; i<belief_points; i++)
            {
                std::vector<Belief> new_beliefs;
                double max_L1_dist = 0; //have to set it to epsilon later
                size_t max_b_ind = 0;
                for(int act=0; act<A; act++)
                {
                    ProbabilityVector z_prob(O);
                    for(int ob=0; ob<O; ob++)
                        z_prob[ob] = updateBeliefUnnormalized(model, bList[i], act, ob).sum();
                    // std::default_random_engine rand(AIToolbox::Impl::Seeder::getSeed());
                    auto z = AIToolbox::sampleProbability(O, z_prob, rand);
                    new_beliefs.emplace_back(updateBelief(model, bList[i], act, z));
                    double l1disttoB = (bList[0]-new_beliefs.back()).lpNorm<1>();
                    for(int ind=1; ind<belief_points; ind++)
                    {
                        double norm = (bList[ind]-new_beliefs.back()).lpNorm<1>();
                        if(norm < l1disttoB) l1disttoB = norm;
                    }
                    if(l1disttoB >= max_L1_dist)
                    {
                        max_L1_dist = l1disttoB;
                        max_b_ind = act;
                    }
                }
                if(max_L1_dist > l1disteps_) bList.emplace_back(new_beliefs[max_b_ind]);
                // bList[i]
            }




            // std::cout<<"chkp1"<<std::flush;

                    // std::cout<<"------Timestep"<<timestep<<"------"<<std::endl;


                    // for(int mchSt=0;mchSt<(int)fsc_.nodes.size();mchSt++)
                    // {
                    //     std::cout<<"--"<<mchSt<<"--"<<std::endl;
                    //     std::cout<<fsc_.nodes[mchSt].action<<std::endl;
                    //     for(int valI=0;valI<(int)S;valI++)
                    //     {
                    //         std::cout<<fsc_.nodes[mchSt].values[valI]<<" ";
                    //     }
                    //     std::cout<<std::endl;
                    //     for(int valI=0;valI<(int)O;valI++)
                    //     {
                    //         std::cout<<fsc_.nodes[mchSt].obsSucc[valI]<<" ";
                    //     }
                    //     std::cout<<std::endl;
                    // }
                    // std::cout<<std::endl;

                    // auto tmp = fsc_.nodes[0];
                    // VEntry tmp1;
                    // tmp1.action=tmp.action;
                    // tmp1.values=tmp.values;
                    // for(int tmpi=0;tmpi<(int)S;tmpi++)
                    // {
                    //     tmp1.values(tmpi,0)++;
                    // }
                    // tmp1.observations=tmp.obsSucc;
                    // w.push_back(tmp1);


                    // std::cout<<"------w------"<<std::endl;
                    // for(int mchSt=0;mchSt<(int)w.size();mchSt++)
                    // {
                    //     std::cout<<"--"<<mchSt<<"--"<<std::endl;
                    //     std::cout<<w[mchSt].action<<std::endl;
                    //     for(int valI=0;valI<(int)S;valI++)
                    //     {
                    //         std::cout<<w[mchSt].values(valI,0)<<" ";
                    //     }
                    //     std::cout<<std::endl;
                    //     for(int valI=0;valI<(int)O;valI++)
                    //     {
                    //         std::cout<<w[mchSt].observations[valI]<<" ";
                    //     }
                    //     std::cout<<std::endl;
                    // }
                    // std::cout<<std::endl;

            // %%%
            // update nodes already chosen
            // %%%
            if(howard_==0 && IMthd_==1)
            {
                VList w1(0);
                for(size_t alpDInd=0;alpDInd<(w.size()); alpDInd++)
                {
                    auto alpD = w[alpDInd];
                    for(int obsInd=0;obsInd<O;obsInd++)
                    {
                        if(std::find(upd_nodes.begin(),upd_nodes.end(),alpD.observations[obsInd])
                                != upd_nodes.end())
                        {
                                w1.push_back(alpD);
                                break;
                        }
                    }
                }
                updateFSC(w1);
            }
            else
            {
                updateFSC(w);
            }
            // v.emplace_back(std::move(w));
            // std::cout<<"chkpAA"<<std::flush;

            // Check convergence
            //  if ( useTolerance )
            //      variation = weakBoundDistance(v[timestep-1], v[timestep]);
            //  }
            //  std::cout<<timestep<<" "<<horizon_<<std::flush;
            //  std::cout<<"chkpAB"<<std::flush;

             solnVec.resize(0,1);
            //  std::cout<<"chkpAB"<<std::flush;
             valMat.resize(0,1);
            //  std::cout<<"chkpAC"<<std::flush;
             coeffMat.resize(0,0);
            //  std::cout<<"chkpAD"<<std::flush;
            // std::cout<<"FSC Size "<<fsc_.nodes.size()<<std::endl<<std::flush;
            // std::cout<<"------FSC------"<<std::endl;
            // for(int mchSt=0;mchSt<(int)fsc_.nodes.size();mchSt++)
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

        // std::cout<<"chkp4"<<std::endl<<std::flush;

        if(howard_ == 0)
        {
            double maxExp=fsc_.nodes[0].values.dot(bel_);
            int maxNode = 0;
            for(int i=0;i<fsc_.nodes.size();i++)
            {
                double val = (fsc_.nodes[i].values.dot(bel_));
                if(maxExp < val)
                {
                    maxExp = val;
                    maxNode = i;
                }
                // std::cout<<"chkp1c"<<std::flush;
            }
            fsc_.nodes[maxNode].mark = 1;

            int noofDel=0;
            //std::cout<<fsc_.nodes.size()<<std::endl<<std::flush;
            // std::cout<<"chkp3a"<<std::endl<<std::flush;
            for(int newFN = 0; newFN < (int)(fsc_.nodes.size()); newFN++)
            {
                // std::cout<<"chkp3aa"<<std::endl<<std::flush;

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
                        if(fsc_.nodes[succInd].mark != 2)
                        {
                            fsc_.nodes[succInd].mark = 2;
                            noofDel++;
                            stateQueue.push(succInd);
                        }
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
            // std::cout<<"chkp3da"<<std::endl<<std::flush;

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
                fsc_.nodes[i].mark=0;
                for(int j=0;j<(int)O;j++)
                {
                    fsc_.nodes[i].obsSucc[j] =
                            oldToNewID[fsc_.nodes[i].obsSucc[j]];
                }
            }
            fsc_.maxId = fsc_.nodes.size();

            // std::cout<<"chkp3da"<<std::endl<<std::flush;



        }

        ValueFunction v(0);
        VList vl(0);
        for(int i=0;i<(int)(fsc_.nodes.size());i++)
        {
            VEntry ve(fsc_.nodes[i].values,fsc_.nodes[i].action,fsc_.nodes[i].obsSucc);
            vl.push_back(ve);
        }
        v.push_back(vl);

        // std::cout<<"chkp4b"<<std::endl<<std::flush;

        // v.push_back(vl);

            // std::cout<<"------FSC------"<<std::endl;
            // for(int mchSt=0;mchSt<(int)fsc_.nodes.size();mchSt++)
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
            //         std::cout<<std::endl<<std::flush;

        std::cout<<"FSC Size "<<fsc_.nodes.size()<<std::endl<<std::flush;


        return std::make_tuple(useTolerance ? variation : 0.0, v);
    }
}

#endif
