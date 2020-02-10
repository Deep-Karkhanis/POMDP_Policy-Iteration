#ifndef AI_TOOLBOX_POMDP_POLICY_ITERATION_HEADER_FILE
#define AI_TOOLBOX_POMDP_POLICY_ITERATION_HEADER_FILE

#include <limits>

#include <AIToolbox/Utils/Probability.hpp>
#include <AIToolbox/Utils/Prune.hpp>
#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/POMDP/TypeTraits.hpp>
#include <AIToolbox/POMDP/Utils.hpp>
#include <AIToolbox/POMDP/Algorithms/Utils/Projecter.hpp>

// #include <AIToolbox/POMDP/Algorithms/Utils/Pruner.hpp>
// #include <AIToolbox/POMDP/Algorithms/Utils/WitnessLP.hpp>

#include <Eigen/Dense>
#include <iostream>

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
    class PolicyIteration {
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
            PolicyIteration(unsigned h, double epsilon);

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
            template <typename M, typename = std::enable_if_t<is_model<M>::value>>
            std::tuple<double, ValueFunction> operator()(const M & model);

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

            size_t S, A, O;
            unsigned horizon_;
            double tolerance_;

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
                    obsSucc = vo;
                    iden = id;
                    mark = 0;
                }
            };

            struct FSC{
                std::vector<MchnState> nodes;
                size_t maxId;
                // std::vector<std::vector<*MchnState> > edges;

                FSC(){
                  nodes = std::vector<MchnState>(0);
                  maxId = 0;
                //   edges = std::vector<std::vector<*MchnState> >(0);
                }
                
            };

            FSC fsc_;


    };

    template <typename M, typename>
    std::tuple<double, ValueFunction> PolicyIteration::operator()(const M & model) {
        // Initialize "global" variables
        S = model.getS();
        A = model.getA();
        O = model.getO();

        makeInitialFSC(); // TODO: May take user input
                                            // S,A,O are object variables

        unsigned timestep = 0;

        Pruner prune(S);
        Projecter projecter(model);

        const bool useTolerance = checkDifferentSmall(tolerance_, 0.0);
        double variation = tolerance_ * 2; // Make it bigger
        
        // std::cout<<"chkp1"<<std::flush;
        
        Eigen::Matrix<double, Eigen::Dynamic,Eigen::Dynamic> coeffMat;            
        Eigen::Matrix<double, Eigen::Dynamic, 1> valMat;
        Eigen::Matrix<double,Eigen::Dynamic, 1> solnVec;


        // while ( timestep < horizon_ && ( !useEpsilon || variation > epsilon_ ) ) 
        while ( timestep<horizon_ && (fsc_.nodes.size() < 1000) && ( !useTolerance || variation > tolerance_ ) )                          
        {
            ++timestep;

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

                    // std::cout<<"chkp1e"<<std::flush;
            
                }
                // std::cout<<"chkp1c"<<std::flush;
            
            }
            
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
        

            // Compute all possible outcomes, from our previous results.
            // This means that for each action-observation pair, we are going
            // to obtain the same number of possible outcomes as the number
            // of entries in our initial vector w.
            
            // auto projs = projecter(v[timestep-1]);
            auto projs = projecter(alpVecs);

            size_t finalWSize = 0;
            // In this method we split the work by action, which will then
            // be joined again at the end of the loop.
            for ( size_t a = 0; a < A; ++a ) {
                // We prune each outcome separately to be sure
                // we do not replicate work later.
                for ( size_t o = 0; o < O; ++o ) {
                    const auto begin = std::begin(projs[a][o]);
                    const auto end   = std::end  (projs[a][o]);
                    projs[a][o].erase(prune(begin, end, unwrap), end);
                }

                // Here we reduce at the minimum the cross-summing, by alternating
                // merges. We pick matches like a reverse binary tree, so that
                // we always pick lists that have been merged the least.
                //
                // Example for O==7:
                //
                //  0 <- 1    2 <- 3    4 <- 5    6
                //  0 ------> 2         4 ------> 6
                //            2 <---------------- 6
                //
                // In particular, the variables are:
                //
                // - oddOld:   Whether our starting step has an odd number of elements.
                //             If so, we skip the last one.
                // - front:    The id of the element at the "front" of our current pass.
                //             note that since passes can be backwards this can be high.
                // - back:     Opposite of front, which excludes the last element if we
                //             have odd elements.
                // - stepsize: The space between each "first" of each new merge.
                // - diff:     The space between each "first" and its match to merge.
                // - elements: The number of elements we have left to merge.

                bool oddOld = O % 2;
                int i, front = 0, back = O - oddOld, stepsize = 2, diff = 1, elements = O;
                while ( elements > 1 ) {
                    for ( i = front; i != back; i += stepsize ) {
                        projs[a][i] = crossSum(projs[a][i], projs[a][i + diff], a, stepsize > 0);
                        const auto begin = std::begin(projs[a][i]);
                        const auto end   = std::end  (projs[a][i]);
                        projs[a][i].erase(prune(begin, end, unwrap), end);
                        --elements;
                    }

                    const bool oddNew = elements % 2;

                    const int tmp   = back;
                    back      = front - ( oddNew ? 0 : stepsize );
                    front     = tmp   - ( oddOld ? 0 : stepsize );
                    stepsize *= -2;
                    diff     *= -2;

                    oddOld = oddNew;
                }
                // Put the result where we can find it
                if (front != 0)
                    projs[a][0] = std::move(projs[a][front]);
                finalWSize += projs[a][0].size();
            }
            VList w;
            w.reserve(finalWSize);

            // Here we don't have to do fancy merging since no cross-summing is involved
            for ( size_t a = 0; a < A; ++a )
                 w.insert(std::end(w), std::make_move_iterator(std::begin(projs[a][0])), std::make_move_iterator(std::end(projs[a][0])));

            // We have them all, and we prune one final time to be sure we have
            // computed the parsimonious set of value functions.
            const auto begin = std::begin(w);
            const auto end   = std::end  (w);
            w.erase(prune(begin, end, unwrap), end);
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
            
                    
            updateFSC(w);
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
            std::cout<<"FSC Size "<<fsc_.nodes.size()<<std::endl<<std::flush;

        }

        ValueFunction v(0);
        VList vl(0);
        for(int i=0;i<(int)(fsc_.nodes.size());i++)
        {
            VEntry ve(fsc_.nodes[i].values,fsc_.nodes[i].action,fsc_.nodes[i].obsSucc);
            vl.push_back(ve);
        }
        v.push_back(vl);
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
            //         std::cout<<std::endl;
            
            
        return std::make_tuple(useTolerance ? variation : 0.0, v);
    }
}

#endif
