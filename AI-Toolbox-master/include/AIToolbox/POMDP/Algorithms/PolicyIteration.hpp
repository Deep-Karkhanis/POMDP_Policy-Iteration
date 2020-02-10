#ifndef AI_TOOLBOX_POMDP_POLICY_ITERATION_HEADER_FILE
#define AI_TOOLBOX_POMDP_POLICY_ITERATION_HEADER_FILE

#include <limits>

#include <AIToolbox/Utils/Probability.hpp>
#include <AIToolbox/Utils/Prune.hpp>
#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/POMDP/TypeTraits.hpp>
#include <AIToolbox/POMDP/Utils.hpp>
#include <AIToolbox/POMDP/Algorithms/Utils/Projecter.hpp>

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
            PolicyIteration(unsigned h, double epsilon, AIToolbox::POMDP::Belief b);

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
            
            template <typename M, typename = std::enable_if_t<is_model<M>::value>>
            VList policyEvaluation(const M & model,int doEval);
   

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
            int B_,howard_, nodeLimit_;
            AIToolbox::POMDP::Belief bel_;

            FSC fsc_, fsc_F;
    };



    template <typename M, typename>
    std::tuple<double, ValueFunction> PolicyIteration::operator()(const M & model) {
        // Initialize "global" variables
        S = model.getS();
        A = model.getA();
        O = model.getO();

        makeInitialFSC(); // TODO: May take user input
        unsigned timestep = 0;

        // srand(time(NULL));
        srand(1);

        Pruner prune(S);
        Projecter projecter(model);

        const bool useTolerance = checkDifferentSmall(tolerance_, 0.0);
        double variation = tolerance_ * 2; // Make it bigger
        
        // std::cout<<"chkpA"<<std::endl<<std::flush;
        VList alpVecs = policyEvaluation(model,1);
        // std::cout<<"------alp------"<<std::endl;
        // for(int mchSt=0;mchSt<(int)alpVecs.size();mchSt++)
        // {
        //     std::cout<<"--"<<mchSt<<"--"<<alpVecs[mchSt].action<<std::endl;
        //     for(int valI=0;valI<(int)S;valI++)
        //         std::cout<<alpVecs[mchSt].values(valI,0)<<" ";
        //     std::cout<<std::endl;
        //     for(int valI=0;valI<(int)O;valI++)
        //         std::cout<<alpVecs[mchSt].observations[valI]<<" ";
        //     std::cout<<std::endl;
        // }
            
        while ( timestep<horizon_ && (fsc_F.nodes.size() <= nodeLimit_) && ( !useTolerance || variation > tolerance_ ) )                          
        {
            ++timestep;
            auto projs = projecter(alpVecs);
            // for ( size_t a = 0; a < A; ++a ) {
            //     std::cout<<a<<std::endl;
            //     for ( size_t o = 0; o < O; ++o ) {
            //         std::cout<<" "<<o<<std::endl;
            //         for(size_t x=0;x<projs[a][o].size();x++){
            //             std::cout<<"  "<<projs[a][o][x].action<<std::endl<<"  ";
            //             for(int valI=0;valI<(int)S;valI++)
            //                 std::cout<<projs[a][o][x].values[valI]<<" ";
            //             std::cout<<std::endl<<"  ";
            //             for(int valI=0;valI<(int)O;valI++)
            //                 std::cout<<projs[a][o][x].observations[valI]<<" ";
            //             std::cout<<std::endl<<std::endl;
            //         }    
            //     }
            // }

            // std::cout<<"------alp------"<<std::endl;
            // for(int mchSt=0;mchSt<(int)alpVecs.size();mchSt++)
            // {
            //     std::cout<<"--"<<mchSt<<"--"<<alpVecs[mchSt].action<<std::endl;
            //     for(int valI=0;valI<(int)S;valI++)
            //         std::cout<<alpVecs[mchSt].values(valI,0)<<" ";
            //     std::cout<<std::endl;
            //     for(int valI=0;valI<(int)O;valI++)
            //         std::cout<<alpVecs[mchSt].observations[valI]<<" ";
            //     std::cout<<std::endl;
            // }
        

            size_t finalWSize = 0;
            for ( size_t a = 0; a < A; ++a ) {
                for ( size_t o = 0; o < O; ++o ) {
                    const auto begin = std::begin(projs[a][o]);
                    const auto end   = std::end  (projs[a][o]);
                    projs[a][o].erase(prune(begin, end, unwrap), end);
                }
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
                if (front != 0)
                    projs[a][0] = std::move(projs[a][front]);
                finalWSize += projs[a][0].size();
            }
            VList w;
            w.reserve(finalWSize);

            for ( size_t a = 0; a < A; ++a )
                 w.insert(std::end(w), std::make_move_iterator(std::begin(projs[a][0])), std::make_move_iterator(std::end(projs[a][0])));

            const auto begin = std::begin(w);
            const auto end   = std::end  (w);
            w.erase(prune(begin, end, unwrap), end);

            // std::cout<<"chkpB"<<std::endl<<std::flush;
            if(howard_==0)
            {
                auto fsc_opt = fsc_F;
                auto alpVecs_opt = alpVecs;
                int maxE = INT_MIN;
                for(int branch=0;branch<B_;branch++)
                {
                    VList w1(0);
                    fsc_=fsc_F;
                    int rnd_numb;
                    int vecsChosen=0;
                    while(vecsChosen == 0)
                    {
                        vecsChosen=0;
                        for(size_t alpDInd=0;alpDInd<(w.size()); alpDInd++)
                        {
                            rnd_numb = rand() % 2;
                            if(rnd_numb == 1)
                            {
                                auto alpD = w[alpDInd];
                                for(int obsInd=0;obsInd<O;obsInd++)
                                {
                                    w1.push_back(alpD);
                                }
                                vecsChosen++;
                            }
                        }
                    }      
                    updateFSC(w1);
                    alpVecs = policyEvaluation(model,fsc_.changed);
            
                    if(maxE < fsc_.maxE)
                    {
                        fsc_opt = fsc_;    
                        alpVecs_opt = alpVecs;           
                    }
                    else if(maxE == fsc_.maxE)
                    {
                        rnd_numb = rand() % 2;
                        if(rnd_numb == 1)
                        {
                            fsc_opt = fsc_;    
                            alpVecs_opt = alpVecs;           
                        }
                    }
                }
                fsc_F = fsc_opt;
                alpVecs = alpVecs_opt;
            }
            else
            {
                fsc_ = fsc_F;
                updateFSC(w);
                alpVecs = policyEvaluation(model,fsc_.changed);
                fsc_F = fsc_;
            }
            std::cout<<"FSC Size "<<fsc_F.nodes.size()<<std::endl<<std::flush;

            // std::cout<<"------w------"<<std::endl;
            // for(int mchSt=0;mchSt<(int)w.size();mchSt++)
            // {
            //     std::cout<<"--"<<mchSt<<"--"<<w[mchSt].action<<std::endl;
            //     for(int valI=0;valI<(int)S;valI++)
            //         std::cout<<w[mchSt].values(valI,0)<<" ";
            //     std::cout<<std::endl;
            //     for(int valI=0;valI<(int)O;valI++)
            //         std::cout<<w[mchSt].observations[valI]<<" ";
            //     std::cout<<std::endl;
            // }
            // std::cout<<"------FSC_------"<<std::endl;
            // for(int mchSt=0;mchSt<(int)fsc_.nodes.size();mchSt++)
            // {
            //     std::cout<<"--"<<mchSt<<"--"<<fsc_.nodes[mchSt].action<<std::endl;
            //     for(int valI=0;valI<(int)S;valI++)
            //         std::cout<<fsc_.nodes[mchSt].values[valI]<<" ";
            //     std::cout<<std::endl;
            //     for(int valI=0;valI<(int)O;valI++)
            //         std::cout<<fsc_.nodes[mchSt].obsSucc[valI]<<" ";
            //     std::cout<<std::endl;
            // }
            // std::cout<<std::endl;
        }

        ValueFunction v(0);
        VList vl(0);
        for(int i=0;i<(int)(fsc_F.nodes.size());i++)
        {
            VEntry ve(fsc_F.nodes[i].values,fsc_F.nodes[i].action,fsc_F.nodes[i].obsSucc);
            vl.push_back(ve);
        }
        v.push_back(vl);
        std::cout<<"Final FSC Size "<<fsc_F.nodes.size()<<std::endl<<std::flush;
        // std::cout<<"------FSC_------"<<std::endl;
        // for(int mchSt=0;mchSt<(int)fsc_.nodes.size();mchSt++)
        // {
        //     std::cout<<"--"<<mchSt<<"--"<<fsc_.nodes[mchSt].action<<std::endl;
        //     for(int valI=0;valI<(int)S;valI++)
        //         std::cout<<fsc_.nodes[mchSt].values[valI]<<" ";
        //     std::cout<<std::endl;
        //     for(int valI=0;valI<(int)O;valI++)
        //         std::cout<<fsc_.nodes[mchSt].obsSucc[valI]<<" ";
        //     std::cout<<std::endl;
        // }
            
        return std::make_tuple(useTolerance ? variation : 0.0, v);
        // return std::make_tuple(fsc_F, v);
    }


///////////////////////////////////////


    template <typename M, typename>
    VList PolicyIteration::policyEvaluation(const M & model,int doEval)
    {
        S = model.getS();
        A = model.getA();
        O = model.getO();

        Eigen::Matrix<double, Eigen::Dynamic,Eigen::Dynamic> coeffMat;            
        Eigen::Matrix<double, Eigen::Dynamic, 1> valMat;
        Eigen::Matrix<double,Eigen::Dynamic, 1> solnVec;
        
        int FSCsize = fsc_.nodes.size();
        int MatDim = FSCsize*S;
        coeffMat.resize(MatDim,MatDim);
        valMat.resize(MatDim,1);

        for(int i=0;i<MatDim;i++)
        {    
            for(int j=0;j<MatDim;j++)
                coeffMat(i,j)=0;
            valMat(i,0) = 0;
        }
        // std::cout<<" chkpA1"<<std::endl<<std::flush;
        
        double discF = model.getDiscount(); 
        for(int alphaV=0;alphaV<FSCsize;alphaV++)
        {   
            // std::cout<<"  chkpA1a"<<std::endl<<std::flush;
        
            int vecAction = fsc_.nodes[alphaV].action;
            for(int iState=0;iState<(int)S;iState++)
            {
                // std::cout<<"   chkpA1aa"<<std::endl<<std::flush;
        
                int coorRow = (alphaV*S)+iState;
                for(int fstate=0;fstate<(int)S;fstate++)
                {   
                    // std::cout<<"    chkpA1aaa"<<std::endl<<std::flush;
                    
                    double pij = model.getTransitionProbability(iState,vecAction,fstate);
                    for(int ob = 0;ob<(int)O;ob++)
                    {
                        // std::cout<<"     chkpA1a4"<<std::endl<<std::flush;
        
                        int succVect = fsc_.nodes[alphaV].obsSucc[ob];
                        int coorCol = (succVect*S) + fstate;
                        double qjth = model.getObservationProbability(fstate,vecAction,ob);
                        coeffMat(coorRow,coorCol) -= (discF*pij*qjth);    
                    }
                    double rij = model.getExpectedReward(iState,vecAction,fstate);
                    // std::cout<<"    chkpA1aab"<<std::endl<<std::flush;
                    
                    valMat(coorRow,0) += (pij*rij);
                }
                coeffMat(coorRow,coorRow) += 1; 
            
                // std::cout<<"   chkpA1ab"<<std::endl<<std::flush;
            }
        }    
        // std::cout<<" chkpA2"<<std::endl<<std::flush;
        
        solnVec.resize(MatDim,1);
        solnVec = coeffMat.colPivHouseholderQr().solve(valMat);
        // std::cout<<" chkpA3"<<std::endl<<std::flush;
        
        // for(int i=0;i<MatDim;i++){
        //     for(int j=0;j<MatDim;j++)
        //         std::cout<<coeffMat(i,j)<<" ";
        //     std::cout<<std::endl;
        // }
        // std::cout<<std::endl<<"B: ";
        // for(int i=0;i<MatDim;i++)
        //     std::cout<<valMat(i,0)<<" ";
        // std::cout<<std::endl<<"X: ";
        // for(int i=0;i<MatDim;i++)
        //     std::cout<<solnVec(i,0)<<" ";
        // std::cout<<std::endl;
        
        double maxExp=fsc_.nodes[0].values.dot(bel_);
        int maxNode = 0;
        VList alpVecs = std::vector<VEntry>(FSCsize);
        for(int i=0;i<FSCsize;i++)
        {    
            alpVecs[i].observations = fsc_.nodes[i].obsSucc;
            alpVecs[i].action = fsc_.nodes[i].action;
            alpVecs[i].values.resize(S,1);
            for(int j=0;j<(int)S;j++)
            {
                int index = i*S + j;
                if(doEval == 1){    
                    alpVecs[i].values(j,0) = solnVec(index,0);    
                    fsc_.nodes[i].values[j] = solnVec(index,0);
                }
                else{
                    alpVecs[i].values(j,0) = fsc_.nodes[i].values[j];   
                }
            }
            double val = (fsc_.nodes[i].values.dot(bel_)); 
            if(maxExp < val)
            {
                maxExp = val;
                maxNode = i;
            }
        }
        fsc_.nodes[maxNode].mark = 1;
        fsc_.maxE = maxExp;
        
        // std::cout<<" chkpA4"<<std::endl<<std::flush;
        
        if(howard_ == 0)
        {            
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
                        succInd = succID;
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
        // std::cout<<" chkpA5"<<std::endl<<std::flush;
        
        return alpVecs;
    }

}

    
#endif
